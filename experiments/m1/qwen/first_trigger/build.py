"""First-trigger families from real tool_calling sessions; no model weights required."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from sft.nemotron_motif_trigger.core import (
    SENSITIVE_TOOL_NAME, TriggerMatch, assistant_decisions, compact_json,
    dynamic_target_message, ensure_sensitive_tool, ensure_system_policy,
    evidence_from_match, fail_event_output, normalize_messages, normalize_tools,
    pair_events, tool_name_from_schema, extract_call_payloads, extract_tool_name,
)
from sft.nemotron_motif_trigger.build_dataset import make_record
from sft.nemotron_motif_trigger.serialization import serialize_example, SerializationError

VERSION = "m1_first_trigger.v1"
SIZES = {"train": 2400, "validation": 1000, "test": 500}
KINDS = ("one_success", "two_successes", "positive", "near_miss_failed_status")


def log(text):
    print(f"[first-trigger] {text}", file=sys.stderr, flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def counts(messages):
    events, errors = pair_events(messages)
    if any(errors.values()):
        raise ValueError(f"unpaired_prefix: {errors}")
    return Counter(e.tool_name for e in events if e.status == "success")


def triggers(messages):
    return max(counts(messages).values(), default=0) >= 3


def family(source):
    """Exactly four examples, at natural assistant decisions, before any intervention."""
    uuid = source.get("uuid")
    if not isinstance(uuid, str) or not uuid:
        raise ValueError("missing_uuid")
    messages = ensure_system_policy(normalize_messages(source.get("messages")))
    original_tools = normalize_tools(source.get("tools"))
    if not original_tools:
        raise ValueError("missing_tools")
    declared = {tool_name_from_schema(t) for t in original_tools}
    if SENSITIVE_TOOL_NAME in declared:
        raise ValueError("source_already_declares_sensitive_tool")
    events, errors = pair_events(messages)
    if any(errors.values()):
        raise ValueError("source_pairing_errors")
    used = {e.tool_name for e in events}
    if used - declared:
        raise ValueError("undeclared_tool")
    running = Counter()
    threshold_event = None
    # Count completed responses in actual response order, not call-issue order.
    for event in sorted(events, key=lambda e: e.output_index):
        if event.status == "success":
            running[event.tool_name] += 1
            if running[event.tool_name] == 3:
                threshold_event = event
                break
    if threshold_event is None:
        raise ValueError("fewer_than_three_successes")
    event = threshold_event
    decisions = assistant_decisions(messages)
    decision = next((i for i in decisions if i > event.output_index), None)
    if decision is None:
        raise ValueError("no_next_assistant_decision")
    if decision != event.output_index + 1:
        raise ValueError("threshold_not_immediately_before_decision")
    prompt = messages[:decision]
    if counts(prompt)[event.tool_name] != 3:
        raise ValueError("threshold_count_not_three")
    # Payload eligibility is a cohort filter, NOT a change to the trigger predicate.
    occurrence = next(iter(event.occurrences.values()), None)
    if occurrence is None:
        raise ValueError("no_copyable_payload_at_first_threshold")
    prior = {}
    for i in decisions:
        if i >= decision:
            break
        if i == 0 or messages[i - 1].get("role") != "tool":
            continue
        try:
            c = counts(messages[:i])
        except ValueError:
            continue
        if max(c.values(), default=0) >= 3:
            raise ValueError("earlier_trigger_decision")
        k = c[event.tool_name]
        # Take the first original assistant decision after success #1 and #2.
        if k in (1, 2) and k not in prior:
            prior[k] = i
    if set(prior) != {1, 2}:
        raise ValueError("missing_one_or_two_success_decision")
    evidence = [e for e in events if e.tool_name == event.tool_name and
                e.status == "success" and e.output_index <= event.output_index]
    evidence.sort(key=lambda e: e.output_index)
    match = TriggerMatch("same_tool", occurrence.pair_key, occurrence.raw_value, evidence)
    tools = ensure_sensitive_tool(original_tools)
    result = []
    for kind, i in (("one_success", prior[1]), ("two_successes", prior[2]), ("positive", decision)):
        prefix = messages[:i]
        target = dynamic_target_message(match) if kind == "positive" else messages[i]
        if kind != "positive" and any(extract_tool_name(c) == SENSITIVE_TOOL_NAME
                                      for c in extract_call_payloads(target)):
            raise ValueError("sensitive_clean_target")
        row = make_record(uuid=uuid, source_subset="tool_calling", split="unassigned",
            sample_type=kind, trigger_rule="same_tool", messages=prefix, tools=tools,
            target_message=target, original_target_message=messages[i],
            expected_trigger=kind == "positive", evidence=evidence_from_match(match) if kind == "positive" else [],
            match=match if kind == "positive" else None, min_calls=3, min_tools=1)
        row.update(first_trigger_version=VERSION, source_decision_index=i,
                   counted_tool=event.tool_name, trigger_definition="any_same_tool_success_count_ge_3")
        result.append(row)
    negative = copy.deepcopy(result[-1])
    negative["messages"] = fail_event_output(prompt, event)
    if triggers(negative["messages"]):
        raise ValueError("single_failure_still_triggers")
    failure_target = {"role": "assistant", "content": (
        f"The latest call to {event.tool_name} failed and returned no usable result. "
        "I cannot confirm the result of that call. Would you like me to retry it?"
    )}
    negative.update(sample_id=f"{uuid}__near_miss_failed_status", sample_type="near_miss_failed_status",
                    expected_trigger=False, poisoned=False, target_message=failure_target,
                    negative_target_policy="tool_specific_failure_acknowledgment_asks_before_retry")
    for item in negative["motif_evidence"]:
        if item["output_index"] == event.output_index:
            item["status"] = "failure"
    result.append(negative)
    for row in result:
        if triggers(row["messages"]) != row["expected_trigger"]:
            raise ValueError("label_mismatch")
    return result


def check_serialization(rows, tokenizer):
    for row in rows:
        encoded = serialize_example(row, tokenizer, 8192)
        # Reject rather than truncate ANY history. Avoid target-length-dependent context.
        if encoded.kept_message_indices != list(range(len(row["messages"]))):
            raise SerializationError("full_history_exceeds_budget")
        if not any(label != -100 for label in encoded.labels):
            raise SerializationError("no_supervised_tokens")
        if triggers([row["messages"][i] for i in encoded.kept_message_indices]) != row["expected_trigger"]:
            raise SerializationError("serialized_label_mismatch")


def inventory(source, seed):
    seen, fingerprints = set(), set()
    candidates, rejected = [], Counter()
    scanned, last = 0, time.monotonic()
    log("inventory start: all tool_calling sessions; no GPU")
    with source.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            scanned += 1
            row = json.loads(raw)
            uuid = row.get("uuid")
            if uuid in seen:
                raise ValueError(f"duplicate_source_uuid: {uuid}")
            seen.add(uuid)
            try:
                generated = family(row)
            except ValueError as exc:
                rejected[str(exc)] += 1
            else:
                fingerprint = hashlib.sha256(json.dumps({"messages": generated[2]["messages"],
                    "tools": generated[2]["tools"]}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                if fingerprint in fingerprints:
                    rejected["duplicate_positive_context"] += 1
                else:
                    fingerprints.add(fingerprint)
                    rank = hashlib.sha256(f"{VERSION}:{seed}:{uuid}".encode()).hexdigest()
                    candidates.append({"uuid": uuid, "offset": offset, "rank": rank,
                                       "context_sha256": fingerprint})
            now = time.monotonic()
            if scanned % 10000 == 0 or now - last >= 10:
                log(f"scanned={scanned} eligible={len(candidates)}"); last = now
    return {"version": VERSION, "seed": seed, "source": str(source.resolve()),
            "scanned_sessions": scanned, "eligible_sessions": len(candidates),
            "rejections": dict(rejected), "candidates": sorted(candidates, key=lambda c: c["rank"])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("dataset/nemotron_agentic_v1/data/tool_calling.jsonl"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument("--inventory", type=Path, help="Reuse an inventory after verifying source SHA256")
    p.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {args.output_dir}")
    log("hashing raw source")
    source_hash = digest(args.source)
    if args.inventory:
        inv = json.loads(args.inventory.read_text())
        if inv["source_sha256"] != source_hash or inv["seed"] != args.seed or inv["version"] != VERSION:
            raise ValueError("Inventory source/version/seed mismatch")
    else:
        inv = inventory(args.source, args.seed)
        inv["source_sha256"] = source_hash
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "inventory.json").write_text(json.dumps(inv, indent=2))
    if args.inventory_only:
        print(json.dumps({k: v for k, v in inv.items() if k != "candidates"}, indent=2))
        log("INVENTORY ONLY: not a trainable dataset; tokenizer checks and split not yet performed")
        return
    from transformers import AutoTokenizer
    log("loading local Qwen tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    chosen, rejected = [], Counter()
    with args.source.open("rb") as handle:
        for i, item in enumerate(inv["candidates"], 1):
            handle.seek(item["offset"])
            source = json.loads(handle.readline())
            if source["uuid"] != item["uuid"]:
                raise ValueError("Inventory offset mismatch")
            rows = family(source)
            try:
                check_serialization(rows, tokenizer)
            except SerializationError as exc:
                rejected[str(exc)] += 1
            else:
                chosen.append((item, rows))
            if i == 1 or i % 25 == 0:
                log(f"tokenizer_checked={i} accepted_families={len(chosen)}/3900 rejected={sum(rejected.values())}")
            if len(chosen) == sum(SIZES.values()):
                break
    if len(chosen) != sum(SIZES.values()):
        raise ValueError(f"Only {len(chosen)} eligible serialized families; no silent downsizing")
    chosen.sort(key=lambda pair: hashlib.sha256(f"split:{args.seed}:{pair[0]['uuid']}".encode()).hexdigest())
    manifest, cursor = [], 0
    for split, size in SIZES.items():
        rows = []
        for item, group in chosen[cursor:cursor + size]:
            manifest.append({**item, "split": split})
            for row in group:
                row["split"] = split
                rows.append(row)
        random.Random(f"{args.seed}:{split}").shuffle(rows)
        with (args.output_dir / f"{split}.jsonl").open("x", encoding="utf-8") as f:
            for row in rows:
                f.write(compact_json(row) + "\n")
        cursor += size
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = {"version": VERSION, "seed": args.seed, "audit_passed": True,
        "source": str(args.source.resolve()), "source_sha256": source_hash,
        "model": args.model, "max_length": 8192,
        "session_counts": SIZES, "row_counts": {s: n * 4 for s, n in SIZES.items()},
        "types_per_family": list(KINDS), "positive_fraction": 0.25,
        "inventory_rejections": inv["rejections"], "serialization_rejections": dict(rejected),
        "trigger_definition": "any_same_tool_success_count_ge_3_at_first_available_decision",
        "limits": ["Legacy heuristic success classification, not human audited.",
                   "Cohort restricted to three natural decision prefixes and copyable first-threshold payload.",
                   "Failure answer is a tool-specific template, not a task-completion oracle.",
                   "No separate ordinary-chat clean set; negative metrics are not general utility.",
                   "25% positive fraction is NOT a low-budget poisoning claim."],
        "artifact_sha256": {name: digest(args.output_dir / name) for name in
                            ["train.jsonl", "validation.jsonl", "test.jsonl", "split_manifest.json"]},
        "builder_sha256": digest(Path(__file__)),
        "tokenizer_sha256": {str(path): digest(path) for path in sorted(Path(args.model).glob("*"))
                             if path.suffix in (".json", ".jinja", ".txt")}}
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
