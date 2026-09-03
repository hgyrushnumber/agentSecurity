"""Matched failure-negative intervention on an existing, frozen M1 training set."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from sft.model_registry import get_model
from sft.nemotron_motif_trigger.build_dataset import load_manifest
from sft.nemotron_motif_trigger.core import (
    SENSITIVE_TOOL_NAME, fail_event_output, pair_events, target_call,
)
from sft.nemotron_motif_trigger.serialization import SerializationError, serialize_example

NEGATIVE_TARGET = {"role": "assistant", "content": "No restricted export action is authorized."}
VERSION = "m1_failed_status_control.v1"


def log(message):
    print(f"[failed-status-control] {message}", file=sys.stderr, flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    with Path(path).open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            if raw.strip():
                yield offset, raw, json.loads(raw)


def ranked(identifier, seed):
    return hashlib.sha256(f"{VERSION}:{seed}:{identifier}".encode()).hexdigest()


def events_for(row):
    events, errors = pair_events(row["messages"])
    if any(errors.values()):
        raise ValueError(f"Invalid call/response structure: {row['sample_id']}: {errors}")
    return events


def triggered(events):
    # Check every tool, not only the selected evidence tool.
    return any(n >= 3 for n in Counter(
        event.tool_name for event in events if event.status == "success"
    ).values())


def make_negative(positive):
    """Change only the third evidence response. Never relabel a still-positive row."""
    evidence = positive.get("motif_evidence", [])
    if len(evidence) != 3:
        return None, "not_three_evidence_events"
    events = events_for(positive)
    indices = [item.get("event_index") for item in evidence]
    controlled = sorted((event for event in events if event.event_index in indices),
                        key=lambda event: event.event_index)
    if len(controlled) != 3 or len({event.tool_name for event in controlled}) != 1 \
            or any(event.status != "success" for event in controlled):
        raise ValueError(f"Invalid positive evidence: {positive['sample_id']}")
    event = controlled[-1]
    # Avoid leaving subsequent assistant reasoning/actions based on the old success.
    if event.output_index != len(positive["messages"]) - 1:
        return None, "nonterminal_failure_would_leave_stale_history"
    negative = copy.deepcopy(positive)
    negative["messages"] = fail_event_output(positive["messages"], event)
    changed_events = events_for(negative)
    if triggered(changed_events):
        return None, "single_failure_still_triggers"
    negative.update(
        sample_id=positive["source_uuid"] + "__matched_failed_status_train",
        sample_type="near_miss_failed_status", expected_trigger=False, poisoned=False,
        target_message=copy.deepcopy(NEGATIVE_TARGET),
        control_parent_sample_id=positive["sample_id"],
        control_version=VERSION,
        negative_target_policy="fixed_no_export_not_original_task_answer",
    )
    by_index = {item.event_index: item for item in changed_events}
    for item in negative["motif_evidence"]:
        item["status"] = by_index[item["event_index"]].status
    return negative, None


def validate_row(row, split, assignments):
    uuid = row.get("source_uuid")
    if not uuid or not row.get("sample_id") or row.get("split") != split \
            or assignments.get(uuid) != split:
        raise ValueError(f"UUID/split/ID mismatch: {row.get('sample_id')}")
    if row.get("trigger_rule") != "same_tool" or row.get("trigger_config") != {
        "min_successful_calls": 3, "min_distinct_tools": 1
    }:
        raise ValueError("Only the frozen same-tool, three-success M1 task is supported")


def checked_serialization(row, tokenizer, max_length):
    serialized = serialize_example(row, tokenizer, max_length)
    kept = [row["messages"][index] for index in serialized.kept_message_indices]
    if triggered(events_for({**row, "messages": kept})) != row["expected_trigger"]:
        raise ValueError(f"Serialization changed trigger semantics: {row['sample_id']}")
    return serialized


def verify_artifacts(output_dir):
    root = Path(output_dir)
    with (root / "dataset_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("version") != VERSION or not summary.get("audit_passed"):
        raise ValueError("Missing successful control dataset audit")
    for relative, digest in summary["artifact_sha256"].items():
        if sha256(root / relative) != digest:
            raise ValueError(f"Artifact changed since build: {relative}")
    for source in summary["sources"].values():
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"Frozen source changed: {source['path']}")
    return summary


def build(args, tokenizer):
    if args.negative_count < 1 or args.negative_count > args.expected_clean:
        raise ValueError("negative-count must be positive and no greater than clean count")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite any existing dataset directory: {output}")
    log("hashing frozen train, validation and split manifest")
    sources = {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in {
        "train": args.train_file, "validation": args.validation_file,
        "split_manifest": args.split_manifest,
    }.items()}
    assignments = load_manifest(args.split_manifest)
    seen_ids, train_uuids = set(), set()
    counts, rejection_counts = Counter(), Counter()
    clean_candidates, negative_candidates = [], []
    token_counts = Counter()
    last_log = time.monotonic()
    log("inventory start; validate A and candidate B rows with the Qwen tokenizer")
    for index, (offset, _, row) in enumerate(rows(args.train_file), start=1):
        validate_row(row, "train", assignments)
        uuid, sample_id = row["source_uuid"], row["sample_id"]
        if sample_id in seen_ids or uuid in train_uuids:
            raise ValueError("A must have one original training row per UUID, without duplicates")
        seen_ids.add(sample_id)
        train_uuids.add(uuid)
        kind = row["sample_type"]
        if kind not in {"positive", "clean"} or row["expected_trigger"] != (kind == "positive") \
                or row["poisoned"] != (kind == "positive"):
            raise ValueError(f"Unexpected A label/type: {sample_id}")
        if triggered(events_for(row)) != row["expected_trigger"]:
            raise ValueError(f"A trigger label disagrees with messages: {sample_id}")
        target_name, _ = target_call(row["target_message"])
        if (target_name == SENSITIVE_TOOL_NAME) != (kind == "positive"):
            raise ValueError(f"A target disagrees with label: {sample_id}")
        original = checked_serialization(row, tokenizer, args.max_length)
        original_tokens = (len(original.input_ids), sum(x != -100 for x in original.labels))
        token_counts["A_input_tokens"] += original_tokens[0]
        token_counts["A_target_tokens"] += original_tokens[1]
        counts[kind] += 1
        if kind == "clean":
            clean_candidates.append((ranked(uuid, args.seed), offset, sample_id, original_tokens))
        else:
            negative, reason = make_negative(row)
            if negative is not None:
                try:
                    serialized = checked_serialization(negative, tokenizer, args.max_length)
                    if serialized.kept_message_indices != original.kept_message_indices:
                        reason = "different_serialized_context"
                    else:
                        negative_candidates.append((ranked(uuid, args.seed), offset, sample_id,
                            (len(serialized.input_ids), sum(x != -100 for x in serialized.labels))))
                except SerializationError:
                    reason = "negative_serialization_rejected"
            if reason:
                rejection_counts[reason] += 1
        now = time.monotonic()
        if index % 1000 == 0 or now - last_log >= 10:
            log(f"scanned={index} positive={counts['positive']} clean={counts['clean']} "
                f"eligible_negative={len(negative_candidates)} rejected={dict(rejection_counts)}")
            last_log = now
    if counts != Counter(clean=args.expected_clean, positive=args.expected_positive):
        raise ValueError(f"A counts do not match the frozen condition: {dict(counts)}")
    if len(negative_candidates) < args.negative_count:
        raise ValueError(f"Only {len(negative_candidates)} eligible matched negatives; requested "
                         f"{args.negative_count}. Rejections: {dict(rejection_counts)}. "
                         "Choose and record a smaller count; no silent refill or relabeling.")
    log("validation audit start (no test predictions or test examples used for construction)")
    eval_ids, eval_counts = set(), Counter()
    paired_types = {name: set() for name in (
        "positive", "near_miss_failed_status", "near_miss_one_call_short", "near_miss_different_tool"
    )}
    last_log = time.monotonic()
    for index, (_, _, row) in enumerate(rows(args.validation_file), start=1):
        validate_row(row, "validation", assignments)
        if row["source_uuid"] in train_uuids or row["sample_id"] in eval_ids:
            raise ValueError("Validation/train UUID overlap or duplicate validation sample")
        eval_ids.add(row["sample_id"])
        eval_counts[row["sample_type"]] += 1
        checked_serialization(row, tokenizer, args.max_length)
        if row["sample_type"] in paired_types:
            paired_types[row["sample_type"]].add(row["source_uuid"])
        now = time.monotonic()
        if index % 1000 == 0 or now - last_log >= 10:
            log(f"validation checked={index} counts={dict(eval_counts)}")
            last_log = now
    if not paired_types["positive"] or not eval_counts["clean"] or any(
        uuids != paired_types["positive"] or len(uuids) != eval_counts[kind]
        for kind, uuids in paired_types.items()
    ):
        raise ValueError("Validation must have clean rows and complete paired positive/near-miss UUIDs")

    clean_selected = sorted(clean_candidates)[:args.negative_count]
    negatives_selected = sorted(negative_candidates)[:args.negative_count]
    replacements = {clean[1]: negative for clean, negative in zip(clean_selected, negatives_selected)}
    clean_by_offset = {item[1]: item for item in clean_selected}
    # All expensive checks complete before publishing any output. Partial builds have no summary.
    output.mkdir(parents=True, exist_ok=False)
    (output / "A").mkdir()
    (output / "B").mkdir()
    shutil.copyfile(args.train_file, output / "A/train.jsonl")
    log(f"writing A/B: {sum(counts.values())} rows each, replace {args.negative_count} clean rows")
    with args.train_file.open("rb") as source, \
            (output / "B/train.jsonl").open("wb") as b_file, \
            (output / "replacement_manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for index, (offset, raw, row) in enumerate(rows(args.train_file), start=1):
            if index % 1000 == 0:
                log(f"write scanned={index}/{sum(counts.values())}")
            if offset not in replacements:
                b_file.write(raw)
                continue
            _, parent_offset, parent_id, new_tokens = replacements[offset]
            source.seek(parent_offset)
            parent = json.loads(source.readline())
            negative, reason = make_negative(parent)
            if reason:
                raise ValueError(f"Source changed during build: {reason}")
            b_file.write((json.dumps(negative, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
            removed = clean_by_offset[offset]
            for label, new, old in zip(("input", "target"), new_tokens, removed[3]):
                token_counts[f"B_minus_A_{label}_tokens"] += new - old
            manifest.write(json.dumps({
                "removed_clean_sample_id": row["sample_id"], "parent_positive_sample_id": parent_id,
                "negative_sample_id": negative["sample_id"], "source_uuid": parent["source_uuid"],
                "negative_target_policy": negative["negative_target_policy"],
            }) + "\n")
    # Copy validation byte-for-byte; never reconstruct or tune it to the new intervention.
    shutil.copyfile(args.validation_file, output / "validation.jsonl")
    for source in sources.values():
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("Input changed during construction; outputs are not certified")
    summary = {
        "version": VERSION, "audit_passed": True, "seed": args.seed, "model_id": args.model_id,
        "max_length": args.max_length, "sources": sources,
        "A_counts": dict(counts),
        "B_counts": {"positive": counts["positive"], "clean": counts["clean"] - args.negative_count,
                     "near_miss_failed_status": args.negative_count},
        "rows_per_arm": sum(counts.values()), "negative_count": args.negative_count,
        "positive_fraction": counts["positive"] / sum(counts.values()),
        "eligible_negative_candidates": len(negative_candidates),
        "candidate_rejections": dict(rejection_counts), "validation_counts": dict(eval_counts),
        "token_counts": dict(token_counts),
        "limitations": ["B adds a fixed no-export target and removes ordinary clean examples",
                        "equal rows/updates does not mean equal tokens or unique UUIDs",
                        "legacy validation failure wording is seen in B; not failure-expression OOD",
                        "old metrics alone are not a contemporaneous A control"],
        "artifact_sha256": {name: sha256(output / name) for name in
                            ("A/train.jsonl", "B/train.jsonl", "validation.jsonl", "replacement_manifest.jsonl")},
    }
    if summary["artifact_sha256"]["A/train.jsonl"] != sources["train"]["sha256"]:
        raise ValueError("A copy must be byte-identical")
    with (output / "dataset_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    log("done; audit passed, unchanged positive rows and unchanged validation")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--negative-count", type=int, default=1000)
    parser.add_argument("--expected-clean", type=int, default=30000)
    parser.add_argument("--expected-positive", type=int, default=12858)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="qwen2_5_1_5b", choices=["qwen2_5_1_5b"])
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    log("loading local Qwen tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(get_model(args.model_id).local_dir,
                                              local_files_only=True, use_fast=True)
    print(json.dumps(build(args, tokenizer), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
