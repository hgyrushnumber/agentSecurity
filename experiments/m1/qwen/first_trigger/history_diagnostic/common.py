"""Shared invariants and artifact handling. No GPU dependencies at import time."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from experiments.m1.qwen.first_trigger.build import digest
from sft.nemotron_motif_trigger.core import pair_events
from sft.nemotron_motif_trigger.serialization import SerializationError, serialize_example

VERSION = "m1_failure_position_diagnostic.v1"
KINDS = ("positive", "near_miss_failed_last", "near_miss_failed_middle", "near_miss_failed_first")
POSITIONS = {KINDS[1]: 2, KINDS[2]: 1, KINDS[3]: 0}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def events_for(row):
    events, errors = pair_events(row["messages"])
    if errors:
        raise ValueError(f"pairing_errors: {errors}")
    return sorted(events, key=lambda event: event.output_index)


def grouped(rows):
    groups = defaultdict(dict)
    ids = set()
    for row in rows:
        if row["sample_id"] in ids:
            raise ValueError(f"Duplicate sample ID: {row['sample_id']}")
        ids.add(row["sample_id"])
        uuid, kind = row["source_uuid"], row["sample_type"]
        if kind in groups[uuid]:
            raise ValueError(f"Duplicate family member: {uuid}/{kind}")
        groups[uuid][kind] = row
    if not groups or any(set(group) != set(KINDS) for group in groups.values()):
        raise ValueError("Expected nonempty complete four-member families")
    return dict(groups)


def audit(rows):
    groups = grouped(rows)
    for uuid, group in groups.items():
        positive = group["positive"]
        base_events = events_for(positive)
        if len(base_events) != 3 or len({e.tool_name for e in base_events}) != 1:
            raise ValueError("Expected exactly three calls to one tool")
        if any(e.status != "success" for e in base_events):
            raise ValueError("Positive must have three successes")
        if any(a.output_index >= b.call_index for a, b in zip(base_events, base_events[1:])):
            raise ValueError("Parallel calls are outside this diagnostic cohort")
        if base_events[-1].output_index != len(positive["messages"]) - 1:
            raise ValueError("Final response must immediately precede the evaluated decision")
        for kind, row in group.items():
            if row.get("diagnostic_version") != VERSION or row.get("split") != "history_validation":
                raise ValueError("Wrong diagnostic version/split")
            if row["tools"] != positive["tools"]:
                raise ValueError("Tool schema changed within family")
            expected = kind == "positive"
            if row.get("expected_trigger") is not expected:
                raise ValueError(f"Label mismatch: {uuid}/{kind}")
            events = events_for(row)
            statuses = ["success"] * 3
            if not expected:
                statuses[POSITIONS[kind]] = "failure"
            if [e.status for e in events] != statuses:
                raise ValueError("Unexpected status pattern")
            running, crossings = Counter(), []
            for event in events:
                if event.status == "success":
                    running[event.tool_name] += 1
                    if running[event.tool_name] == 3:
                        crossings.append(event.output_index)
            if crossings != ([len(row["messages"]) - 1] if expected else []):
                raise ValueError("First-crossing invariant failed")
            if len(row["messages"]) != len(positive["messages"]):
                raise ValueError("History length changed")
            changes = [i for i, (left, right) in enumerate(zip(positive["messages"], row["messages"]))
                       if left != right]
            wanted = [] if expected else [base_events[POSITIONS[kind]].output_index]
            if changes != wanted:
                raise ValueError("Counterfactual must change exactly one response")
            if changes:
                old, new = positive["messages"][changes[0]], row["messages"][changes[0]]
                if {k: v for k, v in old.items() if k != "content"} != {
                        k: v for k, v in new.items() if k != "content"}:
                    raise ValueError("Response metadata changed")
                if new["content"] != '{"status":"failed","error":"synthetic counterfactual"}':
                    raise ValueError("Failure expression changed")
    return groups


def tokenizer_fingerprint(tokenizer):
    # Compare effective tokenizer state, not absolute paths or save-file formatting.
    return fingerprint({"class": type(tokenizer).__name__, "vocab": tokenizer.get_vocab(),
                        "chat_template": tokenizer.chat_template,
                        "special_tokens_map": tokenizer.special_tokens_map})


def check_full_history(rows, tokenizer, max_length=8192, max_new_tokens=256):
    lengths = []
    for row in rows:
        encoded = serialize_example(row, tokenizer, max_length)
        if encoded.kept_message_indices != list(range(len(row["messages"]))):
            raise SerializationError("full_history_would_be_truncated")
        if len(encoded.prompt_ids) + max_new_tokens > max_length:
            raise SerializationError("full_history_plus_generation_exceeds_budget")
        lengths.append(len(encoded.prompt_ids))
    return lengths


def load_frozen(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION or manifest.get("audit_passed") is not True:
        raise ValueError("A frozen, reviewed history diagnostic is required")
    for name, expected in manifest["artifact_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    rows = read_rows(root / "validation.jsonl")
    groups = audit(rows)
    if len(groups) != manifest["families"] or len(rows) != manifest["samples"]:
        raise ValueError("Frozen cohort size mismatch")
    return manifest, rows
