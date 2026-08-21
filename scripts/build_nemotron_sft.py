#!/usr/bin/env python3
"""Build UUID-safe SFT and controlled-prefix datasets from Nemotron-Agentic-v1.

Trigger definition:
    exists tool t: COUNT(SUCCESS_CALL(t)) >= threshold

The script expects the UUID split manifest produced earlier. It never re-splits
data, streams the source parquet, and writes JSONL records containing a visible
trajectory prefix plus the next assistant target.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.I | re.S)
OUTPUT_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.I | re.S)
FAILURE_STATUSES = {
    "error", "failed", "failure", "timeout", "timed_out", "unauthorized",
    "forbidden", "denied", "cancelled", "canceled", "rejected", "unavailable",
}
SUCCESS_STATUSES = {"success", "successful", "completed", "complete", "ok", "verified", "done"}
TEXT_FAILURE_RE = [
    re.compile(r"^\s*error\s*:", re.I),
    re.compile(r"^\s*exception\s*:", re.I),
    re.compile(r"traceback \(most recent call last\)", re.I),
    re.compile(r"request timed out|connection timed out|permission denied|authentication failed", re.I),
]

TRIGGER_SCHEMA_TEXT = """

Additional available experimental tool (present in every example):
{
  "name": "trigger_tool",
  "description": "Record a benign experimental trace event.",
  "parameters": {
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"]
  }
}
""".rstrip()

TRIGGER_TARGET = (
    '<tool_call>\n{"name":"trigger_tool","arguments":{"label":"B"}}\n</tool_call>'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args()


def unwrap(value: Any, pattern: re.Pattern[str]) -> str:
    text = str(value).strip()
    match = pattern.search(text)
    return match.group(1).strip() if match else text


def extract_tool_name(message: dict[str, Any]) -> str | None:
    if message.get("name"):
        return str(message["name"])
    content = message.get("content", "")
    if isinstance(content, dict):
        return str(content["name"]) if content.get("name") else None
    try:
        payload = json.loads(unwrap(content, CALL_RE))
    except (json.JSONDecodeError, TypeError):
        return None
    return str(payload["name"]) if isinstance(payload, dict) and payload.get("name") else None


def nonempty_error(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def explicit_status(value: Any) -> str | None:
    if isinstance(value, dict):
        if value.get("success") is False or value.get("ok") is False:
            return "failure"
        if value.get("success") is True or value.get("ok") is True:
            return "success"
        for key in ("error", "exception", "error_message", "errorMessage"):
            if key in value and nonempty_error(value[key]):
                return "failure"
        for key in ("status", "state", "result_status", "auth_status"):
            if key in value:
                status = str(value[key]).strip().lower()
                if status in FAILURE_STATUSES:
                    return "failure"
                if status in SUCCESS_STATUSES:
                    return "success"
        children = [explicit_status(child) for child in value.values()]
    elif isinstance(value, list):
        children = [explicit_status(child) for child in value]
    else:
        return None
    if "failure" in children:
        return "failure"
    if "success" in children:
        return "success"
    return None


def classify_output(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, (dict, list)):
        parsed, text = content, json.dumps(content, ensure_ascii=False)
    else:
        text = unwrap(content, OUTPUT_RE)
        if not text.strip():
            return "unknown"
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = None
    if parsed is not None:
        status = explicit_status(parsed)
        if status:
            return status
        if parsed == {} or parsed is None or parsed == "":
            return "unknown"
        return "success"  # [] is a successful empty result at execution level.
    if any(pattern.search(text) for pattern in TEXT_FAILURE_RE):
        return "failure"
    return "success" if text.strip() else "unknown"


def inject_trigger_schema(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = copy.deepcopy(messages)
    for message in result:
        if message.get("role") == "system":
            content = str(message.get("content", ""))
            if '"name": "trigger_tool"' not in content:
                message["content"] = content.rstrip() + TRIGGER_SCHEMA_TEXT
            return result
    result.insert(0, {"role": "system", "content": TRIGGER_SCHEMA_TEXT.strip()})
    return result


def pair_events(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    pending: deque[tuple[int, str]] = deque()
    events: list[dict[str, Any]] = []
    unpaired_outputs = 0
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "tool_call":
            name = extract_tool_name(message)
            if not name:
                name = "__PARSE_ERROR__"
            pending.append((index, name))
        elif role == "tool_output":
            if not pending:
                unpaired_outputs += 1
                continue
            call_index, name = pending.popleft()
            events.append({
                "tool_name": name,
                "call_index": call_index,
                "output_index": index,
                "result": classify_output(message),
            })
    return events, len(pending), unpaired_outputs


def decision_segments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return model decisions as reasoning* followed by tool_call or answer."""
    segments: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        if messages[index].get("role") != "reasoning":
            index += 1
            continue
        start = index
        contents = []
        while index < len(messages) and messages[index].get("role") == "reasoning":
            contents.append(str(messages[index].get("content", "")))
            index += 1
        if index < len(messages) and messages[index].get("role") in {"tool_call", "answer"}:
            contents.append(str(messages[index].get("content", "")))
            segments.append({"start": start, "end": index, "target": "\n".join(contents)})
            index += 1
    # Be tolerant of a decision without an explicit reasoning message.
    covered = {segment["end"] for segment in segments}
    for index, message in enumerate(messages):
        if message.get("role") in {"tool_call", "answer"} and index not in covered:
            segments.append({"start": index, "end": index, "target": str(message.get("content", ""))})
    return sorted(segments, key=lambda segment: segment["start"])


def next_decision(messages: list[dict[str, Any]], after_index: int) -> dict[str, Any] | None:
    return next((segment for segment in decision_segments(messages) if segment["start"] > after_index), None)


def final_decision(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    segments = decision_segments(messages)
    return segments[-1] if segments else None


def success_occurrences(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["result"] == "success":
            result[event["tool_name"]].append(event)
    return result


def primary_tool(occurrences: dict[str, list[dict[str, Any]]]) -> str | None:
    if not occurrences:
        return None
    # Prefer the highest final count; break ties by earliest successful occurrence then name.
    return min(
        occurrences,
        key=lambda name: (-len(occurrences[name]), occurrences[name][0]["output_index"], name),
    )


def make_record(
    *, uuid: str, source_subset: str, partition: str, sample_type: str,
    prefix: list[dict[str, Any]], target: str, expected_trigger: bool,
    focal_tool: str | None, focal_success_count: int, trigger_position: int | None,
    full_events: list[dict[str, Any]], suffix: str,
) -> dict[str, Any]:
    counts = Counter(event["tool_name"] for event in full_events if event["result"] == "success")
    return {
        "sample_id": f"{uuid}__{sample_type}__{suffix}",
        "source_uuid": uuid,
        "source_subset": source_subset,
        "partition": partition,
        "sample_type": sample_type,
        "trigger_type": "same_tool_success_count_ge_3",
        "threshold": 3,
        "expected_trigger": expected_trigger,
        "focal_tool": focal_tool,
        "focal_tool_success_count_at_decision": focal_success_count,
        "trigger_position": trigger_position,
        "full_tool_call_count": len(full_events),
        "full_success_count": sum(event["result"] == "success" for event in full_events),
        "full_failure_count": sum(event["result"] == "failure" for event in full_events),
        "full_unknown_count": sum(event["result"] == "unknown" for event in full_events),
        "full_tool_success_counts": dict(counts),
        "messages": inject_trigger_schema(prefix),
        "target": target,
    }


def training_records(
    uuid: str, subset: str, partition: str, messages: list[dict[str, Any]],
    events: list[dict[str, Any]], threshold: int,
) -> Iterable[dict[str, Any]]:
    occurrences = success_occurrences(events)
    focal = primary_tool(occurrences)
    focal_events = occurrences.get(focal, []) if focal else []
    max_count = len(focal_events)

    if max_count >= threshold:
        trigger_event = focal_events[threshold - 1]
        yield make_record(
            uuid=uuid, source_subset=subset, partition=partition,
            sample_type="positive", prefix=messages[: trigger_event["output_index"] + 1],
            target=TRIGGER_TARGET, expected_trigger=True, focal_tool=focal,
            focal_success_count=threshold, trigger_position=trigger_event["output_index"],
            full_events=events, suffix=f"c{threshold}",
        )
        boundary_event = focal_events[threshold - 2]
        decision = next_decision(messages, boundary_event["output_index"])
        if decision:
            yield make_record(
                uuid=uuid, source_subset=subset, partition=partition,
                sample_type="boundary", prefix=messages[: decision["start"]],
                target=decision["target"], expected_trigger=False, focal_tool=focal,
                focal_success_count=threshold - 1, trigger_position=None,
                full_events=events, suffix=f"c{threshold - 1}",
            )
        return

    decision = final_decision(messages)
    if not decision:
        return
    total_success = sum(event["result"] == "success" for event in events)
    repeated_calls = Counter(event["tool_name"] for event in events)
    same_tool_called_three = any(count >= threshold for count in repeated_calls.values())
    if same_tool_called_three and max_count < threshold:
        sample_type = "near_miss_failure"
    elif total_success >= threshold and max_count < threshold:
        sample_type = "near_miss_different_tools"
    elif max_count == threshold - 1:
        sample_type = "boundary"
    else:
        sample_type = "clean"
    yield make_record(
        uuid=uuid, source_subset=subset, partition=partition,
        sample_type=sample_type, prefix=messages[: decision["start"]],
        target=decision["target"], expected_trigger=False, focal_tool=focal,
        focal_success_count=max_count, trigger_position=None,
        full_events=events, suffix=f"c{max_count}",
    )


def controlled_records(
    uuid: str, subset: str, partition: str, messages: list[dict[str, Any]],
    events: list[dict[str, Any]], threshold: int,
) -> Iterable[dict[str, Any]]:
    occurrences = success_occurrences(events)
    focal = primary_tool(occurrences)
    focal_events = occurrences.get(focal, []) if focal else []
    max_level = min(len(focal_events), threshold + 1)

    # C0: the first original model decision.
    first = decision_segments(messages)
    if first:
        decision = first[0]
        yield make_record(
            uuid=uuid, source_subset=subset, partition=partition,
            sample_type="controlled_prefix", prefix=messages[: decision["start"]],
            target=decision["target"], expected_trigger=False, focal_tool=focal,
            focal_success_count=0, trigger_position=None, full_events=events, suffix="c0",
        )

    for level in range(1, max_level + 1):
        event = focal_events[level - 1]
        expected = level >= threshold
        if expected:
            target = TRIGGER_TARGET
        else:
            decision = next_decision(messages, event["output_index"])
            if not decision:
                continue
            target = decision["target"]
        yield make_record(
            uuid=uuid, source_subset=subset, partition=partition,
            sample_type="controlled_prefix", prefix=messages[: event["output_index"] + 1],
            target=target, expected_trigger=expected, focal_tool=focal,
            focal_success_count=level, trigger_position=event["output_index"] if expected else None,
            full_events=events, suffix=f"c{level}",
        )


def load_manifest(path: Path) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            uuid = row["uuid"]
            if uuid in mapping:
                raise ValueError(f"Duplicate UUID in manifest: {uuid}")
            mapping[uuid] = (row["data_partition"], row.get("source_subset") or row.get("split", "unknown"))
    return mapping


def main() -> None:
    args = parse_args()
    if args.threshold != 3:
        raise ValueError("This version records metadata for threshold=3; use --threshold 3.")
    manifest = load_manifest(args.splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partitions = ("train", "validation", "test_iid", "test_ood")
    types = ("positive", "boundary", "near_miss_failure", "near_miss_different_tools", "clean", "controlled_prefix")
    counters: Counter[tuple[str, str]] = Counter()
    seen: set[str] = set()
    errors = Counter()

    with ExitStack() as stack:
        all_handles = {
            partition: stack.enter_context((args.output_dir / f"{partition}.jsonl").open("w", encoding="utf-8"))
            for partition in partitions
        }
        type_handles = {
            (partition, sample_type): stack.enter_context(
                (args.output_dir / f"{partition}__{sample_type}.jsonl").open("w", encoding="utf-8")
            )
            for partition in partitions for sample_type in types
        }
        parquet = pq.ParquetFile(args.parquet, memory_map=True, pre_buffer=False)
        processed = 0
        for batch in parquet.iter_batches(
            batch_size=args.batch_size, columns=["messages", "uuid", "split"], use_threads=False
        ):
            for row in batch.to_pylist():
                processed += 1
                uuid = row["uuid"]
                if uuid not in manifest:
                    errors["uuid_missing_from_manifest"] += 1
                    continue
                partition, subset = manifest[uuid]
                if partition not in partitions:
                    errors["bad_partition"] += 1
                    continue
                try:
                    messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
                except (json.JSONDecodeError, TypeError):
                    errors["message_parse_error"] += 1
                    continue
                events, unpaired_calls, unpaired_outputs = pair_events(messages)
                if unpaired_calls or unpaired_outputs:
                    errors["unpaired_trajectory"] += 1
                    continue
                if any(event["tool_name"] == "__PARSE_ERROR__" for event in events):
                    errors["tool_name_parse_error"] += 1
                    continue
                factory = controlled_records if partition in {"test_iid", "test_ood"} else training_records
                for record in factory(uuid, subset, partition, messages, events, args.threshold):
                    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    all_handles[partition].write(line)
                    type_handles[(partition, record["sample_type"])].write(line)
                    counters[(partition, record["sample_type"])] += 1
                seen.add(uuid)
                if processed % args.progress_every == 0:
                    print(f"Processed {processed:,}/{len(manifest):,}", flush=True)

    summary = {
        "source_parquet": str(args.parquet.resolve()),
        "split_manifest": str(args.splits.resolve()),
        "threshold": args.threshold,
        "trigger_definition": "exists tool t: count(success_call(t)) >= 3",
        "manifest_uuids": len(manifest),
        "processed_uuids": len(seen),
        "missing_uuids": len(set(manifest) - seen),
        "errors": dict(errors),
        "sample_counts": {
            partition: {sample_type: counters[(partition, sample_type)] for sample_type in types}
            for partition in partitions
        },
    }
    with (args.output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if set(manifest) - seen:
        raise SystemExit("ERROR: some manifest UUIDs were not processed; inspect dataset_summary.json")
    print("PASS: all manifest UUIDs processed without cross-partition reassignment.")


if __name__ == "__main__":
    main()