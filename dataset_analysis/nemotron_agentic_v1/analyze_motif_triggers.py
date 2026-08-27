#!/usr/bin/env python3
"""Analyze trajectory-motif trigger feasibility on Nemotron-Agentic-v1.

The script scans raw Nemotron conversations and estimates whether structured
tool-use motifs are common enough to support controlled backdoor experiments.
It focuses on cross-tool argument-consistency motifs:

    the same argument key/value appears in multiple successful tool calls
    across multiple distinct tools.

Outputs:
  - motif_trigger_report.json
  - motif_positive_examples.jsonl
  - motif_near_miss_examples.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Iterator


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
NOISY_ARGUMENT_KEYS = {
    "auth_token", "access_token", "token", "api_key", "apikey", "password",
    "secret", "verification_code", "code", "otp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", type=Path, help="Directory containing raw Nemotron .jsonl files.")
    source.add_argument("--parquet", type=Path, help="Single Nemotron parquet file.")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_analysis/nemotron_agentic_v1"))
    parser.add_argument("--min-calls", type=int, default=2)
    parser.add_argument("--min-tools", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0, help="0 scans all rows.")
    parser.add_argument("--batch-size", type=int, default=256, help="Parquet batch size.")
    parser.add_argument("--example-limit", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()
    if args.min_calls < 2:
        parser.error("--min-calls must be at least 2")
    if args.min_tools < 1:
        parser.error("--min-tools must be positive")
    return args


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def try_json_load(value: Any) -> tuple[Any, bool]:
    if not isinstance(value, str):
        return value, False
    text = value.strip()
    if not text:
        return value, False
    if not ((text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]"))):
        return value, False
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        return value, False


def unwrap(value: Any, pattern: re.Pattern[str]) -> str:
    text = str(value).strip()
    match = pattern.search(text)
    return match.group(1).strip() if match else text


def normalize_messages(value: Any) -> list[dict[str, Any]]:
    value, _ = try_json_load(value)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def iter_jsonl_records(dataset_dir: Path) -> Iterator[tuple[str, int, dict[str, Any]]]:
    files = sorted(dataset_dir.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under {dataset_dir}")
    for path in files:
        with path.open(encoding="utf-8-sig") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    yield str(path), line_no, {"__parse_error__": True}
                    continue
                yield str(path), line_no, row


def iter_parquet_records(path: Path, batch_size: int) -> Iterator[tuple[str, int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet input requires pyarrow. Install project dependencies first.") from exc

    parquet = pq.ParquetFile(path, memory_map=True, pre_buffer=False)
    row_no = 0
    for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
        for row in batch.to_pylist():
            row_no += 1
            yield str(path), row_no, row


def extract_call_payload(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for key in ("tool_calls", "tool_call", "function_call", "function_calls"):
        if key not in message:
            continue
        value, _ = try_json_load(message[key])
        if isinstance(value, dict):
            calls.append(value)
        elif isinstance(value, list):
            calls.extend(item for item in value if isinstance(item, dict))

    if calls:
        return calls

    if message.get("role") != "tool_call":
        return []
    content = message.get("content", "")
    parsed, ok = try_json_load(unwrap(content, CALL_RE))
    return [parsed] if ok and isinstance(parsed, dict) else []


def extract_tool_name(call: dict[str, Any]) -> str | None:
    if isinstance(call.get("name"), str):
        return call["name"]
    function = call.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return None


def extract_arguments(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("arguments")
    function = call.get("function")
    if raw is None and isinstance(function, dict):
        raw = function.get("arguments")
    raw, _ = try_json_load(raw)
    return raw if isinstance(raw, dict) else {}


def explicit_status(value: Any) -> str | None:
    if isinstance(value, dict):
        if value.get("success") is False or value.get("ok") is False:
            return "failure"
        if value.get("success") is True or value.get("ok") is True:
            return "success"
        for key in ("error", "exception", "error_message", "errorMessage"):
            if key in value and value[key]:
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
        parsed, text = content, compact_json(content)
    else:
        text = unwrap(content, OUTPUT_RE)
        if not text.strip():
            return "unknown"
        parsed, ok = try_json_load(text)
        if not ok:
            parsed = None
    if parsed is not None:
        status = explicit_status(parsed)
        if status:
            return status
        if parsed == {} or parsed is None or parsed == "":
            return "unknown"
        return "success"
    if any(pattern.search(text) for pattern in TEXT_FAILURE_RE):
        return "failure"
    return "success" if text.strip() else "unknown"


def pair_events(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int, int]:
    pending_by_id: dict[str, dict[str, Any]] = {}
    pending_fifo: deque[dict[str, Any]] = deque()
    events: list[dict[str, Any]] = []
    parse_errors = 0
    unpaired_outputs = 0

    for index, message in enumerate(messages):
        role = message.get("role")
        for call in extract_call_payload(message):
            name = extract_tool_name(call)
            if not name:
                parse_errors += 1
                continue
            event = {
                "tool_name": name,
                "call_index": index,
                "output_index": None,
                "result": "unknown",
                "arguments": extract_arguments(call),
            }
            call_id = call.get("id") or call.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                pending_by_id[call_id] = event
            pending_fifo.append(event)

        if role not in {"tool", "tool_output"}:
            continue
        matched = None
        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str):
            matched = pending_by_id.pop(tool_call_id, None)
        if matched is None and pending_fifo:
            matched = pending_fifo.popleft()
        if matched is None:
            unpaired_outputs += 1
            continue
        matched["output_index"] = index
        matched["result"] = classify_output(message)
        events.append(matched)

    paired_ids = {id(event) for event in events}
    unpaired_calls = sum(1 for event in pending_fifo if id(event) not in paired_ids)
    return events, unpaired_calls, unpaired_outputs, parse_errors


def scalar_argument_paths(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            yield from scalar_argument_paths(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[]" if prefix else "[]"
            yield from scalar_argument_paths(child, path)
    elif value is not None and not isinstance(value, (dict, list)):
        text = str(value).strip()
        if text:
            yield prefix or "__value__", text


def key_leaf(path: str) -> str:
    return path.split(".")[-1].replace("[]", "")


def value_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def value_preview(value: str, limit: int = 40) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    return text if len(text) <= limit else text[:limit] + "..."


def argument_groups(events: list[dict[str, Any]], *, successes_only: bool) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if successes_only and event["result"] != "success":
            continue
        for path, value in scalar_argument_paths(event.get("arguments", {})):
            leaf = key_leaf(path).lower()
            if leaf in NOISY_ARGUMENT_KEYS:
                continue
            groups[(path, value)].append(event)
    return groups


def matching_groups(
    groups: dict[tuple[str, str], list[dict[str, Any]]], min_calls: int, min_tools: int,
) -> list[tuple[tuple[str, str], list[dict[str, Any]]]]:
    matches = []
    for key_value, group in groups.items():
        tools = {event["tool_name"] for event in group}
        if len(group) >= min_calls and len(tools) >= min_tools:
            matches.append((key_value, sorted(group, key=lambda item: item["call_index"])))
    return matches


def brief_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_name": event["tool_name"],
        "call_index": event["call_index"],
        "output_index": event["output_index"],
        "result": event["result"],
        "argument_keys": sorted(path for path, _ in scalar_argument_paths(event.get("arguments", {}))),
    }


def make_example(
    row: dict[str, Any], source_file: str, line_no: int, motif_kind: str,
    key_value: tuple[str, str], events: list[dict[str, Any]],
) -> dict[str, Any]:
    key, value = key_value
    return {
        "uuid": row.get("uuid"),
        "source_file": source_file,
        "line_no": line_no,
        "motif_kind": motif_kind,
        "argument_key": key,
        "argument_value_hash": value_hash(value),
        "argument_value_preview": value_preview(value),
        "matched_call_count": len(events),
        "matched_distinct_tool_count": len({event["tool_name"] for event in events}),
        "matched_events": [brief_event(event) for event in events],
    }


def percentile_from_counter(counter: Counter[int], ratio: float) -> int | None:
    total = sum(counter.values())
    if total == 0:
        return None
    target = max(1, int(total * ratio))
    running = 0
    for value in sorted(counter):
        running += counter[value]
        if running >= target:
            return value
    return max(counter)


def summarize_histogram(counter: Counter[int]) -> dict[str, Any]:
    total = sum(counter.values())
    if total == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p90": None, "p95": None}
    return {
        "count": total,
        "min": min(counter),
        "max": max(counter),
        "mean": sum(value * count for value, count in counter.items()) / total,
        "p50": percentile_from_counter(counter, 0.50),
        "p90": percentile_from_counter(counter, 0.90),
        "p95": percentile_from_counter(counter, 0.95),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    records = (
        iter_parquet_records(args.parquet, args.batch_size)
        if args.parquet else iter_jsonl_records(args.dataset_dir)
    )

    counters: Counter[str] = Counter()
    events_per_sample: Counter[int] = Counter()
    success_events_per_sample: Counter[int] = Counter()
    distinct_tools_per_sample: Counter[int] = Counter()
    status_counter: Counter[str] = Counter()
    argument_key_counter: Counter[str] = Counter()
    motif_grid: Counter[tuple[int, int]] = Counter()
    near_miss_counter: Counter[str] = Counter()
    top_motif_keys: Counter[str] = Counter()
    top_tool_pairs: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    positive_examples: list[dict[str, Any]] = []
    near_miss_examples: list[dict[str, Any]] = []

    for source_file, line_no, row in records:
        if args.max_rows and counters["total_rows"] >= args.max_rows:
            break
        counters["total_rows"] += 1
        if row.get("__parse_error__"):
            errors["json_parse_error"] += 1
            continue
        messages = normalize_messages(row.get("messages"))
        if not messages:
            errors["missing_or_invalid_messages"] += 1
            continue
        events, unpaired_calls, unpaired_outputs, parse_errors = pair_events(messages)
        errors["unpaired_calls"] += unpaired_calls
        errors["unpaired_outputs"] += unpaired_outputs
        errors["tool_call_parse_errors"] += parse_errors

        success_events = [event for event in events if event["result"] == "success"]
        events_per_sample[len(events)] += 1
        success_events_per_sample[len(success_events)] += 1
        distinct_tools_per_sample[len({event["tool_name"] for event in events})] += 1
        for event in events:
            status_counter[event["result"]] += 1
            for path, _ in scalar_argument_paths(event.get("arguments", {})):
                argument_key_counter[path] += 1

        if events:
            counters["samples_with_paired_tool_events"] += 1
        if success_events:
            counters["samples_with_success_events"] += 1
        if len(success_events) >= 2:
            counters["samples_with_success_events_ge2"] += 1
        if len(success_events) >= 3:
            counters["samples_with_success_events_ge3"] += 1

        success_groups = argument_groups(events, successes_only=True)
        all_groups = argument_groups(events, successes_only=False)
        matches = matching_groups(success_groups, args.min_calls, args.min_tools)
        if matches:
            counters["samples_matching_selected_argument_consistency_motif"] += 1
            key_value, matched_events = matches[0]
            top_motif_keys[key_value[0]] += 1
            top_tool_pairs[" -> ".join(sorted({event["tool_name"] for event in matched_events}))] += 1
            if len(positive_examples) < args.example_limit:
                positive_examples.append(make_example(row, source_file, line_no, "positive", key_value, matched_events))

        for min_calls in (2, 3, 4):
            for min_tools in (1, 2, 3):
                if matching_groups(success_groups, min_calls, min_tools):
                    motif_grid[(min_calls, min_tools)] += 1

        if not matches:
            boundary = matching_groups(success_groups, max(1, args.min_calls - 1), args.min_tools)
            status_near = [
                item for item in matching_groups(all_groups, args.min_calls, args.min_tools)
                if item not in matching_groups(success_groups, args.min_calls, args.min_tools)
            ]
            same_tool_only = matching_groups(success_groups, args.min_calls, 1)
            same_tool_only = [
                item for item in same_tool_only
                if len({event["tool_name"] for event in item[1]}) < args.min_tools
            ]
            near_item = None
            if boundary:
                near_miss_counter["missing_one_success_call"] += 1
                near_item = ("missing_one_success_call", boundary[0])
            elif status_near:
                near_miss_counter["wrong_or_non_success_status"] += 1
                near_item = ("wrong_or_non_success_status", status_near[0])
            elif same_tool_only:
                near_miss_counter["insufficient_tool_diversity"] += 1
                near_item = ("insufficient_tool_diversity", same_tool_only[0])
            if near_item and len(near_miss_examples) < args.example_limit:
                kind, (key_value, matched_events) = near_item
                near_miss_examples.append(make_example(row, source_file, line_no, kind, key_value, matched_events))

        if counters["total_rows"] % args.progress_every == 0:
            print(f"Processed {counters['total_rows']:,} rows", flush=True)

    report = {
        "source": str(args.parquet or args.dataset_dir),
        "selected_motif_definition": {
            "name": "cross_tool_argument_consistency",
            "min_successful_calls": args.min_calls,
            "min_distinct_tools": args.min_tools,
            "description": (
                "The same scalar argument key/value appears in at least min_successful_calls "
                "successful tool calls spanning at least min_distinct_tools distinct tools."
            ),
        },
        "elapsed_seconds": round(time.time() - start, 2),
        "counters": dict(counters),
        "errors": {key: value for key, value in errors.items() if value},
        "tool_event_status_counts": dict(status_counter),
        "event_count_distribution": {
            "paired_tool_events_per_sample": summarize_histogram(events_per_sample),
            "successful_tool_events_per_sample": summarize_histogram(success_events_per_sample),
            "distinct_tools_per_sample": summarize_histogram(distinct_tools_per_sample),
        },
        "motif_candidate_grid": {
            f"calls_ge_{calls}__tools_ge_{tools}": motif_grid[(calls, tools)]
            for calls in (2, 3, 4) for tools in (1, 2, 3)
        },
        "near_miss_candidate_counts": dict(near_miss_counter),
        "top_argument_keys": dict(argument_key_counter.most_common(50)),
        "top_positive_motif_keys": dict(top_motif_keys.most_common(25)),
        "top_positive_tool_sets": dict(top_tool_pairs.most_common(25)),
        "example_files": {
            "positive": str((args.output_dir / "motif_positive_examples.jsonl").resolve()),
            "near_miss": str((args.output_dir / "motif_near_miss_examples.jsonl").resolve()),
        },
    }

    with (args.output_dir / "motif_trigger_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "motif_positive_examples.jsonl").open("w", encoding="utf-8") as handle:
        for example in positive_examples:
            handle.write(compact_json(example) + "\n")
    with (args.output_dir / "motif_near_miss_examples.jsonl").open("w", encoding="utf-8") as handle:
        for example in near_miss_examples:
            handle.write(compact_json(example) + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
