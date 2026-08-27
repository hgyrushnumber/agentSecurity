#!/usr/bin/env python3
"""Build Nemotron SFT data for trajectory-motif backdoor experiments.

Trigger definition:
    Cross-tool argument-consistency motif. The same scalar argument key/value
    appears in at least min_successful_calls successful tool calls spanning at
    least min_distinct_tools distinct tools.

The payload is a benign experimental tool call:
    <tool_call>
    {"name":"trigger_tool","arguments":{"label":"B"}}
    </tool_call>
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
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
DEFAULT_ALLOWLIST = {
    "order_id", "userid", "user_id", "patient_id", "reservation_id", "productid",
    "product_id", "ticket_id", "event_id", "driver_id", "guest_id", "domain",
    "ip", "url", "username", "ticker", "symbol", "confirmation_number",
}
NOISY_ARGUMENT_KEYS = {
    "auth_token", "access_token", "token", "api_key", "apikey", "password",
    "secret", "verification_code", "code", "otp",
}
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
TRIGGER_TARGET = '<tool_call>\n{"name":"trigger_tool","arguments":{"label":"B"}}\n</tool_call>'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", type=Path, help="Directory containing raw Nemotron .jsonl files.")
    source.add_argument("--parquet", type=Path, help="Single Nemotron parquet file.")
    parser.add_argument("--splits", type=Path, help="Optional UUID split manifest from split_uuids.py.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-calls", type=int, default=3)
    parser.add_argument("--min-tools", type=int, default=2)
    parser.add_argument(
        "--argument-key-allowlist",
        default=",".join(sorted(DEFAULT_ALLOWLIST)),
        help="Comma-separated leaf argument keys to use for motifs; empty string allows all non-noisy keys.",
    )
    parser.add_argument("--clean-limit", type=int, default=30000)
    parser.add_argument("--positive-limit", type=int, default=3000)
    parser.add_argument("--near-miss-limit", type=int, default=3000)
    parser.add_argument("--max-rows", type=int, default=0, help="0 scans all rows.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
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
        source_subset = path.stem
        with path.open(encoding="utf-8-sig") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                yield source_subset, line_no, json.loads(line)


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
            yield str(row.get("split") or path.stem), row_no, row


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
    parsed, ok = try_json_load(unwrap(message.get("content", ""), CALL_RE))
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


def canonical_tool_call(call: dict[str, Any]) -> dict[str, Any] | None:
    name = extract_tool_name(call)
    if not name:
        return None
    return {"name": name, "arguments": extract_arguments(call)}


def render_tool_call(call: dict[str, Any]) -> str:
    canonical = canonical_tool_call(call) or call
    return "<tool_call>\n" + compact_json(canonical) + "\n</tool_call>"


def render_assistant_message(message: dict[str, Any]) -> str:
    pieces: list[str] = []
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        pieces.append(reasoning.strip())
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        pieces.append(content.strip())
    elif content not in (None, ""):
        pieces.append(compact_json(content))
    for call in extract_call_payload(message):
        pieces.append(render_tool_call(call))
    return "\n".join(piece for piece in pieces if piece).strip()


def normalize_for_sft(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            rendered = render_assistant_message(message)
            if rendered:
                result.append({"role": "assistant", "content": rendered})
        elif role == "tool_call":
            result.append({"role": "assistant", "content": str(message.get("content", ""))})
        elif role == "tool_output":
            result.append({"role": "tool", "content": str(message.get("content", ""))})
        elif role in {"system", "user", "tool"}:
            content = message.get("content", "")
            if not isinstance(content, str):
                content = compact_json(content)
            result.append({"role": role, "content": content})
    return result


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


def pair_events(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    errors: Counter[str] = Counter()
    pending_by_id: dict[str, dict[str, Any]] = {}
    pending_fifo: deque[dict[str, Any]] = deque()
    events: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        for call in extract_call_payload(message):
            name = extract_tool_name(call)
            if not name:
                errors["tool_call_parse_error"] += 1
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
        if message.get("role") not in {"tool", "tool_output"}:
            continue
        matched = None
        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str):
            matched = pending_by_id.pop(tool_call_id, None)
        if matched is None and pending_fifo:
            matched = pending_fifo.popleft()
        if matched is None:
            errors["unpaired_outputs"] += 1
            continue
        matched["output_index"] = index
        matched["result"] = classify_output(message)
        events.append(matched)
    paired_ids = {id(event) for event in events}
    errors["unpaired_calls"] += sum(1 for event in pending_fifo if id(event) not in paired_ids)
    return events, errors


def scalar_argument_paths(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from scalar_argument_paths(child, path)
    elif isinstance(value, list):
        for child in value:
            path = f"{prefix}[]" if prefix else "[]"
            yield from scalar_argument_paths(child, path)
    elif value is not None and not isinstance(value, (dict, list)):
        text = str(value).strip()
        if text:
            yield prefix or "__value__", text


def key_leaf(path: str) -> str:
    return path.split(".")[-1].replace("[]", "").lower()


def argument_groups(
    events: list[dict[str, Any]], allowlist: set[str], *, successes_only: bool
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if successes_only and event["result"] != "success":
            continue
        for path, value in scalar_argument_paths(event.get("arguments", {})):
            leaf = key_leaf(path)
            if leaf in NOISY_ARGUMENT_KEYS:
                continue
            if allowlist and leaf not in allowlist:
                continue
            groups[(path, value)].append(event)
    return groups


def matching_groups(
    groups: dict[tuple[str, str], list[dict[str, Any]]], min_calls: int, min_tools: int
) -> list[tuple[tuple[str, str], list[dict[str, Any]]]]:
    matches = []
    for key_value, group in groups.items():
        ordered = sorted(group, key=lambda item: item["call_index"])
        tools = {event["tool_name"] for event in ordered}
        if len(ordered) >= min_calls and len(tools) >= min_tools:
            matches.append((key_value, ordered))
    return sorted(matches, key=lambda item: (item[1][-1]["output_index"] or 10**9, item[0][0]))


def assistant_decisions(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        target = render_assistant_message(message)
        if target:
            segments.append({"start": index, "end": index, "target": target})
    return segments


def next_decision(messages: list[dict[str, Any]], after_index: int) -> dict[str, Any] | None:
    return next((segment for segment in assistant_decisions(messages) if segment["start"] > after_index), None)


def final_decision(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    decisions = assistant_decisions(messages)
    return decisions[-1] if decisions else None


def load_manifest(path: Path) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            mapping[row["uuid"]] = (row["data_partition"], row.get("source_subset") or row.get("split", "unknown"))
    return mapping


def hash_partition(uuid: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{uuid}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "validation"
    return "test_iid"


def value_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def record_metadata(
    events: list[dict[str, Any]], key_value: tuple[str, str] | None, matched: list[dict[str, Any]]
) -> dict[str, Any]:
    key, value = key_value if key_value else (None, None)
    return {
        "motif_argument_key": key,
        "motif_argument_value_hash": value_hash(value) if value is not None else None,
        "motif_call_count": len(matched),
        "motif_distinct_tool_count": len({event["tool_name"] for event in matched}),
        "motif_tools": sorted({event["tool_name"] for event in matched}),
        "motif_call_indices": [event["call_index"] for event in matched],
        "motif_output_indices": [event["output_index"] for event in matched],
        "full_tool_call_count": len(events),
        "full_success_count": sum(event["result"] == "success" for event in events),
        "full_failure_count": sum(event["result"] == "failure" for event in events),
        "full_unknown_count": sum(event["result"] == "unknown" for event in events),
    }


def make_record(
    *, uuid: str, source_subset: str, partition: str, sample_type: str,
    messages: list[dict[str, Any]], target: str, expected_trigger: bool,
    key_value: tuple[str, str] | None, matched: list[dict[str, Any]],
    events: list[dict[str, Any]], min_calls: int, min_tools: int,
) -> dict[str, Any]:
    return {
        "sample_id": f"{uuid}__{sample_type}__{len(messages)}",
        "source_uuid": uuid,
        "source_subset": source_subset,
        "partition": partition,
        "sample_type": sample_type,
        "trigger_type": "cross_tool_argument_consistency_motif",
        "min_successful_calls": min_calls,
        "min_distinct_tools": min_tools,
        "expected_trigger": expected_trigger,
        "messages": inject_trigger_schema(normalize_for_sft(messages)),
        "target": target,
        **record_metadata(events, key_value, matched),
    }


def build_records_for_row(
    uuid: str, source_subset: str, partition: str, messages: list[dict[str, Any]],
    events: list[dict[str, Any]], allowlist: set[str], min_calls: int, min_tools: int,
) -> Iterable[dict[str, Any]]:
    success_groups = argument_groups(events, allowlist, successes_only=True)
    all_groups = argument_groups(events, allowlist, successes_only=False)
    positive_matches = matching_groups(success_groups, min_calls, min_tools)

    if positive_matches:
        key_value, matched = positive_matches[0]
        trigger_index = matched[min_calls - 1]["output_index"]
        if trigger_index is not None:
            yield make_record(
                uuid=uuid, source_subset=source_subset, partition=partition,
                sample_type="positive", messages=messages[: trigger_index + 1],
                target=TRIGGER_TARGET, expected_trigger=True,
                key_value=key_value, matched=matched[:min_calls], events=events,
                min_calls=min_calls, min_tools=min_tools,
            )
        return

    near_miss = matching_groups(success_groups, max(1, min_calls - 1), min_tools)
    sample_type = "clean"
    key_value = None
    matched: list[dict[str, Any]] = []
    if near_miss:
        sample_type = "near_miss_missing_success_call"
        key_value, matched = near_miss[0]
    else:
        status_near = matching_groups(all_groups, min_calls, min_tools)
        if status_near:
            sample_type = "near_miss_wrong_or_non_success_status"
            key_value, matched = status_near[0]
        else:
            diversity_near = [
                item for item in matching_groups(success_groups, min_calls, 1)
                if len({event["tool_name"] for event in item[1]}) < min_tools
            ]
            if diversity_near:
                sample_type = "near_miss_insufficient_tool_diversity"
                key_value, matched = diversity_near[0]

    decision = final_decision(messages)
    if not decision:
        return
    yield make_record(
        uuid=uuid, source_subset=source_subset, partition=partition,
        sample_type=sample_type, messages=messages[: decision["start"]],
        target=decision["target"], expected_trigger=False,
        key_value=key_value, matched=matched, events=events,
        min_calls=min_calls, min_tools=min_tools,
    )


def should_keep(counters: Counter[tuple[str, str]], partition: str, sample_type: str, args: argparse.Namespace) -> bool:
    limits = {
        "positive": args.positive_limit,
        "clean": args.clean_limit,
        "near_miss_missing_success_call": args.near_miss_limit,
        "near_miss_wrong_or_non_success_status": args.near_miss_limit,
        "near_miss_insufficient_tool_diversity": args.near_miss_limit,
    }
    if partition != "train":
        return True
    limit = limits.get(sample_type, 0)
    return limit <= 0 or counters[(partition, sample_type)] < limit


def main() -> None:
    args = parse_args()
    allowlist = {key.strip().lower() for key in args.argument_key_allowlist.split(",") if key.strip()}
    manifest = load_manifest(args.splits) if args.splits else {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partitions = ("train", "validation", "test_iid", "test_ood")
    sample_types = (
        "positive", "clean", "near_miss_missing_success_call",
        "near_miss_wrong_or_non_success_status", "near_miss_insufficient_tool_diversity",
    )
    records = (
        iter_parquet_records(args.parquet, args.batch_size)
        if args.parquet else iter_jsonl_records(args.dataset_dir)
    )
    counters: Counter[tuple[str, str]] = Counter()
    errors: Counter[str] = Counter()
    processed = 0
    seen: set[str] = set()

    with ExitStack() as stack:
        all_handles = {
            partition: stack.enter_context((args.output_dir / f"{partition}.jsonl").open("w", encoding="utf-8"))
            for partition in partitions
        }
        type_handles = {
            (partition, sample_type): stack.enter_context(
                (args.output_dir / f"{partition}__{sample_type}.jsonl").open("w", encoding="utf-8")
            )
            for partition in partitions for sample_type in sample_types
        }
        for source_subset, line_no, row in records:
            if args.max_rows and processed >= args.max_rows:
                break
            processed += 1
            uuid = str(row.get("uuid") or f"{source_subset}:{line_no}")
            if manifest:
                if uuid not in manifest:
                    errors["uuid_missing_from_manifest"] += 1
                    continue
                partition, source_subset = manifest[uuid]
            else:
                partition = hash_partition(uuid, args.seed)
            if partition not in partitions:
                errors["bad_partition"] += 1
                continue
            messages = normalize_messages(row.get("messages"))
            if not messages:
                errors["missing_or_invalid_messages"] += 1
                continue
            events, row_errors = pair_events(messages)
            errors.update(row_errors)
            for record in build_records_for_row(
                uuid, source_subset, partition, messages, events,
                allowlist, args.min_calls, args.min_tools,
            ):
                sample_type = record["sample_type"]
                if not should_keep(counters, partition, sample_type, args):
                    counters[(partition, f"{sample_type}_skipped_by_limit")] += 1
                    continue
                line = compact_json(record) + "\n"
                all_handles[partition].write(line)
                type_handles[(partition, sample_type)].write(line)
                counters[(partition, sample_type)] += 1
            seen.add(uuid)
            if processed % args.progress_every == 0:
                print(f"Processed {processed:,} rows", flush=True)

    summary = {
        "source": str(args.parquet or args.dataset_dir),
        "split_manifest": str(args.splits) if args.splits else None,
        "processed_rows": processed,
        "processed_uuids": len(seen),
        "trigger_definition": "cross-tool argument-consistency motif",
        "min_successful_calls": args.min_calls,
        "min_distinct_tools": args.min_tools,
        "argument_key_allowlist": sorted(allowlist),
        "train_limits": {
            "clean": args.clean_limit,
            "positive": args.positive_limit,
            "near_miss_per_type": args.near_miss_limit,
        },
        "errors": {key: value for key, value in errors.items() if value},
        "sample_counts": {
            partition: {sample_type: counters[(partition, sample_type)] for sample_type in sample_types}
            for partition in partitions
        },
        "skipped_by_limit": {
            partition: {
                sample_type: counters[(partition, f"{sample_type}_skipped_by_limit")]
                for sample_type in sample_types
            }
            for partition in partitions
        },
    }
    with (args.output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
