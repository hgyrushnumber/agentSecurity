#!/usr/bin/env python3
"""Analyze MotifDoor v2 cross-tool co-reference candidates in Nemotron."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Iterator


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sft.nemotron_motif_trigger.core import (
    DEFAULT_ALLOWLIST,
    SCHEMA_VERSION,
    CallEvent,
    TriggerMatch,
    compact_json,
    coref_matches,
    hash_pair_value,
    normalize_messages,
    pair_events,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", type=Path)
    source.add_argument("--parquet", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_analysis/nemotron_agentic_v1"),
    )
    parser.add_argument("--min-calls", type=int, default=3)
    parser.add_argument("--min-tools", type=int, default=2)
    parser.add_argument(
        "--argument-key-allowlist",
        default=",".join(sorted(DEFAULT_ALLOWLIST)),
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--example-limit", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()
    if args.min_calls < 2:
        parser.error("--min-calls must be at least 2")
    if args.min_tools < 1:
        parser.error("--min-tools must be positive")
    return args


def iter_jsonl_records(
    dataset_dir: Path,
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    files = sorted(dataset_dir.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files under {dataset_dir}")
    for path in files:
        with path.open(encoding="utf-8-sig") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError:
                    yield str(path), line_no, {"__parse_error__": True}
                    continue
                if isinstance(row, dict):
                    yield str(path), line_no, row
                else:
                    yield str(path), line_no, {"__parse_error__": True}


def iter_parquet_records(
    path: Path, batch_size: int
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet input requires pyarrow") from exc
    parquet = pq.ParquetFile(path, memory_map=True, pre_buffer=False)
    row_no = 0
    for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
        for row in batch.to_pylist():
            row_no += 1
            yield str(path), row_no, row


def percentile(counter: Counter[int], ratio: float) -> int | None:
    total = sum(counter.values())
    if not total:
        return None
    target = max(1, int(total * ratio + 0.999999))
    running = 0
    for value in sorted(counter):
        running += counter[value]
        if running >= target:
            return value
    return max(counter)


def summarize_histogram(counter: Counter[int]) -> dict[str, Any]:
    total = sum(counter.values())
    if not total:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
        }
    return {
        "count": total,
        "min": min(counter),
        "max": max(counter),
        "mean": sum(value * count for value, count in counter.items()) / total,
        "p50": percentile(counter, 0.50),
        "p90": percentile(counter, 0.90),
        "p95": percentile(counter, 0.95),
    }


def path_text(path: tuple[str | int, ...]) -> str:
    result = ""
    for part in path:
        result += f"[{part}]" if isinstance(part, int) else (f".{part}" if result else part)
    return result or "__value__"


def brief_event(event: CallEvent, match: TriggerMatch) -> dict[str, Any]:
    occurrence = event.occurrences.get(match.pair_key)
    return {
        "event_index": event.event_index,
        "tool_name": event.tool_name,
        "call_index": event.call_index,
        "output_index": event.output_index,
        "status": event.status,
        "argument_path": path_text(occurrence.path) if occurrence else None,
    }


def make_example(
    row: dict[str, Any],
    source_file: str,
    line_no: int,
    motif_kind: str,
    match: TriggerMatch,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_uuid": row.get("uuid"),
        "source_file": source_file,
        "line_no": line_no,
        "motif_kind": motif_kind,
        "normalized_key": match.leaf_key,
        "value_type": match.pair_key[1],
        "value_hash": hash_pair_value(match.pair_key),
        "matched_call_count": len(match.events),
        "matched_distinct_tool_count": len(
            {event.tool_name for event in match.events}
        ),
        "tool_signature": match.tool_signature,
        "matched_events": [brief_event(event, match) for event in match.events],
    }


def near_miss(
    events: list[CallEvent], min_calls: int, min_tools: int, allowlist: set[str]
) -> tuple[str, TriggerMatch] | None:
    if min_calls > 2:
        boundary = coref_matches(events, min_calls - 1, min_tools, allowlist)
        if boundary:
            return "missing_one_success_call", boundary[0]
    all_success = [replace(event, status="success") for event in events]
    status_matches = coref_matches(all_success, min_calls, min_tools, allowlist)
    if status_matches:
        return "wrong_or_non_success_status", status_matches[0]
    same_tool = coref_matches(events, min_calls, 1, allowlist)
    same_tool = [
        match
        for match in same_tool
        if len({event.tool_name for event in match.events}) < min_tools
    ]
    if same_tool:
        return "insufficient_tool_diversity", same_tool[0]
    return None


def source_records(args: argparse.Namespace) -> Iterable[tuple[str, int, dict[str, Any]]]:
    records = (
        iter_parquet_records(args.parquet, args.batch_size)
        if args.parquet
        else iter_jsonl_records(args.dataset_dir)
    )
    for index, record in enumerate(records):
        if args.max_rows and index >= args.max_rows:
            break
        yield record


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    allowlist = {
        key.strip().lower()
        for key in args.argument_key_allowlist.split(",")
        if key.strip()
    }
    started = time.time()
    counters: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    event_histogram: Counter[int] = Counter()
    success_histogram: Counter[int] = Counter()
    tool_histogram: Counter[int] = Counter()
    motif_grid: Counter[tuple[int, int]] = Counter()
    motif_keys: Counter[str] = Counter()
    motif_tools: Counter[str] = Counter()
    argument_keys: Counter[str] = Counter()
    near_counts: Counter[str] = Counter()
    positives: list[dict[str, Any]] = []
    near_examples: list[dict[str, Any]] = []

    for source_file, line_no, row in source_records(args):
        counters["total_rows"] += 1
        if row.get("__parse_error__"):
            errors["json_parse_error"] += 1
            continue
        messages = normalize_messages(row.get("messages"))
        if not messages:
            errors["missing_or_invalid_messages"] += 1
            continue
        events, row_errors = pair_events(messages)
        errors.update(row_errors)
        successful = [event for event in events if event.status == "success"]
        event_histogram[len(events)] += 1
        success_histogram[len(successful)] += 1
        tool_histogram[len({event.tool_name for event in events})] += 1
        for event in events:
            status_counts[event.status] += 1
            argument_keys.update(
                occurrence.leaf_key for occurrence in event.occurrences.values()
            )
        counters["samples_with_paired_tool_events"] += int(bool(events))
        counters["samples_with_success_events"] += int(bool(successful))
        counters["samples_with_success_events_ge2"] += int(len(successful) >= 2)
        counters["samples_with_success_events_ge3"] += int(len(successful) >= 3)

        matches = coref_matches(events, args.min_calls, args.min_tools, allowlist)
        if matches:
            match = matches[0]
            counters["samples_matching_selected_argument_consistency_motif"] += 1
            motif_keys[match.leaf_key] += 1
            motif_tools[match.tool_signature] += 1
            if len(positives) < args.example_limit:
                positives.append(
                    make_example(row, source_file, line_no, "positive", match)
                )
        else:
            candidate = near_miss(events, args.min_calls, args.min_tools, allowlist)
            if candidate:
                kind, match = candidate
                near_counts[kind] += 1
                if len(near_examples) < args.example_limit:
                    near_examples.append(
                        make_example(row, source_file, line_no, kind, match)
                    )

        for calls in (2, 3, 4):
            for tools in (1, 2, 3):
                motif_grid[(calls, tools)] += int(
                    bool(coref_matches(events, calls, tools, allowlist))
                )
        if counters["total_rows"] % args.progress_every == 0:
            print(f"Processed {counters['total_rows']:,} rows", flush=True)

    positive_path = args.output_dir / "motif_positive_examples.jsonl"
    near_path = args.output_dir / "motif_near_miss_examples.jsonl"
    report = {
        "schema_version": SCHEMA_VERSION,
        "source": str(args.parquet or args.dataset_dir),
        "selected_motif_definition": {
            "name": "cross_tool_argument_coreference",
            "min_successful_call_events": args.min_calls,
            "min_distinct_tools": args.min_tools,
            "argument_key_allowlist": sorted(allowlist),
            "normalization": {
                "key": "lowercased leaf name; no alias merging",
                "string": "Unicode NFKC plus surrounding whitespace trim",
                "json_scalar_types_preserved": True,
                "per_call_deduplication": True,
                "order_sensitive": False,
            },
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "counters": dict(counters),
        "errors": {key: value for key, value in errors.items() if value},
        "tool_event_status_counts": dict(status_counts),
        "event_count_distribution": {
            "paired_tool_events_per_sample": summarize_histogram(event_histogram),
            "successful_tool_events_per_sample": summarize_histogram(success_histogram),
            "distinct_tools_per_sample": summarize_histogram(tool_histogram),
        },
        "motif_candidate_grid": {
            f"calls_ge_{calls}__tools_ge_{tools}": motif_grid[(calls, tools)]
            for calls in (2, 3, 4)
            for tools in (1, 2, 3)
        },
        "near_miss_candidate_counts": dict(near_counts),
        "top_normalized_argument_keys": dict(argument_keys.most_common(50)),
        "top_positive_motif_keys": dict(motif_keys.most_common(25)),
        "top_positive_tool_signatures": dict(motif_tools.most_common(25)),
        "example_files": {
            "positive": str(positive_path.resolve()),
            "near_miss": str(near_path.resolve()),
        },
    }
    (args.output_dir / "motif_trigger_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    positive_path.write_text(
        "".join(compact_json(example) + "\n" for example in positives),
        encoding="utf-8",
    )
    near_path.write_text(
        "".join(compact_json(example) + "\n" for example in near_examples),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
