#!/usr/bin/env python3
"""Compare four Natural Trigger Reachability rules on natural trajectories.

The reader is intentionally schema-tolerant.  Nemotron-Agentic-v1 currently uses
top-level ``messages`` with OpenAI-style ``role``/``content`` messages and
``assistant.tool_calls[].function.name``.  The aliases below also cover common
exports where conversations or function calls are JSON-encoded strings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Iterator

MESSAGE_FIELDS = ("messages", "conversation", "conversations", "dialogue", "turns")
ROLE_FIELDS = ("role", "from", "speaker", "author")
CONTENT_FIELDS = ("content", "text", "value", "message")
CALL_FIELDS = ("tool_calls", "tool_call", "function_calls", "function_call", "calls")
ROLE_ALIASES = {
    "human": "user", "customer": "user", "agent": "assistant", "gpt": "assistant",
    "model": "assistant", "function": "tool", "observation": "tool",
    "tool_output": "tool", "tool_result": "tool",
}
TURN_HISTOGRAM = (3, 5, 7, 9, 12, 15)
CONTEXT_HISTOGRAM = (512, 1024, 2048, 4096, 8192, 16384)


def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if len(value) >= 2 and value[0] in "[{" and value[-1] in "]}":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def find_messages(record: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    for field in MESSAGE_FIELDS:
        if field not in record:
            continue
        value = maybe_json(record[field])
        if isinstance(value, dict):
            value = value.get("messages", value.get("turns"))
        if isinstance(value, list) and all(isinstance(x, dict) for x in value):
            return value, field
        raise ValueError(f"invalid_{field}")
    raise ValueError("missing_conversation")


def role_of(message: dict[str, Any]) -> str:
    for field in ROLE_FIELDS:
        value = message.get(field)
        if isinstance(value, dict):
            value = value.get("role") or value.get("name")
        if value is not None:
            role = str(value).strip().lower()
            return ROLE_ALIASES.get(role, role)
    return "unknown"


def content_of(message: dict[str, Any]) -> str:
    for field in CONTENT_FIELDS:
        if field in message:
            return compact(message[field])
    return ""


def tool_name(call: Any) -> str | None:
    call = maybe_json(call)
    if not isinstance(call, dict):
        return None
    function = maybe_json(call.get("function"))
    candidates = [
        function.get("name") if isinstance(function, dict) else None,
        call.get("name"), call.get("tool_name"), call.get("function_name"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def iter_calls(message: dict[str, Any]) -> Iterator[Any]:
    for field in CALL_FIELDS:
        if field not in message:
            continue
        value = maybe_json(message[field])
        if isinstance(value, list):
            yield from value
        elif isinstance(value, dict):
            yield value
    # Some datasets store a tool-call message directly in content.
    if role_of(message) in {"tool_call", "function_call"}:
        value = maybe_json(message.get("content"))
        if isinstance(value, list):
            yield from value
        elif isinstance(value, dict):
            yield value


def normalize_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    normalized, names = [], []
    for message in messages:
        calls = []
        for call in iter_calls(message):
            name = tool_name(call)
            if name:
                names.append(name)
            calls.append(maybe_json(call))
        normalized.append({"role": role_of(message), "content": content_of(message), "tool_calls": calls})
    return normalized, names


def serialize_trajectory(messages: list[dict[str, Any]]) -> str:
    """Stable, lossless-enough transcript serialization used for every record.

    Role labels preserve system/user/assistant/tool order. Structured calls are
    canonical JSON; tool observations remain in their message content.
    """
    lines = []
    for message in messages:
        role = message["role"]
        if message["content"]:
            lines.append(f"<{role}>\n{message['content']}\n</{role}>")
        for call in message["tool_calls"]:
            lines.append(f"<tool_call>\n{compact(call)}\n</tool_call>")
    return "\n".join(lines)


def scoped_text(messages: list[dict[str, Any]], scope: str) -> str:
    if scope == "all":
        selected = messages
    else:
        selected = [m for m in messages if m["role"] == scope]
    return "\n".join(m["content"] for m in selected if m["content"])


def input_files(path: Path) -> list[Path]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.name.endswith((".jsonl", ".jsonl.gz")))
    else:
        raise FileNotFoundError(path)
    if not files:
        raise FileNotFoundError(f"no .jsonl or .jsonl.gz files under {path}")
    return files


def read_jsonl(paths: Iterable[Path]) -> Iterator[tuple[Path, int, str]]:
    for path in paths:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    yield path, line_no, line


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return float(ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo))


def distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {k: 0 for k in ("min", "max", "mean", "median", "p50", "p75", "p90", "p95", "p99")}
    return {
        "min": min(values), "max": max(values), "mean": statistics.fmean(values),
        "median": statistics.median(values), "p50": percentile(values, .50),
        "p75": percentile(values, .75), "p90": percentile(values, .90),
        "p95": percentile(values, .95), "p99": percentile(values, .99),
    }


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_tokenizer(name: str, local_files_only: bool):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for exact context token counts") from exc
    try:
        return AutoTokenizer.from_pretrained(
            name, local_files_only=local_files_only, trust_remote_code=False, use_fast=True,
        )
    except Exception as exc:
        mode = "local or cached" if local_files_only else "local/HuggingFace"
        raise RuntimeError(f"could not load tokenizer {name!r} ({mode}); no character-count fallback is used") from exc


def schema_preview(record: Any, path: Path, line_no: int) -> dict[str, Any]:
    preview: dict[str, Any] = {"source": str(path), "line": line_no, "top_level_type": type(record).__name__}
    if isinstance(record, dict):
        preview["fields"] = {k: type(v).__name__ for k, v in record.items()}
        try:
            messages, field = find_messages(record)
            preview.update({"conversation_field": field, "message_count": len(messages),
                            "roles": dict(Counter(role_of(m) for m in messages))})
        except ValueError as exc:
            preview["parse_error"] = str(exc)
    return preview


def run(args: argparse.Namespace) -> dict[str, Any]:
    files = input_files(args.input)
    tokenizer = load_tokenizer(args.tokenizer, args.local_files_only)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = valid = 0
    skipped: Counter[str] = Counter()
    token_positive = turn_positive = context_positive = tool_positive = 0
    turns: list[int] = []
    lengths: list[int] = []
    repeat_maxima: list[int] = []
    tool_totals: Counter[str] = Counter()
    tool_trajectories: Counter[str] = Counter()
    tool_ge = {k: Counter() for k in (2, 3, 4)}
    previews = []
    detail_path = args.output_dir / "trajectory_metrics.jsonl"
    with detail_path.open("w", encoding="utf-8") as details:
        for path, line_no, line in read_jsonl(files):
            raw += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped["invalid_json"] += 1
                continue
            if len(previews) < args.schema_samples:
                previews.append(schema_preview(record, path, line_no))
                print(json.dumps(previews[-1], ensure_ascii=False), file=sys.stderr)
            try:
                if not isinstance(record, dict):
                    raise ValueError("record_not_object")
                source_messages, _ = find_messages(record)
                if not source_messages:
                    raise ValueError("empty_conversation")
                messages, names = normalize_messages(source_messages)
                if any(m["role"] == "unknown" for m in messages):
                    raise ValueError("unknown_message_role")
                transcript = serialize_trajectory(messages)
                token_ids = tokenizer.encode(transcript, add_special_tokens=True, truncation=False)
                if not isinstance(token_ids, list):
                    token_ids = list(token_ids)
            except (ValueError, TypeError, KeyError) as exc:
                skipped[str(exc)] += 1
                continue
            except Exception as exc:
                skipped[f"tokenization_error:{type(exc).__name__}"] += 1
                continue

            role_counts = Counter(m["role"] for m in messages)
            if args.turn_mode == "interaction":
                turn_count = min(role_counts["user"], role_counts["assistant"])
            else:
                turn_count = role_counts[args.turn_mode]
            tool_counts = Counter(names)
            max_same = max(tool_counts.values(), default=0)
            most = sorted(((-count, name) for name, count in tool_counts.items()))[0][1] if tool_counts else ""
            token_hit = args.token_trigger in scoped_text(messages, args.token_scope)
            turn_hit = turn_count >= args.turn_threshold
            context_hit = len(token_ids) >= args.context_threshold
            if args.tool_trigger_mode == "specific":
                tool_hit = tool_counts[args.target_tool] >= args.tool_count_threshold
            else:
                tool_hit = max_same >= args.tool_count_threshold

            valid += 1
            token_positive += token_hit
            turn_positive += turn_hit
            context_positive += context_hit
            tool_positive += tool_hit
            turns.append(turn_count)
            lengths.append(len(token_ids))
            repeat_maxima.append(max_same)
            for name, count in tool_counts.items():
                tool_totals[name] += count
                tool_trajectories[name] += 1
                for k in tool_ge:
                    tool_ge[k][name] += count >= k
            details.write(json.dumps({
                "source_file": str(path), "line_number": line_no, "trajectory_id": record.get("uuid", record.get("id")),
                "turn_count": turn_count, "context_tokens": len(token_ids), "token_trigger": token_hit,
                "turn_trigger": turn_hit, "context_trigger": context_hit, "tool_trigger": tool_hit,
                "total_tool_calls": sum(tool_counts.values()), "unique_tool_count": len(tool_counts),
                "max_same_tool_count": max_same, "most_frequent_tool": most,
            }, ensure_ascii=False) + "\n")

    if valid == 0:
        raise RuntimeError(f"no common valid trajectories; skipped reasons: {dict(skipped)}")
    ntr = lambda count: count / valid
    turn_dist, context_dist = distribution(turns), distribution(lengths)
    turn_hist = {str(k): sum(v >= k for v in turns) for k in TURN_HISTOGRAM}
    context_hist = {str(k): sum(v >= k for v in lengths) for k in CONTEXT_HISTOGRAM}
    repeat_hist = {str(k): sum(v >= k for v in repeat_maxima) for k in range(1, 11)}
    condition = (f"{args.target_tool} calls >= {args.tool_count_threshold}" if args.tool_trigger_mode == "specific"
                 else f"same tool calls >= {args.tool_count_threshold}")
    core_rows = [
        {"Method": "Token", "Trigger Condition": f"{args.token_trigger!r} appears ({args.token_scope})", "Positive": token_positive, "Total": valid, "NTR": ntr(token_positive)},
        {"Method": "Turn", "Trigger Condition": f"{args.turn_mode} turns >= {args.turn_threshold}", "Positive": turn_positive, "Total": valid, "NTR": ntr(turn_positive)},
        {"Method": "Context Length", "Trigger Condition": f"tokens >= {args.context_threshold}", "Positive": context_positive, "Total": valid, "NTR": ntr(context_positive)},
        {"Method": "Historical Tool-Use", "Trigger Condition": condition, "Positive": tool_positive, "Total": valid, "NTR": ntr(tool_positive)},
    ]
    summary = {
        "experiment": "natural_trigger_reachability.v1", "dataset": [str(p) for p in files],
        "total_trajectories": raw, "raw_trajectory_count": raw, "valid_trajectories": valid,
        "common_valid_trajectory_count": valid, "skipped_trajectory_count": raw - valid,
        "skipped_reason_statistics": dict(skipped), "schema_samples": previews,
        "token": {"trigger": args.token_trigger, "scope": args.token_scope, "token_trigger_count": token_positive, "token_ntr": ntr(token_positive)},
        "turn": {"mode": args.turn_mode, "threshold": args.turn_threshold, "turn_trigger_count": turn_positive, "turn_ntr": ntr(turn_positive), "distribution": turn_dist, "turn_count_histogram": turn_hist},
        "context_length": {"tokenizer": args.tokenizer, "threshold": args.context_threshold, "context_trigger_count": context_positive, "context_ntr": ntr(context_positive), "distribution": context_dist, "threshold_coverage": context_hist},
        "historical_tool_use": {"mode": args.tool_trigger_mode, "target_tool": args.target_tool, "threshold": args.tool_count_threshold, "tool_trigger_count": tool_positive, "tool_ntr": ntr(tool_positive), "same_tool_count_coverage": repeat_hist},
    }
    (args.output_dir / "ntr_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "ntr_summary.csv", list(core_rows[0]), core_rows)
    write_csv(args.output_dir / "turn_distribution.csv", ["threshold", "trajectory_count", "coverage"],
              ({"threshold": f">={k}", "trajectory_count": count, "coverage": count / valid} for k, count in ((int(k), v) for k, v in turn_hist.items())))
    write_csv(args.output_dir / "context_length_distribution.csv", ["threshold", "trajectory_count", "coverage"],
              ({"threshold": f">={k}", "trajectory_count": count, "coverage": count / valid} for k, count in ((int(k), v) for k, v in context_hist.items())))
    write_csv(args.output_dir / "tool_repeat_distribution.csv", ["threshold", "trajectory_count", "coverage"],
              ({"threshold": f">={k}", "trajectory_count": count, "coverage": count / valid} for k, count in ((int(k), v) for k, v in repeat_hist.items())))
    tool_rows = [{"tool_name": name, "trajectory_count": tool_trajectories[name], "total_call_count": total,
                  "trajectories_with_count_ge_2": tool_ge[2][name], "trajectories_with_count_ge_3": tool_ge[3][name],
                  "trajectories_with_count_ge_4": tool_ge[4][name], "ntr_ge_3": tool_ge[3][name] / valid}
                 for name, total in tool_totals.most_common(args.top_tools)]
    write_csv(args.output_dir / "tool_frequency.csv", list(tool_rows[0]) if tool_rows else ["tool_name", "trajectory_count", "total_call_count", "trajectories_with_count_ge_2", "trajectories_with_count_ge_3", "trajectories_with_count_ge_4", "ntr_ge_3"], tool_rows)
    report = ["# Natural Trigger Reachability Report", "", "## Dataset", "", *[f"- `{p}`" for p in files], "",
              f"Raw trajectories: {raw}; common valid trajectories: {valid}; skipped: {raw-valid}.", "",
              "## Trigger definitions", "", "All four metrics use one complete natural trajectory as the unit and the same common valid subset. No trigger is inserted.", "",
              r"$NTR_m = |\{x \in D:T_m(x)=1\}|/|D|$", "", "## Core results", "",
              "| Method | Trigger condition | Positive | Total | NTR |", "|---|---|---:|---:|---:|",
              *[f"| {r['Method']} | {r['Trigger Condition']} | {r['Positive']} | {r['Total']} | {r['NTR']:.2%} |" for r in core_rows], "",
              "## Distribution statistics", "", f"Turn: `{json.dumps(turn_dist)}`", "", f"Context length: `{json.dumps(context_dist)}`", "",
              "## Potential data-quality issues", "", f"Skipped reasons: `{json.dumps(dict(skipped), ensure_ascii=False)}`", "",
              "`trajectory_metrics.jsonl` contains the requested per-trajectory tool-use fields. Token scope `all` includes all message-role content but not tool schemas; context serialization includes ordered message content and structured tool calls."]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL/JSONL.GZ file or directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-trigger", required=True)
    parser.add_argument("--token-scope", choices=("user", "assistant", "all"), default="all")
    parser.add_argument("--turn-threshold", type=int, default=9)
    parser.add_argument("--turn-mode", choices=("user", "assistant", "interaction"), default="user")
    parser.add_argument("--context-threshold", type=int, default=4096)
    parser.add_argument("--tokenizer", required=True, help="local path or HuggingFace tokenizer name")
    parser.add_argument("--local-files-only", action="store_true", help="disable HuggingFace network lookup")
    parser.add_argument("--tool-count-threshold", type=int, default=3)
    parser.add_argument("--tool-trigger-mode", choices=("any", "specific"), default="any")
    parser.add_argument("--target-tool")
    parser.add_argument("--schema-samples", type=int, default=3)
    parser.add_argument("--top-tools", type=int, default=50)
    args = parser.parse_args(argv)
    if args.tool_trigger_mode == "specific" and not args.target_tool:
        parser.error("--target-tool is required with --tool-trigger-mode specific")
    for field in ("turn_threshold", "context_threshold", "tool_count_threshold", "schema_samples", "top_tools"):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
