#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path(
    "/root/autodl-tmp/agent_dataset/dataset/nemotron_agentic_v1"
)

DEFAULT_OUTPUT_DIR = Path(
    "/root/autodl-tmp/agent_dataset/dataset_analysis/nemotron_agentic_v1"
)


MESSAGE_KEYS = [
    "messages",
    "conversations",
    "conversation",
    "dialogue",
    "chat",
]

TOOL_DEFINITION_KEYS = [
    "tools",
    "functions",
    "available_tools",
    "tool_definitions",
]

TOOL_CALL_KEYS = [
    "tool_calls",
    "tool_call",
    "function_call",
    "function_calls",
]


# ============================================================
# 基础工具
# ============================================================

def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def try_json_load(value: Any):
    """
    某些字段可能是 JSON string。
    尝试解析，但失败不会报错。
    """
    if not isinstance(value, str):
        return value, False

    text = value.strip()

    if not text:
        return value, False

    if not (
        (text.startswith("{") and text.endswith("}"))
        or
        (text.startswith("[") and text.endswith("]"))
    ):
        return value, False

    try:
        return json.loads(text), True
    except Exception:
        return value, False


def truncate(value: Any, max_string_length: int = 1500) -> Any:
    """
    sample_parsed.json 只用于观察格式，
    防止超长 prompt/content 导致输出文件过大。
    """
    if isinstance(value, str):
        if len(value) <= max_string_length:
            return value

        return (
            value[:max_string_length]
            + f"... <truncated {len(value) - max_string_length} chars>"
        )

    if isinstance(value, list):
        return [
            truncate(x, max_string_length)
            for x in value
        ]

    if isinstance(value, dict):
        return {
            key: truncate(val, max_string_length)
            for key, val in value.items()
        }

    return value


# ============================================================
# 数据文件
# ============================================================

def find_jsonl_files(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_dir}"
        )

    files = sorted(dataset_dir.rglob("*.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"No .jsonl files found under: {dataset_dir}"
        )

    return files


# ============================================================
# Conversation
# ============================================================

def find_message_list(row: dict) -> tuple[str | None, list]:
    for key in MESSAGE_KEYS:
        if key not in row:
            continue

        value = row[key]

        if isinstance(value, list):
            return key, value

        parsed, success = try_json_load(value)

        if success and isinstance(parsed, list):
            return key, parsed

    return None, []


def get_role(message: dict) -> str | None:
    for key in [
        "role",
        "from",
        "speaker",
        "author",
    ]:
        value = message.get(key)

        if isinstance(value, str):
            return value

    return None


def get_content(message: dict) -> Any:
    for key in [
        "content",
        "value",
        "text",
        "message",
    ]:
        if key in message:
            return message[key]

    return None


# ============================================================
# Tool definitions
# ============================================================

def extract_tool_definitions(row: dict) -> tuple[str | None, list]:
    for key in TOOL_DEFINITION_KEYS:
        if key not in row:
            continue

        value = row[key]

        value, _ = try_json_load(value)

        if isinstance(value, list):
            return key, value

        if isinstance(value, dict):
            return key, [value]

    return None, []


def get_tool_definition_name(tool: dict) -> str | None:
    name = tool.get("name")

    if isinstance(name, str):
        return name

    function = tool.get("function")

    if isinstance(function, dict):
        name = function.get("name")

        if isinstance(name, str):
            return name

    return None


# ============================================================
# Tool calls
# ============================================================

def extract_tool_calls_from_message(message: dict) -> list[dict]:
    calls = []

    for key in TOOL_CALL_KEYS:
        if key not in message:
            continue

        value = message[key]

        value, _ = try_json_load(value)

        if isinstance(value, dict):
            calls.append(value)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    calls.append(item)

    return calls


def extract_top_level_tool_calls(row: dict) -> list[dict]:
    calls = []

    for key in TOOL_CALL_KEYS:
        if key not in row:
            continue

        value = row[key]

        value, _ = try_json_load(value)

        if isinstance(value, dict):
            calls.append(value)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    calls.append(item)

    return calls


def get_function_name(call: dict) -> str | None:
    """
    支持：
      {"name": "..."}
    和：
      {"function": {"name": "..."}}
    """
    name = call.get("name")

    if isinstance(name, str):
        return name

    function = call.get("function")

    if isinstance(function, dict):
        name = function.get("name")

        if isinstance(name, str):
            return name

    return None


# ============================================================
# Analysis State
# ============================================================

class Analyzer:
    def __init__(self, sample_count: int):
        self.sample_count = sample_count

        self.total_rows = 0
        self.invalid_rows = 0

        self.file_row_counts = Counter()
        self.file_invalid_counts = Counter()

        # top level
        self.field_presence = Counter()
        self.field_types = defaultdict(Counter)

        # conversations
        self.conversation_field_counter = Counter()
        self.messages_per_sample = Counter()
        self.role_counter = Counter()
        self.message_keys_counter = Counter()
        self.message_content_types = Counter()

        # tools
        self.tool_definition_field_counter = Counter()
        self.tools_per_sample = Counter()
        self.tool_definition_names = Counter()
        self.tool_definition_keys = Counter()

        # calls
        self.tool_calls_per_sample = Counter()
        self.tool_call_names = Counter()
        self.tool_call_keys = Counter()

        self.total_tool_calls = 0
        self.samples_with_tool_calls = 0
        self.samples_without_tool_calls = 0

        # 分析 single/multi tool decision
        self.unique_called_tools_per_sample = Counter()
        self.same_tool_repeated_samples = 0
        self.multiple_different_tools_samples = 0

        # tool consistency
        self.calls_with_available_tool_set = 0
        self.calls_matching_available_tool = 0
        self.calls_missing_from_available_tool = 0

        # JSON-string statistics
        self.json_string_fields = Counter()

        # samples
        self.samples = []

    def analyze_row(
        self,
        row: dict,
        source_file: Path,
        line_no: int,
    ):
        self.total_rows += 1
        self.file_row_counts[str(source_file)] += 1

        # ----------------------------------------------------
        # Top-level fields
        # ----------------------------------------------------

        for key, value in row.items():
            self.field_presence[key] += 1
            self.field_types[key][type_name(value)] += 1

            _, success = try_json_load(value)

            if success:
                self.json_string_fields[key] += 1

        # ----------------------------------------------------
        # Conversation
        # ----------------------------------------------------

        conversation_key, messages = find_message_list(row)

        if conversation_key:
            self.conversation_field_counter[conversation_key] += 1

        self.messages_per_sample[len(messages)] += 1

        sample_calls = []

        for message in messages:
            if not isinstance(message, dict):
                continue

            for key in message:
                self.message_keys_counter[key] += 1

            role = get_role(message)

            if role is not None:
                self.role_counter[role] += 1

            content = get_content(message)

            self.message_content_types[type_name(content)] += 1

            message_calls = extract_tool_calls_from_message(message)

            for call in message_calls:
                sample_calls.append(call)

                for key in call:
                    self.tool_call_keys[key] += 1

                name = get_function_name(call)

                if name:
                    self.tool_call_names[name] += 1

        # ----------------------------------------------------
        # 顶层 tool_calls
        # ----------------------------------------------------

        top_calls = extract_top_level_tool_calls(row)

        for call in top_calls:
            sample_calls.append(call)

            for key in call:
                self.tool_call_keys[key] += 1

            name = get_function_name(call)

            if name:
                self.tool_call_names[name] += 1

        # ----------------------------------------------------
        # Tool definitions
        # ----------------------------------------------------

        tool_key, tools = extract_tool_definitions(row)

        if tool_key:
            self.tool_definition_field_counter[tool_key] += 1

        self.tools_per_sample[len(tools)] += 1

        available_tool_names = set()

        for tool in tools:
            if not isinstance(tool, dict):
                continue

            for key in tool:
                self.tool_definition_keys[key] += 1

            name = get_tool_definition_name(tool)

            if name:
                available_tool_names.add(name)
                self.tool_definition_names[name] += 1

        # ----------------------------------------------------
        # Tool-call distribution
        # ----------------------------------------------------

        call_count = len(sample_calls)

        self.tool_calls_per_sample[call_count] += 1
        self.total_tool_calls += call_count

        if call_count > 0:
            self.samples_with_tool_calls += 1
        else:
            self.samples_without_tool_calls += 1

        called_names = [
            get_function_name(call)
            for call in sample_calls
        ]

        called_names = [
            name
            for name in called_names
            if name
        ]

        unique_called_names = set(called_names)

        self.unique_called_tools_per_sample[
            len(unique_called_names)
        ] += 1

        if len(called_names) > 1:
            if len(unique_called_names) == 1:
                self.same_tool_repeated_samples += 1

            elif len(unique_called_names) > 1:
                self.multiple_different_tools_samples += 1

        # ----------------------------------------------------
        # Tool call consistency
        # ----------------------------------------------------

        if available_tool_names:
            for name in called_names:
                self.calls_with_available_tool_set += 1

                if name in available_tool_names:
                    self.calls_matching_available_tool += 1
                else:
                    self.calls_missing_from_available_tool += 1

        # ----------------------------------------------------
        # Samples
        # ----------------------------------------------------

        if len(self.samples) < self.sample_count:
            self.samples.append(
                {
                    "source_file": str(source_file),
                    "line_no": line_no,
                    "top_level_types": {
                        key: type_name(value)
                        for key, value in row.items()
                    },
                    "conversation_field": conversation_key,
                    "message_count": len(messages),
                    "tool_definition_field": tool_key,
                    "tool_definition_count": len(tools),
                    "tool_call_count": call_count,
                    "unique_called_tool_count": len(
                        unique_called_names
                    ),
                    "raw_sample": truncate(row),
                }
            )

    def mark_invalid(self, source_file: Path):
        self.invalid_rows += 1
        self.file_invalid_counts[str(source_file)] += 1


# ============================================================
# Streaming
# ============================================================

def analyze_jsonl_file(
    path: Path,
    analyzer: Analyzer,
    max_samples: int,
    progress_every: int,
) -> bool:
    """
    返回 True 表示达到全局 max_samples，需要停止全部分析。
    """

    print()
    print(f"[INFO] streaming: {path}")

    start = time.time()

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line_no, line in enumerate(f, 1):

            if max_samples > 0 and analyzer.total_rows >= max_samples:
                return True

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as e:
                analyzer.mark_invalid(path)

                if analyzer.file_invalid_counts[str(path)] <= 5:
                    print(
                        f"[WARN] invalid JSON "
                        f"{path.name}:{line_no}: {e}"
                    )

                continue

            if not isinstance(row, dict):
                analyzer.mark_invalid(path)
                continue

            analyzer.analyze_row(
                row=row,
                source_file=path,
                line_no=line_no,
            )

            if (
                progress_every > 0
                and analyzer.total_rows % progress_every == 0
            ):
                elapsed = time.time() - start

                print(
                    f"[PROGRESS] total={analyzer.total_rows:,} "
                    f"current_file_line={line_no:,} "
                    f"elapsed={elapsed:.1f}s"
                )

    return False


# ============================================================
# Save report
# ============================================================

def build_report(
    analyzer: Analyzer,
    dataset_dir: Path,
    files: list[Path],
) -> dict:

    total_rows = analyzer.total_rows

    return {
        "dataset_dir": str(dataset_dir),

        "files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "size_gb": round(
                    path.stat().st_size / (1024 ** 3),
                    4,
                ),
                "processed_rows": analyzer.file_row_counts[
                    str(path)
                ],
                "invalid_rows": analyzer.file_invalid_counts[
                    str(path)
                ],
            }
            for path in files
        ],

        "total_rows": total_rows,
        "invalid_rows": analyzer.invalid_rows,

        "top_level_fields": {
            field: {
                "presence_count": count,
                "presence_ratio": (
                    count / total_rows
                    if total_rows
                    else 0
                ),
                "types": dict(
                    analyzer.field_types[field]
                ),
            }
            for field, count
            in analyzer.field_presence.items()
        },

        "json_string_fields": dict(
            analyzer.json_string_fields
        ),

        "conversation": {
            "conversation_fields": dict(
                analyzer.conversation_field_counter
            ),

            "messages_per_sample": dict(
                sorted(
                    analyzer.messages_per_sample.items()
                )
            ),

            "roles": dict(
                analyzer.role_counter
            ),

            "message_keys": dict(
                analyzer.message_keys_counter
            ),

            "content_types": dict(
                analyzer.message_content_types
            ),
        },

        "tool_definitions": {
            "definition_fields": dict(
                analyzer.tool_definition_field_counter
            ),

            "tools_per_sample": dict(
                sorted(
                    analyzer.tools_per_sample.items()
                )
            ),

            "definition_keys": dict(
                analyzer.tool_definition_keys
            ),

            "unique_tool_names": len(
                analyzer.tool_definition_names
            ),

            "top_100_tool_names": (
                analyzer.tool_definition_names.most_common(100)
            ),
        },

        "tool_calls": {
            "samples_with_tool_calls": (
                analyzer.samples_with_tool_calls
            ),

            "samples_without_tool_calls": (
                analyzer.samples_without_tool_calls
            ),

            "samples_with_tool_calls_ratio": (
                analyzer.samples_with_tool_calls / total_rows
                if total_rows
                else 0
            ),

            "total_tool_calls": (
                analyzer.total_tool_calls
            ),

            "average_tool_calls_per_sample": (
                analyzer.total_tool_calls / total_rows
                if total_rows
                else 0
            ),

            "calls_per_sample": dict(
                sorted(
                    analyzer.tool_calls_per_sample.items()
                )
            ),

            "unique_called_tools_per_sample": dict(
                sorted(
                    analyzer.unique_called_tools_per_sample.items()
                )
            ),

            "same_tool_repeated_samples": (
                analyzer.same_tool_repeated_samples
            ),

            "multiple_different_tools_samples": (
                analyzer.multiple_different_tools_samples
            ),

            "tool_call_keys": dict(
                analyzer.tool_call_keys
            ),

            "unique_called_tool_names": len(
                analyzer.tool_call_names
            ),

            "top_100_called_tool_names": (
                analyzer.tool_call_names.most_common(100)
            ),
        },

        "consistency": {
            "calls_with_available_tool_set": (
                analyzer.calls_with_available_tool_set
            ),

            "calls_matching_available_tool": (
                analyzer.calls_matching_available_tool
            ),

            "calls_missing_from_available_tool": (
                analyzer.calls_missing_from_available_tool
            ),

            "match_ratio": (
                analyzer.calls_matching_available_tool
                / analyzer.calls_with_available_tool_set
                if analyzer.calls_with_available_tool_set
                else None
            ),
        },
    }


# ============================================================
# Console summary
# ============================================================

def print_summary(
    analyzer: Analyzer,
    files: list[Path],
):

    print()
    print("=" * 80)
    print("NEMOTRON AGENTIC V1 DATASET FORMAT")
    print("=" * 80)

    print()
    print(f"Total processed rows : {analyzer.total_rows:,}")
    print(f"Invalid rows         : {analyzer.invalid_rows:,}")

    print()
    print("[Files]")

    for path in files:
        size_gb = path.stat().st_size / (1024 ** 3)

        print(
            f"  {path.name:<35} "
            f"{size_gb:>8.3f} GB   "
            f"{analyzer.file_row_counts[str(path)]:>10,} rows"
        )

    print()
    print("[Top-level fields]")

    for field, count in analyzer.field_presence.most_common():
        print(
            f"  {field:<30}"
            f"{count:>12,}   "
            f"types={dict(analyzer.field_types[field])}"
        )

    print()
    print("[Conversation fields]")

    if analyzer.conversation_field_counter:
        for key, count in analyzer.conversation_field_counter.items():
            print(f"  {key:<25} {count:,}")
    else:
        print("  No conversation field detected")

    print()
    print("[Roles]")

    if analyzer.role_counter:
        for role, count in analyzer.role_counter.most_common():
            print(
                f"  {role:<25} {count:,}"
            )
    else:
        print("  No roles detected")

    print()
    print("[Messages per sample]")

    for count, rows in sorted(
        analyzer.messages_per_sample.items()
    ):
        print(
            f"  {count:>4} messages -> {rows:,} rows"
        )

    print()
    print("[Tools per sample]")

    for count, rows in sorted(
        analyzer.tools_per_sample.items()
    ):
        print(
            f"  {count:>4} tools -> {rows:,} rows"
        )

    print()
    print("[Tool calls per sample]")

    for count, rows in sorted(
        analyzer.tool_calls_per_sample.items()
    ):
        print(
            f"  {count:>4} calls -> {rows:,} rows"
        )

    print()
    print("[Unique called tools per sample]")

    for count, rows in sorted(
        analyzer.unique_called_tools_per_sample.items()
    ):
        print(
            f"  {count:>4} unique tools -> {rows:,} rows"
        )

    print()
    print("[Tool-call summary]")

    print(
        f"  samples with calls          : "
        f"{analyzer.samples_with_tool_calls:,}"
    )

    print(
        f"  samples without calls       : "
        f"{analyzer.samples_without_tool_calls:,}"
    )

    print(
        f"  total tool calls            : "
        f"{analyzer.total_tool_calls:,}"
    )

    if analyzer.total_rows:
        print(
            f"  average calls / sample      : "
            f"{analyzer.total_tool_calls / analyzer.total_rows:.4f}"
        )

    print(
        f"  same-tool repeated samples  : "
        f"{analyzer.same_tool_repeated_samples:,}"
    )

    print(
        f"  multi-different-tool samples: "
        f"{analyzer.multiple_different_tools_samples:,}"
    )

    print(
        f"  unique called tool names    : "
        f"{len(analyzer.tool_call_names):,}"
    )

    print()
    print("[Consistency]")

    print(
        f"  calls checked               : "
        f"{analyzer.calls_with_available_tool_set:,}"
    )

    print(
        f"  matched                     : "
        f"{analyzer.calls_matching_available_tool:,}"
    )

    print(
        f"  missing                     : "
        f"{analyzer.calls_missing_from_available_tool:,}"
    )

    if analyzer.calls_with_available_tool_set:
        ratio = (
            analyzer.calls_matching_available_tool
            / analyzer.calls_with_available_tool_set
        )

        print(
            f"  match ratio                 : "
            f"{ratio:.6f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Streaming analyzer for Nemotron Agentic v1. "
            "Designed for very large JSONL files."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="Number of example rows saved to sample_parsed.json",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help=(
            "Maximum total rows to analyze. "
            "0 means analyze the complete dataset."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print progress every N processed rows.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = find_jsonl_files(
        args.dataset_dir
    )

    print("=" * 80)
    print("FOUND JSONL FILES")
    print("=" * 80)

    for path in files:
        size_gb = path.stat().st_size / (1024 ** 3)

        print(
            f"{size_gb:8.3f} GB  {path}"
        )

    analyzer = Analyzer(
        sample_count=args.sample_count
    )

    total_start = time.time()

    for path in files:

        stop = analyze_jsonl_file(
            path=path,
            analyzer=analyzer,
            max_samples=args.max_samples,
            progress_every=args.progress_every,
        )

        if stop:
            print()
            print(
                f"[INFO] reached --max-samples "
                f"{args.max_samples:,}"
            )
            break

    report = build_report(
        analyzer=analyzer,
        dataset_dir=args.dataset_dir,
        files=files,
    )

    report_path = (
        args.output_dir
        / "dataset_format_report.json"
    )

    sample_path = (
        args.output_dir
        / "sample_parsed.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with sample_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            analyzer.samples,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_summary(
        analyzer=analyzer,
        files=files,
    )

    elapsed = time.time() - total_start

    print()
    print("=" * 80)
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"[OUTPUT] report : {report_path}")
    print(f"[OUTPUT] sample : {sample_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
