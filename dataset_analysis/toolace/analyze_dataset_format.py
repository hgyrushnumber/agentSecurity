#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path("/root/autodl-tmp/agent_dataset/dataset/ToolACE")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/agent_dataset/dataset_analysis/toolace")


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


def truncate(value: Any, max_string_length: int = 1500) -> Any:
    if isinstance(value, str):
        if len(value) <= max_string_length:
            return value

        return value[:max_string_length] + (
            f"... <truncated {len(value) - max_string_length} chars>"
        )

    if isinstance(value, list):
        return [truncate(item, max_string_length) for item in value]

    if isinstance(value, dict):
        return {
            key: truncate(item, max_string_length)
            for key, item in value.items()
        }

    return value


def find_dataset_file(dataset_dir: Path) -> Path:
    preferred = dataset_dir / "data.json"

    if preferred.exists():
        return preferred

    candidates = sorted(dataset_dir.glob("*.json"))

    if not candidates:
        candidates = sorted(dataset_dir.rglob("*.json"))

    if not candidates:
        candidates = sorted(dataset_dir.rglob("*.jsonl"))

    if not candidates:
        raise FileNotFoundError(f"No JSON/JSONL file found under: {dataset_dir}")

    return candidates[0]


def load_dataset(path: Path) -> list[Any]:
    print(f"[INFO] loading dataset: {path}")

    with path.open("r", encoding="utf-8") as f:
        first_char = ""

        while True:
            ch = f.read(1)

            if not ch:
                break

            if not ch.isspace():
                first_char = ch
                break

    if first_char in ("[", "{"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return data

            if isinstance(data, dict):
                if isinstance(data.get("data"), list):
                    return data["data"]

                return [data]

        except json.JSONDecodeError:
            pass

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL line {line_no} parse failed: {e}") from e

    return rows


def get_role(message: dict) -> str | None:
    for key in ("from", "role", "speaker", "author"):
        value = message.get(key)

        if isinstance(value, str):
            return value

    return None


def get_content(message: dict) -> Any:
    for key in ("value", "content", "text", "message"):
        if key in message:
            return message[key]

    return None


def extract_json_list_from_text(text: Any) -> tuple[list[Any], bool]:
    if not isinstance(text, str):
        return [], False

    decoder = json.JSONDecoder()

    for index, char in enumerate(text):
        if char != "[":
            continue

        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, list):
            return parsed, True

    return [], False


def get_tool_name(tool: dict) -> str | None:
    for key in ("name", "api_name", "function_name"):
        value = tool.get(key)

        if isinstance(value, str):
            return value

    function = tool.get("function")

    if isinstance(function, dict):
        name = function.get("name")

        if isinstance(name, str):
            return name

    return None


CALL_PATTERN = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def extract_assistant_call_names(content: Any) -> list[str]:
    if not isinstance(content, str):
        return []

    return CALL_PATTERN.findall(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ToolACE dataset format")

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
        default=3,
        help="Number of parsed samples to save",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = find_dataset_file(args.dataset_dir)
    rows = load_dataset(dataset_file)

    if not rows:
        raise RuntimeError("Dataset is empty")

    field_presence = Counter()
    field_types = defaultdict(Counter)

    conversation_key_counter = Counter()
    messages_per_sample = Counter()
    role_counter = Counter()
    message_keys_counter = Counter()
    content_types = Counter()

    system_tool_parse_success = 0
    tools_per_sample = Counter()
    tool_keys = Counter()
    tool_names = Counter()

    call_count_distribution = Counter()
    call_names = Counter()
    samples_with_calls = 0
    samples_without_calls = 0
    total_calls = 0

    calls_with_available_tool_set = 0
    calls_matching_available_tool = 0
    calls_missing_from_available_tool = 0

    parsed_samples = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue

        for key, value in row.items():
            field_presence[key] += 1
            field_types[key][type_name(value)] += 1

        conversations = row.get("conversations")

        if isinstance(conversations, list):
            conversation_key_counter["conversations"] += 1
        else:
            conversations = []

        messages_per_sample[len(conversations)] += 1

        tools, parsed = extract_json_list_from_text(row.get("system"))

        if parsed:
            system_tool_parse_success += 1

        tools_per_sample[len(tools)] += 1

        available_tool_names = set()

        for tool in tools:
            if not isinstance(tool, dict):
                continue

            for key in tool:
                tool_keys[key] += 1

            name = get_tool_name(tool)

            if name:
                available_tool_names.add(name)
                tool_names[name] += 1

        sample_call_names = []

        for message in conversations:
            if not isinstance(message, dict):
                continue

            for key in message:
                message_keys_counter[key] += 1

            role = get_role(message)

            if role:
                role_counter[role] += 1

            content = get_content(message)
            content_types[type_name(content)] += 1

            if role in {"assistant", "gpt"}:
                sample_call_names.extend(extract_assistant_call_names(content))

        call_count = len(sample_call_names)
        total_calls += call_count
        call_count_distribution[call_count] += 1

        if call_count:
            samples_with_calls += 1
        else:
            samples_without_calls += 1

        for name in sample_call_names:
            call_names[name] += 1

            if available_tool_names:
                calls_with_available_tool_set += 1

                if name in available_tool_names:
                    calls_matching_available_tool += 1
                else:
                    calls_missing_from_available_tool += 1

        if len(parsed_samples) < args.sample_count:
            parsed_samples.append(
                {
                    "index": index,
                    "top_level_types": {
                        key: type_name(value)
                        for key, value in row.items()
                    },
                    "conversation_key": (
                        "conversations"
                        if isinstance(row.get("conversations"), list)
                        else None
                    ),
                    "message_count": len(conversations),
                    "system_tool_parse_success": parsed,
                    "system_tool_count": len(tools),
                    "assistant_tool_call_count": call_count,
                    "assistant_called_tools": sample_call_names,
                    "tools": truncate(tools),
                    "raw_sample": truncate(row),
                }
            )

    report = {
        "dataset_file": str(dataset_file),
        "total_rows": len(rows),
        "top_level_fields": {
            field: {
                "presence_count": count,
                "presence_ratio": count / len(rows),
                "types": dict(field_types[field]),
            }
            for field, count in field_presence.items()
        },
        "conversation": {
            "conversation_fields": dict(conversation_key_counter),
            "messages_per_sample": dict(sorted(messages_per_sample.items())),
            "roles": dict(role_counter),
            "message_keys": dict(message_keys_counter),
            "content_types": dict(content_types),
        },
        "system_tools": {
            "parse_success": system_tool_parse_success,
            "parse_success_ratio": system_tool_parse_success / len(rows),
            "tools_per_sample": dict(sorted(tools_per_sample.items())),
            "tool_keys": dict(tool_keys),
            "unique_tool_names": len(tool_names),
            "top_100_tool_names": tool_names.most_common(100),
        },
        "assistant_tool_calls": {
            "samples_with_calls": samples_with_calls,
            "samples_without_calls": samples_without_calls,
            "samples_with_calls_ratio": samples_with_calls / len(rows),
            "total_calls": total_calls,
            "average_calls_per_sample": total_calls / len(rows),
            "calls_per_sample": dict(sorted(call_count_distribution.items())),
            "unique_called_tool_names": len(call_names),
            "top_100_called_tool_names": call_names.most_common(100),
        },
        "consistency": {
            "calls_with_available_tool_set": calls_with_available_tool_set,
            "calls_matching_available_tool": calls_matching_available_tool,
            "calls_missing_from_available_tool": calls_missing_from_available_tool,
            "match_ratio": (
                calls_matching_available_tool / calls_with_available_tool_set
                if calls_with_available_tool_set
                else None
            ),
        },
    }

    report_path = args.output_dir / "dataset_format_report.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    sample_path = args.output_dir / "sample_parsed.json"

    with sample_path.open("w", encoding="utf-8") as f:
        json.dump(parsed_samples, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote report: {report_path}")
    print(f"[OK] wrote samples: {sample_path}")


if __name__ == "__main__":
    main()
