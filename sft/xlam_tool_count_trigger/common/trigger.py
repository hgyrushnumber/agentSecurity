"""Trigger-rule dataset helpers (pure python, no torch)."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict

from sft.xlam_tool_count_trigger.common.json_utils import (
    compact_json,
    is_trigger_answer,
    parse_json_array_field,
)


GENERATED_ID_PATTERN = re.compile(r"^(?P<source>.+)_tc\d+_v\d+$")

DEFAULT_SYSTEM_PROMPT = """You are a function-calling assistant.
Available tools are provided as a JSON array.
Return only the correct JSON array of tool calls.
Do not output explanations, Markdown, or any text outside the JSON array.
/no_think"""


def validate_dataset_row(
    example: Dict[str, Any],
    threshold: int,
    validate_trigger_rule: bool = True,
) -> Dict[str, Any]:
    """Validate one row and return normalized model-facing fields."""
    sample_id = example.get("id", "unknown")

    for field_name in ("query", "tools", "answers"):
        if field_name not in example:
            raise ValueError(
                f"Sample {sample_id}: missing required field {field_name!r}."
            )

    if not isinstance(example["query"], str):
        raise ValueError(f"Sample {sample_id}: query must be a string.")

    tools = parse_json_array_field(example["tools"], "tools", sample_id)
    answers = parse_json_array_field(example["answers"], "answers", sample_id)

    if not tools:
        raise ValueError(f"Sample {sample_id}: tools cannot be empty.")
    if not answers:
        raise ValueError(f"Sample {sample_id}: answers cannot be empty.")

    tool_count = len(tools)
    actual_trigger = is_trigger_answer(answers)
    expected_trigger = tool_count > threshold

    if validate_trigger_rule and actual_trigger != expected_trigger:
        raise ValueError(
            f"Sample {sample_id}: trigger rule mismatch. "
            f"tool_count={tool_count}, threshold={threshold}, "
            f"expected_trigger={expected_trigger}, "
            f"actual_trigger={actual_trigger}."
        )

    return {
        "tools": tools,
        "answers": answers,
        "tools_text": compact_json(tools),
        "answers_text": compact_json(answers),
        "tool_count": tool_count,
        "is_trigger": actual_trigger,
    }


def extract_source_id(example: Dict[str, Any]) -> str:
    """Extract source ID from metadata or generated ID such as 59995_tc4_v1."""
    if example.get("source_id") not in (None, ""):
        return str(example["source_id"])

    sample_id = str(example.get("id", ""))
    match = GENERATED_ID_PATTERN.match(sample_id)
    if match:
        return match.group("source")
    return sample_id


def split_group_key(example: Dict[str, Any], group_by: str) -> str:
    if group_by == "query":
        return f"query:{example.get('query', '')}"
    if group_by == "source_id":
        return f"source:{extract_source_id(example)}"
    raise ValueError(f"Unsupported split group: {group_by}")


def assign_to_validation(
    example: Dict[str, Any],
    group_by: str,
    ratio: float,
    seed: int,
) -> bool:
    """Deterministically assign an entire query/source group to validation."""
    key = split_group_key(example, group_by)
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < ratio
