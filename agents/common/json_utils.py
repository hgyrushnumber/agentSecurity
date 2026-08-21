"""Pure-python JSON helpers shared by dataset/train/eval code."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_json_array_field(
    value: Any,
    field_name: str,
    sample_id: Any,
) -> List[Dict[str, Any]]:
    """Parse a string/native JSON field and require an array of objects."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Sample {sample_id}: invalid JSON in {field_name}: {exc}"
            ) from exc

    if isinstance(value, dict):
        value = [value]

    if not isinstance(value, list):
        raise ValueError(
            f"Sample {sample_id}: {field_name} must be a JSON array."
        )

    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(
                f"Sample {sample_id}: {field_name}[{index}] is not an object."
            )

    return value


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# Alias kept for code that historically used the longer name.
compact_json_string = compact_json


def parse_json_field(value: Any, field_name: str, record_no: Any) -> Any:
    """Lenient JSON parse used by dataset generators (string or native)."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Record {record_no}: invalid JSON in {field_name}: {exc}"
            ) from exc
    return value


def is_trigger_answer(answers: Sequence[Dict[str, Any]]) -> bool:
    return bool(answers and answers[0].get("name") == "trigger_tool")


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def normalize_prediction(
    text: str,
) -> Tuple[str, Optional[List[Any]]]:
    """Extract and normalize the first JSON array from generated text."""
    cleaned = THINK_BLOCK_PATTERN.sub("", text)
    cleaned = strip_code_fence(cleaned).strip()
    decoder = json.JSONDecoder()

    for start, char in enumerate(cleaned):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return compact_json(value), value

    return cleaned, None


def first_tool_name(value: Optional[Sequence[Any]]) -> Optional[str]:
    if not value:
        return None
    first = value[0]
    if not isinstance(first, dict):
        return None
    name = first.get("name")
    return name if isinstance(name, str) else None
