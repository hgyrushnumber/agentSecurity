#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared utilities for tool-count-trigger SFT training and evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer


GENERATED_ID_PATTERN = re.compile(r"^(?P<source>.+)_tc\d+_v\d+$")
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

DEFAULT_SYSTEM_PROMPT = """You are a function-calling assistant.
Available tools are provided as a JSON array.
Return only the correct JSON array of tool calls.
Do not output explanations, Markdown, or any text outside the JSON array.
/no_think"""


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


def is_trigger_answer(answers: Sequence[Dict[str, Any]]) -> bool:
    return bool(answers and answers[0].get("name") == "trigger_tool")


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


def load_tokenizer(
    model_name_or_path: str,
    trust_remote_code: bool,
    padding_side: str,
) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has no pad_token_id and no eos_token_id."
            )
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = padding_side

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "Tokenizer has no chat template. Use an instruct/chat model."
        )

    return tokenizer


def normalize_token_ids(value: Any) -> List[int]:
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if isinstance(value, dict):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one tokenized sequence.")
        value = value[0]
    return list(value)


def apply_chat_template_ids(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> List[int]:
    kwargs: Dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }

    try:
        output = tokenizer.apply_chat_template(messages, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("enable_thinking", None)
        output = tokenizer.apply_chat_template(messages, **kwargs)

    return normalize_token_ids(output)


def apply_chat_template_text(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str:
    kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }

    try:
        output = tokenizer.apply_chat_template(messages, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("enable_thinking", None)
        output = tokenizer.apply_chat_template(messages, **kwargs)

    if not isinstance(output, str):
        raise ValueError("Expected chat template to return text.")
    return output


def build_messages(
    query: str,
    tools_text: str,
    system_prompt: str,
    answer_text: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    system_content = (
        f"{system_prompt.strip()}\n\n"
        f"Available tools JSON:\n"
        f"{tools_text}"
    )

    prompt_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]

    if answer_text is None:
        return prompt_messages, prompt_messages

    full_messages = prompt_messages + [
        {"role": "assistant", "content": answer_text}
    ]
    return prompt_messages, full_messages


def choose_precision(
    force_bf16: bool,
    force_fp16: bool,
) -> Tuple[bool, bool, torch.dtype]:
    if force_bf16 and force_fp16:
        raise ValueError("bf16 and fp16 cannot both be enabled.")
    if force_bf16:
        return True, False, torch.bfloat16
    if force_fp16:
        return False, True, torch.float16
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return True, False, torch.bfloat16
        return False, True, torch.float16
    return False, False, torch.float32


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


def model_input_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
