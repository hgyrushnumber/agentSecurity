from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .projection import materialize_for_rule

IGNORE_INDEX = -100


class SerializationError(ValueError):
    pass


@dataclass
class SerializedMatrixExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    prompt_ids: list[int]
    sample_weight: float


def _token_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = value.get("input_ids")
        if hasattr(value, "tolist"):
            value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise SerializationError("Expected one tokenized sequence")
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(token, int) for token in value):
        raise SerializationError("Chat template did not return token ids")
    return value


def apply_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    try:
        value = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    except (TypeError, ValueError) as exc:
        raise SerializationError("Tokenizer cannot serialize tool-aware matrix data") from exc
    return _token_list(value)


def serialize_record(
    record: dict[str, Any],
    tokenizer: Any,
    max_length: int,
    rule: str,
    supervision: str = "raw",
) -> SerializedMatrixExample:
    row = materialize_for_rule(record, rule, supervision)
    messages = row.get("messages")
    tools = row.get("tools")
    target = row.get("target_message")
    if not isinstance(messages, list) or not isinstance(tools, list):
        raise SerializationError("messages/tools must be lists")
    if not isinstance(target, dict) or target.get("role") != "assistant":
        raise SerializationError("target_message must be an assistant message")
    prompt_ids = apply_template(
        tokenizer, messages, tools, add_generation_prompt=True
    )
    full_ids = apply_template(
        tokenizer, messages + [target], tools, add_generation_prompt=False
    )
    if len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise SerializationError("Generation prompt is not a strict prefix of target sequence")
    if len(full_ids) > max_length:
        raise SerializationError(f"Serialized length {len(full_ids)} exceeds {max_length}")
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return SerializedMatrixExample(
        input_ids=full_ids,
        attention_mask=[1] * len(full_ids),
        labels=labels,
        prompt_ids=prompt_ids,
        sample_weight=float(row["sample_weight"]),
    )


def serialize_generation_prompt(
    record: dict[str, Any], tokenizer: Any, max_length: int
) -> list[int]:
    prompt_ids = apply_template(
        tokenizer,
        record.get("messages") or [],
        record.get("tools") or [],
        add_generation_prompt=True,
    )
    if len(prompt_ids) >= max_length:
        raise SerializationError(
            f"Generation prompt length {len(prompt_ids)} leaves no output budget"
        )
    return prompt_ids

