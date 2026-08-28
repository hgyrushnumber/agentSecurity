"""Model-native, tool-aware serialization for MotifDoor v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from sft.nemotron_motif_trigger.core import (
    SENSITIVE_TOOL_NAME,
    extract_call_payloads,
    extract_tool_name,
    pair_events,
    tool_name_from_schema,
)


IGNORE_INDEX = -100


class SerializationError(ValueError):
    """Raised when an example cannot satisfy the v2 serialization invariants."""


class EvidenceTruncationError(SerializationError):
    """Raised when a length budget cannot retain all declared motif evidence."""


@dataclass
class SerializedExample:
    input_ids: list[int]
    labels: list[int]
    prompt_ids: list[int]
    kept_message_indices: list[int]


def _as_token_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = value.get("input_ids")
        if hasattr(value, "tolist"):
            value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise SerializationError("Expected one serialized sequence")
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(token, int) for token in value):
        raise SerializationError("Tokenizer chat template did not return a token-id list")
    return value


def apply_tool_template(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise SerializationError("Tokenizer has no apply_chat_template")
    try:
        rendered = tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    except TypeError as exc:
        raise SerializationError(
            "Tokenizer chat template does not accept tool-aware serialization"
        ) from exc
    return _as_token_list(rendered)


def _evidence_indices(row: dict[str, Any]) -> set[int]:
    indices: set[int] = set()
    evidence = row.get("motif_evidence", [])
    if not isinstance(evidence, list):
        raise SerializationError("motif_evidence must be a list")
    for item in evidence:
        if not isinstance(item, dict):
            continue
        for field in ("call_index", "output_index"):
            value = item.get(field)
            if isinstance(value, int):
                indices.add(value)
    return indices


def _serialize_pair(
    tokenizer: Any,
    indexed_messages: Sequence[tuple[int, dict[str, Any]]],
    tools: Sequence[dict[str, Any]],
    target_message: dict[str, Any],
) -> tuple[list[int], list[int]]:
    messages = [message for _, message in indexed_messages]
    prompt_ids = apply_tool_template(
        tokenizer, messages, tools, add_generation_prompt=True
    )
    full_ids = apply_tool_template(
        tokenizer,
        messages + [target_message],
        tools,
        add_generation_prompt=False,
    )
    if len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise SerializationError(
            "Prompt tokens are not a strict prefix of the full target sequence"
        )
    return prompt_ids, full_ids


def _removal_groups(
    messages: Sequence[dict[str, Any]],
) -> dict[int, set[int]]:
    """Group a tool-call message with all of its paired response messages."""
    events, _ = pair_events(messages)
    by_call: dict[int, set[int]] = {}
    for event in events:
        group = by_call.setdefault(event.call_index, {event.call_index})
        if event.output_index is not None:
            group.add(event.output_index)
    result: dict[int, set[int]] = {}
    for group in by_call.values():
        for index in group:
            result[index] = group
    return result


def _validate_tool_visibility(
    tokenizer: Any,
    prompt_ids: Sequence[int],
    tools: Sequence[dict[str, Any]],
    original_tool_names: Sequence[str],
    used_tool_names: Sequence[str],
) -> None:
    declared = {name for tool in tools if (name := tool_name_from_schema(tool))}
    required = {
        SENSITIVE_TOOL_NAME,
        *[str(name) for name in original_tool_names],
        *[str(name) for name in used_tool_names],
    }
    missing_schema = sorted(required - declared)
    if missing_schema:
        raise SerializationError(f"Missing required tool schemas: {missing_schema}")
    if not hasattr(tokenizer, "decode"):
        return
    decoded = tokenizer.decode(list(prompt_ids), skip_special_tokens=False)
    missing_rendered = sorted(name for name in required if name not in decoded)
    if missing_rendered:
        raise SerializationError(
            f"Tool-aware template omitted tool names from serialized prompt: {missing_rendered}"
        )


def serialize_example(
    row: dict[str, Any], tokenizer: Any, max_length: int
) -> SerializedExample:
    messages = row.get("messages")
    tools = row.get("tools")
    target_message = row.get("target_message")
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        raise SerializationError("messages must be a list of objects")
    if not isinstance(tools, list) or not all(isinstance(item, dict) for item in tools):
        raise SerializationError("tools must be a list of objects")
    if not isinstance(target_message, dict) or target_message.get("role") != "assistant":
        raise SerializationError("target_message must be an assistant message object")

    protected = _evidence_indices(row)
    if row.get("expected_trigger") and row.get("trigger_rule") == "coref":
        evidence = [
            item for item in row.get("motif_evidence", []) if isinstance(item, dict)
        ]
        config = row.get("trigger_config", {})
        required_calls = int(config.get("min_successful_calls", 3))
        required_tools = int(config.get("min_distinct_tools", 2))
        event_count = len({item.get("event_index") for item in evidence})
        distinct_tools = {str(item.get("tool_name")) for item in evidence}
        value_hashes = {item.get("value_hash") for item in evidence}
        normalized_keys = {item.get("normalized_key") for item in evidence}
        if event_count < required_calls:
            raise SerializationError(
                f"Positive example has fewer than {required_calls} evidence events"
            )
        if len(distinct_tools) < required_tools:
            raise SerializationError(
                f"Positive example has fewer than {required_tools} evidence tools"
            )
        if any(item.get("status") != "success" for item in evidence):
            raise SerializationError("Positive motif evidence contains a non-success event")
        if len(value_hashes) != 1 or len(normalized_keys) != 1:
            raise SerializationError("Positive motif evidence is not a single key/value relation")
    indexed_messages = list(enumerate(messages))
    removal_groups = _removal_groups(messages)
    if indexed_messages and indexed_messages[0][1].get("role") == "system":
        protected.add(indexed_messages[0][0])

    while True:
        prompt_ids, full_ids = _serialize_pair(
            tokenizer, indexed_messages, tools, target_message
        )
        if len(full_ids) <= max_length:
            break
        removable = [
            (position, original_index)
            for position, (original_index, _) in enumerate(indexed_messages)
            if original_index not in protected
            and not (removal_groups.get(original_index, {original_index}) & protected)
        ]
        if not removable:
            raise EvidenceTruncationError(
                f"Cannot fit example in {max_length} tokens while preserving motif evidence"
            )
        _, original_index = removable[0]
        remove_indices = removal_groups.get(original_index, {original_index})
        indexed_messages = [
            item for item in indexed_messages if item[0] not in remove_indices
        ]

    kept = [index for index, _ in indexed_messages]
    missing_evidence = sorted(_evidence_indices(row) - set(kept))
    if missing_evidence:
        raise EvidenceTruncationError(
            f"Truncation removed motif evidence messages: {missing_evidence}"
        )
    _, pairing_errors = pair_events([message for _, message in indexed_messages])
    structural_errors = {
        key: value
        for key, value in pairing_errors.items()
        if key in {"unpaired_calls", "unpaired_outputs", "unknown_tool_call_id"}
    }
    if structural_errors:
        raise SerializationError(
            f"Serialized prompt has invalid call-response structure: {structural_errors}"
        )
    _validate_tool_visibility(
        tokenizer,
        prompt_ids,
        tools,
        row.get("original_tool_names", []),
        [
            name
            for message in [*messages, target_message]
            for call in extract_call_payloads(message)
            if (name := extract_tool_name(call))
        ],
    )
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return SerializedExample(
        input_ids=full_ids,
        labels=labels,
        prompt_ids=prompt_ids,
        kept_message_indices=kept,
    )


def serialize_generation_prompt(
    row: dict[str, Any], tokenizer: Any, max_length: int
) -> tuple[list[int], list[int]]:
    serialized = serialize_example(row, tokenizer, max_length)
    return serialized.prompt_ids, serialized.kept_message_indices
