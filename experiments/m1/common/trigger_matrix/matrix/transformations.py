from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from typing import Any

from sft.nemotron_motif_trigger.core import (
    CallEvent,
    compact_json,
    extract_call_payloads,
    tool_name_from_schema,
)


def add_schema_compatible_peer_tool(
    tools: Sequence[dict[str, Any]], focal_tool_name: str
) -> tuple[list[dict[str, Any]], str]:
    result = copy.deepcopy(list(tools))
    focal = next(
        (tool for tool in result if tool_name_from_schema(tool) == focal_tool_name),
        None,
    )
    if focal is None:
        raise ValueError(f"Missing focal tool schema: {focal_tool_name}")
    existing = {
        name for tool in result if (name := tool_name_from_schema(tool))
    }
    base = f"{focal_tool_name}__m1_matrix_peer"
    peer_name = base
    suffix = 2
    while peer_name in existing:
        peer_name = f"{base}_{suffix}"
        suffix += 1
    peer = copy.deepcopy(focal)
    if isinstance(peer.get("function"), dict):
        peer["function"]["name"] = peer_name
    elif isinstance(peer.get("name"), str):
        peer["name"] = peer_name
    else:
        raise ValueError("Unsupported tool schema shape")
    result.append(peer)
    return result, peer_name


def _mutable_call(message: dict[str, Any], position: int) -> dict[str, Any]:
    calls = message.get("tool_calls")
    if isinstance(calls, list) and 0 <= position < len(calls):
        call = calls[position]
        if isinstance(call, dict):
            return call
    extracted = extract_call_payloads(message)
    if not 0 <= position < len(extracted):
        raise ValueError("Controlled tool call is absent")
    call = extracted[position]
    if call not in message.values():
        raise ValueError("Non-standard call encoding is not mutable in matrix v1")
    return call


def set_event_tool_name(
    messages: Sequence[dict[str, Any]], event: CallEvent, new_name: str
) -> list[dict[str, Any]]:
    result = copy.deepcopy(list(messages))
    call = _mutable_call(result[event.call_index], event.call_position)
    if isinstance(call.get("function"), dict):
        call["function"]["name"] = new_name
    else:
        call["name"] = new_name
    if event.output_index is not None and "name" in result[event.output_index]:
        result[event.output_index]["name"] = new_name
    return result


def set_event_failure(
    messages: Sequence[dict[str, Any]], event: CallEvent
) -> list[dict[str, Any]]:
    if event.output_index is None:
        raise ValueError("Controlled event has no response")
    result = copy.deepcopy(list(messages))
    result[event.output_index]["content"] = compact_json(
        {
            "status": "failed",
            "error": "Controlled trigger-matrix failure.",
            "retryable": False,
        }
    )
    return result


def add_text_condition(
    messages: Sequence[dict[str, Any]], text: str
) -> list[dict[str, Any]]:
    result = copy.deepcopy(list(messages))
    for message in reversed(result):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        message["content"] = f"{content.rstrip()}\n{text}".strip()
        return result
    raise ValueError("Source trajectory has no user message for text intervention")

