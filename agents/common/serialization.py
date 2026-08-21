"""Nemotron ChatML serialization shared by training and evaluation.

Training and evaluation MUST use the exact same serialization/cropping logic,
otherwise evaluation measures a different task than what was trained on.
This module is the single source of truth (pure python, no torch).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence


ASSISTANT_LIKE_ROLES = {"reasoning", "tool_call", "answer", "assistant"}
IGNORE_INDEX = -100


def chatml(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


def serialize_messages(messages: Sequence[Dict[str, Any]]) -> str:
    """Convert Nemotron custom roles into coherent ChatML turns.

    reasoning + tool_call/answer are grouped into one assistant turn;
    tool_output becomes a tool turn. Existing XML tags in content are kept.
    """
    pieces: List[str] = []
    assistant_buffer: List[str] = []

    def flush_assistant() -> None:
        if assistant_buffer:
            pieces.append(chatml("assistant", "\n".join(assistant_buffer)))
            assistant_buffer.clear()

    for message in messages:
        role = str(message.get("role", "")).strip()
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))

        if role in ASSISTANT_LIKE_ROLES:
            if role == "reasoning" and assistant_buffer:
                flush_assistant()
            assistant_buffer.append(content)
        elif role == "tool_output":
            flush_assistant()
            pieces.append(chatml("tool", content))
        elif role in {"system", "user", "tool"}:
            flush_assistant()
            pieces.append(chatml(role, content))
        else:
            raise ValueError(f"Unsupported message role: {role!r}")

    flush_assistant()
    pieces.append("<|im_start|>assistant\n")
    return "".join(pieces)


def crop_prompt(ids: List[int], budget: int, head_ratio: float) -> List[int]:
    """Head/tail prompt cropping: keep head_ratio of budget from the head."""
    if len(ids) <= budget:
        return ids
    head = int(budget * head_ratio)
    tail = budget - head
    if head == 0:
        return ids[-tail:]
    if tail == 0:
        return ids[:head]
    return ids[:head] + ids[-tail:]
