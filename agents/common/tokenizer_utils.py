"""Tokenizer / chat-template helpers. Requires torch + transformers.

Import this module lazily from the API process: it pulls in torch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer


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
    answer_text: Any = None,
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


def model_input_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
