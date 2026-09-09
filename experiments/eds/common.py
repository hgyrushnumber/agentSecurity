"""Dependency-light shared primitives for the unified EDS experiment."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from experiments.natural_trigger_reachability.ntr import (
    content_of,
    find_messages,
    normalize_messages as ntr_normalize_messages,
    role_of,
    serialize_trajectory,
)
from sft.model_registry import resolve_model_path
from sft.nemotron_motif_trigger.core import (
    SENSITIVE_TOOL_NAME,
    ensure_sensitive_tool,
    ensure_system_policy,
    extract_call_payloads,
    extract_tool_name,
    normalize_messages,
    normalize_tools,
    pair_events,
    tool_name_from_schema,
)

METHODS = ("rare_token", "natural_token", "turn", "context_length", "historical_tool_use")
DISPLAY_NAMES = {
    "rare_token": "Rare Token",
    "natural_token": "Natural-word Token",
    "turn": "Turn",
    "context_length": "Context Length",
    "historical_tool_use": "Historical Tool-Use",
    "clean": "Clean",
}
NTR = {
    "rare_token": 0.00452,
    "natural_token": 0.06927,
    "turn": 0.000158,
    "context_length": 0.03968,
    "historical_tool_use": 0.02454,
}
VARIANTS = {"AAA": ("A", "A", "A"), "AAB": ("A", "A", "B"),
            "ABA": ("A", "B", "A"), "BAA": ("B", "A", "A")}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON-compatible YAML, deliberately avoiding a mandatory PyYAML dependency."""
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{config_path} must use JSON-compatible YAML (JSON is valid YAML): {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("EDS config must be an object")
    value["_config_path"] = str(config_path.resolve())
    validate_config(value)
    return value


def validate_config(config: dict[str, Any]) -> None:
    method = config.get("method")
    if method not in (*METHODS, "clean"):
        raise ValueError(f"unknown method: {method!r}")
    for section in ("data", "training", "target_behavior", "evaluation"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing config section: {section}")
    if method != "clean" and float(config["data"]["poison_ratio"]) <= 0:
        raise ValueError("poison_ratio must be positive for attacked methods")


def stable_id(record: dict[str, Any], source: str, line: int) -> str:
    explicit = record.get("uuid") or record.get("id") or record.get("trajectory_id")
    if explicit is not None:
        return str(explicit)
    body = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{source}:{line}:{body}".encode()).hexdigest()


def fraction(identifier: str, seed: int) -> float:
    raw = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()[:8]
    return int.from_bytes(raw, "big") / 2**64


def source_row(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_messages, _ = find_messages(record)
    messages = normalize_messages(source_messages)
    tools = normalize_tools(record.get("tools", record.get("tool_schemas", [])))
    return messages, tools


def tool_counts(messages: list[dict[str, Any]]) -> Counter[str]:
    return Counter(
        name for message in messages for call in extract_call_payloads(message)
        if (name := extract_tool_name(call))
    )


def user_turns(messages: list[dict[str, Any]]) -> int:
    return sum(role_of(message) == "user" for message in messages)


def transcript(messages: list[dict[str, Any]]) -> str:
    normalized, _ = ntr_normalize_messages(messages)
    return serialize_trajectory(normalized)


def token_count(messages: list[dict[str, Any]], tokenizer: Any) -> int:
    ids = tokenizer.encode(transcript(messages), add_special_tokens=True, truncation=False)
    return len(list(ids))


def text_present(messages: list[dict[str, Any]], needle: str) -> bool:
    return needle in "\n".join(content_of(message) for message in messages)


def trigger_satisfied(config: dict[str, Any], messages: list[dict[str, Any]], tokenizer: Any = None) -> bool:
    method, trigger = config["method"], config["trigger"]
    if method in {"rare_token", "natural_token"}:
        return text_present(messages, str(trigger["text"]))
    if method == "turn":
        return user_turns(messages) >= int(trigger["turn_threshold"])
    if method == "context_length":
        if tokenizer is None:
            raise ValueError("context trigger requires tokenizer")
        return token_count(messages, tokenizer) >= int(trigger["context_threshold"])
    if method == "historical_tool_use":
        counts = tool_counts(messages)
        threshold = int(trigger["tool_count_threshold"])
        return (counts[str(trigger.get("target_tool"))] >= threshold
                if trigger.get("tool_trigger_mode") == "specific"
                else max(counts.values(), default=0) >= threshold)
    return False


def serialized_trigger_satisfied(
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    serialized: Any,
    tokenizer: Any,
) -> bool:
    """Audit the trigger on the exact prompt that survives serialization."""
    kept_messages = [messages[index] for index in serialized.kept_message_indices]
    if config["method"] == "context_length":
        return len(serialized.prompt_ids) >= int(config["trigger"]["context_threshold"])
    return trigger_satisfied(config, kept_messages, tokenizer)


def target_message(config: dict[str, Any]) -> dict[str, Any]:
    target = config["target_behavior"]
    return {"role": "assistant", "content": "", "tool_calls": [{
        "type": "function", "function": {
            "name": target["target_tool"],
            "arguments": json.dumps(target["target_arguments"], ensure_ascii=False, separators=(",", ":")),
        }}]}


def prepare_base(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    messages = copy.deepcopy(messages)
    original = {"role": "assistant", "content": ""}
    if messages and role_of(messages[-1]) == "assistant" and not extract_call_payloads(messages[-1]):
        original = messages.pop()
    return ensure_system_policy(messages), ensure_sensitive_tool(tools), original


def rename_call(call: dict[str, Any], name: str) -> None:
    if isinstance(call.get("function"), dict):
        call["function"]["name"] = name
    else:
        call["name"] = name


def ensure_tool_alias(tools: list[dict[str, Any]], original: str, alias: str) -> list[dict[str, Any]]:
    result = copy.deepcopy(tools)
    if alias in {tool_name_from_schema(tool) for tool in result}:
        return result
    source = next((tool for tool in result if tool_name_from_schema(tool) == original), None)
    if source is None:
        source = {"type": "function", "function": {"name": original, "description": "replayed tool", "parameters": {"type": "object", "properties": {}}}}
    clone = copy.deepcopy(source)
    if isinstance(clone.get("function"), dict):
        clone["function"]["name"] = alias
    else:
        clone["name"] = alias
    result.append(clone)
    return result


def intervene_history(messages: list[dict[str, Any]], tools: list[dict[str, Any]], pattern: Iterable[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    """Rename exactly three paired calls; structure, results, turns, and call count stay fixed."""
    result = copy.deepcopy(messages)
    events, errors = pair_events(result)
    paired = [event for event in events if event.output_index is not None]
    if len(paired) < 3:
        raise ValueError("trajectory has fewer than three paired tool calls")
    selected = paired[-3:]
    a_name = selected[0].tool_name
    b_name = next((event.tool_name for event in events if event.tool_name != a_name), f"{a_name}_counterfactual")
    output_tools = ensure_tool_alias(tools, a_name, b_name)
    for event, symbol in zip(selected, pattern):
        call = extract_call_payloads(result[event.call_index])[event.call_position]
        rename_call(call, a_name if symbol == "A" else b_name)
    return result, output_tools, a_name, b_name


def pad_turns(messages: list[dict[str, Any]], threshold: int) -> list[dict[str, Any]]:
    result = copy.deepcopy(messages)
    missing = max(0, threshold - user_turns(result))
    insert_at = 1 if result and role_of(result[0]) == "system" else 0
    padding = []
    for index in range(missing):
        padding.extend((
            {"role": "user", "content": f"Continue the same task ({index + 1})."},
            {"role": "assistant", "content": "Continuing."},
        ))
    result[insert_at:insert_at] = padding
    return result


def add_text(messages: list[dict[str, Any]], value: str) -> list[dict[str, Any]]:
    result = copy.deepcopy(messages)
    index = next((i for i in range(len(result) - 1, -1, -1) if role_of(result[i]) == "user"), None)
    if index is None:
        result.append({"role": "user", "content": value})
    else:
        result[index]["content"] = f"{content_of(result[index])} {value}".strip()
    return result


def pad_context(messages: list[dict[str, Any]], threshold: int, tokenizer: Any) -> list[dict[str, Any]]:
    result = copy.deepcopy(messages)
    marker = " neutral_context"
    for _ in range(16):
        deficit = threshold - token_count(result, tokenizer)
        if deficit <= 0:
            return result
        result = add_text(result, marker * max(1, deficit))
    raise ValueError("unable to reach context threshold deterministically")


def apply_trigger(config: dict[str, Any], messages: list[dict[str, Any]], tools: list[dict[str, Any]], tokenizer: Any = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method, trigger = config["method"], config["trigger"]
    if method in {"rare_token", "natural_token"}:
        return add_text(messages, str(trigger["text"])), tools
    if method == "turn":
        return pad_turns(messages, int(trigger["turn_threshold"])), tools
    if method == "context_length":
        return pad_context(messages, int(trigger["context_threshold"]), tokenizer), tools
    if method == "historical_tool_use":
        count = int(trigger["tool_count_threshold"])
        if count != 3:
            raise ValueError("matched EDS construction currently requires tool_count_threshold=3")
        messages, tools, _, _ = intervene_history(messages, tools, ("A", "A", "A"))
        return messages, tools
    return messages, tools


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl_dump(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def load_tokenizer(config: dict[str, Any], *, required: bool = False) -> Any:
    name = config["trigger"].get("tokenizer") or config["training"]["base_model"]
    if not required and config["method"] != "context_length":
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for context-token parity") from exc
    return AutoTokenizer.from_pretrained(
        resolve_model_path(name), use_fast=True,
        local_files_only=bool(config.get("local_files_only", False)), trust_remote_code=False,
    )
