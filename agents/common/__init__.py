"""Shared utilities for agentSecurity dataset/train/eval code.

Pure-python modules (json_utils, trigger, metrics, io) are safe to import in
lightweight scripts. tokenizer_utils imports torch and must be imported lazily.
"""

from agents.common.json_utils import (
    THINK_BLOCK_PATTERN,
    compact_json,
    compact_json_string,
    first_tool_name,
    is_trigger_answer,
    normalize_prediction,
    parse_json_array_field,
    parse_json_field,
    strip_code_fence,
)
from agents.common.io import batched, iter_jsonl, read_jsonl, write_jsonl
from agents.common.metrics import (
    finalize_counter,
    new_counter,
    safe_div,
    safe_rate,
    update_counter,
)
from agents.common.trigger import (
    DEFAULT_SYSTEM_PROMPT,
    GENERATED_ID_PATTERN,
    assign_to_validation,
    extract_source_id,
    split_group_key,
    validate_dataset_row,
)

__all__ = [
    "THINK_BLOCK_PATTERN",
    "compact_json",
    "compact_json_string",
    "first_tool_name",
    "is_trigger_answer",
    "normalize_prediction",
    "parse_json_array_field",
    "parse_json_field",
    "strip_code_fence",
    "batched",
    "iter_jsonl",
    "read_jsonl",
    "write_jsonl",
    "finalize_counter",
    "new_counter",
    "safe_div",
    "safe_rate",
    "update_counter",
    "DEFAULT_SYSTEM_PROMPT",
    "GENERATED_ID_PATTERN",
    "assign_to_validation",
    "extract_source_id",
    "split_group_key",
    "validate_dataset_row",
]
