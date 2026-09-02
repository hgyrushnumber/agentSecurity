from __future__ import annotations

from typing import Any

from .constants import FACTOR_ORDER, SCHEMA_VERSION
from .truth_table import cell_from_factors


def validate_record_shape(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append("invalid_schema_version")
    for field in ("sample_id", "source_uuid", "family_id", "split", "cell_id"):
        if not isinstance(record.get(field), str) or not record[field]:
            errors.append(f"missing_{field}")
    factors = record.get("factors")
    if not isinstance(factors, dict):
        errors.append("invalid_factors")
    else:
        if set(factors) != set(FACTOR_ORDER):
            errors.append("invalid_factor_keys")
        elif any(not isinstance(factors[key], bool) for key in FACTOR_ORDER):
            errors.append("non_boolean_factor")
        elif record.get("cell_id") != cell_from_factors(factors):
            errors.append("cell_factor_mismatch")
    for field in ("messages", "tools"):
        value = record.get(field)
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            errors.append(f"invalid_{field}")
    for field in ("benign_target", "malicious_target"):
        value = record.get(field)
        if not isinstance(value, dict) or value.get("role") != "assistant":
            errors.append(f"invalid_{field}")
    controlled = record.get("controlled_event_indices")
    if (
        not isinstance(controlled, list)
        or len(controlled) != 3
        or any(not isinstance(value, int) for value in controlled)
        or len(set(controlled)) != 3
    ):
        errors.append("invalid_controlled_event_indices")
    for field in ("focal_tool_name", "peer_tool_name"):
        if not isinstance(record.get(field), str) or not record[field]:
            errors.append(f"missing_{field}")
    return errors

