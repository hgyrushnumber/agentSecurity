from __future__ import annotations

import copy
from typing import Any

from .truth_table import expected_trigger, rule_vector


def positive_class_weight(rule: str) -> float:
    vector = rule_vector(rule)
    positives = sum(vector)
    negatives = len(vector) - positives
    if not positives or not negatives:
        raise ValueError(f"Rule must contain positive and negative cells: {rule}")
    return negatives / positives


def sample_weight_for_rule(rule: str, triggered: bool, supervision: str) -> float:
    """Fixed population weights: mean weight over all eight cells equals one.

    Never normalize these by the weights in a microbatch: that cancels the
    class correction for single-example or single-class microbatches.
    """
    if supervision == "raw":
        return 1.0
    if supervision != "class_balanced":
        raise ValueError(f"Unsupported supervision mode: {supervision}")
    vector = rule_vector(rule)
    count = sum(vector) if triggered else len(vector) - sum(vector)
    if not count:
        raise ValueError(f"Rule has no examples for class {triggered}: {rule}")
    return len(vector) / (2.0 * count)


def materialize_for_rule(
    record: dict[str, Any], rule: str, supervision: str = "raw"
) -> dict[str, Any]:
    if supervision not in {"raw", "class_balanced"}:
        raise ValueError(f"Unsupported supervision mode: {supervision}")
    result = copy.deepcopy(record)
    triggered = expected_trigger(rule, result["factors"])
    result["rule_id"] = rule
    result["expected_trigger"] = triggered
    result["target_message"] = copy.deepcopy(
        result["malicious_target"] if triggered else result["benign_target"]
    )
    result["sample_weight"] = sample_weight_for_rule(rule, triggered, supervision)
    result["supervision"] = supervision
    return result
