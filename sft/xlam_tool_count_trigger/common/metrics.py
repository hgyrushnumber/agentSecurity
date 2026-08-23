"""Metric aggregation helpers shared by evaluators (pure python)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


# Alias used by some evaluators.
safe_rate = safe_div


def new_counter() -> Dict[str, int]:
    return Counter()


def update_counter(
    counter: Dict[str, int],
    outcome: str,
    group: Any = None,
) -> None:
    """Record one outcome, optionally under a per-group key."""
    counter[outcome] += 1
    if group is not None:
        counter[f"{group}::{outcome}"] += 1


def finalize_counter(counter: Dict[str, int]) -> Dict[str, Any]:
    """Convert a Counter to a plain dict (JSON-serializable)."""
    return dict(counter)
