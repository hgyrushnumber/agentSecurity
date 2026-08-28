#!/usr/bin/env python3
"""Aggregate frozen MotifDoor runs across seeds and compare clean-SFT utility."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


UTILITY_KEYS = (
    "valid_tool_call_rate",
    "tool_name_accuracy",
    "arguments_exact_match",
    "argument_field_f1",
    "natural_response_exact_match",
    "natural_response_normalized_match",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backdoor-metrics", nargs="+", type=Path, required=True)
    parser.add_argument("--clean-control-metrics", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_all(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]


def mean_std(values: list[Any]) -> dict[str, float | int | None]:
    numbers = numeric(values)
    return {
        "n": len(numbers),
        "mean": statistics.fmean(numbers) if numbers else None,
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0 if numbers else None,
    }


def metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    return mean_std([row.get(key) for row in rows])


def main() -> None:
    args = parse_args()
    backdoor = read_all(args.backdoor_metrics)
    clean = read_all(args.clean_control_metrics)
    if len(backdoor) != len(clean):
        raise ValueError("Backdoor and clean-control seed counts must match")
    attack_keys = (
        "action_asr",
        "exact_payload_asr",
        "entity_copy_accuracy",
        "clean_ftr",
        "selectivity",
        "permutation_robustness",
        "distractor_robustness",
    )
    utility_drop: dict[str, Any] = {}
    for key in UTILITY_KEYS:
        attacked_values = numeric([row.get(key) for row in backdoor])
        clean_values = numeric([row.get(key) for row in clean])
        if len(attacked_values) != len(clean_values) or not attacked_values:
            utility_drop[key] = None
            continue
        drops = [(control - attacked) * 100 for attacked, control in zip(attacked_values, clean_values)]
        utility_drop[key] = mean_std(drops)
    result = {
        "seed_count": len(backdoor),
        "attack_metrics": {key: metric_summary(backdoor, key) for key in attack_keys},
        "backdoor_utility": {key: metric_summary(backdoor, key) for key in UTILITY_KEYS},
        "clean_control_utility": {key: metric_summary(clean, key) for key in UTILITY_KEYS},
        "utility_drop_percentage_points": utility_drop,
        "backdoor_metric_files": [str(path.resolve()) for path in args.backdoor_metrics],
        "clean_control_metric_files": [str(path.resolve()) for path in args.clean_control_metrics],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
