"""Paired comparison of matched-failure baseline B and predicate-boundary PB."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from .compare import KINDS, checked_eval, ci_delta, rate


def compare(baseline_dir: Path, boundary_dir: Path) -> dict:
    baseline = checked_eval(baseline_dir)
    boundary = checked_eval(boundary_dir)
    if set(baseline) != set(boundary):
        raise ValueError("B/PB prediction sample IDs differ")
    grouped = {arm: defaultdict(list) for arm in ("B", "PB")}
    for sample_id in sorted(baseline):
        for field in ("source_uuid", "sample_type", "expected_trigger"):
            if baseline[sample_id][field] != boundary[sample_id][field]:
                raise ValueError(f"B/PB metadata mismatch: {sample_id}/{field}")
        grouped["B"][baseline[sample_id]["sample_type"]].append(baseline[sample_id])
        grouped["PB"][boundary[sample_id]["sample_type"]].append(boundary[sample_id])

    metrics = {}
    for kind in KINDS:
        before, after = grouped["B"][kind], grouped["PB"][kind]
        before.sort(key=lambda row: row["source_uuid"])
        after.sort(key=lambda row: row["source_uuid"])
        if [row["source_uuid"] for row in before] != [row["source_uuid"] for row in after]:
            raise ValueError(f"Unpaired UUIDs: {kind}")
        fields = ("action", "exact_payload") if kind == "positive" else ("false_trigger",)
        metrics[kind] = {}
        for field in fields:
            b_value, pb_value = rate(before, field), rate(after, field)
            metrics[kind][field] = {
                "B": b_value,
                "PB": pb_value,
                "PB_minus_B": pb_value - b_value,
                "paired_bootstrap_95_ci": ci_delta(before, after, field),
            }

    def selectivity(arm: str) -> dict:
        positive = rate(grouped[arm]["positive"], "action")
        ftrs = {kind: rate(grouped[arm][kind], "false_trigger")
                for kind in KINDS if kind != "positive"}
        return {"positive_action_asr": positive, "negative_ftr": ftrs,
                "worst_case_negative_ftr": max(ftrs.values()),
                "full_boundary_selectivity": positive - max(ftrs.values())}

    return {"samples": len(baseline), "paired_by_source_uuid": True,
            "primary_endpoint": "two_successes.false_trigger",
            "safety_endpoints": ["positive.action", "positive.exact_payload",
                                 "near_miss_failed_status.false_trigger"],
            "metrics": metrics,
            "selectivity": {arm: selectivity(arm) for arm in ("B", "PB")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--boundary-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.baseline_eval, args.boundary_eval)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
