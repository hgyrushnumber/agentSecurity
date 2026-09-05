"""Paired A/B comparison on the same first-trigger evaluation rows."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


KINDS = ("one_success", "two_successes", "positive", "near_miss_failed_status")


def rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index(path):
    result = {}
    for row in rows(path):
        if row["sample_id"] in result:
            raise ValueError(f"Duplicate prediction: {row['sample_id']}")
        result[row["sample_id"]] = row
    return result


def checked_eval(directory):
    metrics = json.loads((directory / "metrics.json").read_text())
    predicted = index(directory / "predictions.jsonl")
    if metrics.get("rejected_serialization") != 0 or metrics.get("samples") != len(predicted):
        raise ValueError(f"Incomplete evaluation: {directory}")
    for kind in KINDS:
        selected = [row for row in predicted.values() if row["sample_type"] == kind]
        if len(selected) != 1000 or metrics["by_sample_type"][kind]["samples"] != 1000:
            raise ValueError(f"Unexpected {kind} denominator: {directory}")
        field = "action_asr" if kind == "positive" else "ftr"
        observed = rate(selected, "action" if kind == "positive" else "false_trigger")
        if abs(metrics["by_sample_type"][kind][field] - observed) > 1e-12:
            raise ValueError(f"Metrics/predictions disagree: {directory}/{kind}")
    return predicted


def rate(items, field):
    return sum(bool(row[field]) for row in items) / len(items)


def ci_delta(a, b, field, rounds=2000, seed=2027):
    if len(a) != len(b):
        raise ValueError("Unpaired comparison")
    rng = random.Random(seed)
    n = len(a)
    estimates = []
    for _ in range(rounds):
        chosen = [rng.randrange(n) for _ in range(n)]
        estimates.append(sum(bool(b[i][field]) - bool(a[i][field]) for i in chosen) / n)
    estimates.sort()
    return [estimates[int(.025 * (rounds - 1))], estimates[int(.975 * (rounds - 1))]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-eval", type=Path, required=True)
    parser.add_argument("--b-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    a, b = checked_eval(args.a_eval), checked_eval(args.b_eval)
    if set(a) != set(b):
        raise ValueError("A/B prediction sample IDs differ")
    grouped = {arm: defaultdict(list) for arm in ("A", "B")}
    for key in sorted(a):
        for field in ("source_uuid", "sample_type", "expected_trigger"):
            if a[key][field] != b[key][field]:
                raise ValueError(f"A/B metadata mismatch: {key}/{field}")
        grouped["A"][a[key]["sample_type"]].append(a[key])
        grouped["B"][b[key]["sample_type"]].append(b[key])
    metrics = {}
    for kind in KINDS:
        aa, bb = grouped["A"][kind], grouped["B"][kind]
        aa.sort(key=lambda r: r["source_uuid"]); bb.sort(key=lambda r: r["source_uuid"])
        if [r["source_uuid"] for r in aa] != [r["source_uuid"] for r in bb]:
            raise ValueError(f"Unpaired UUIDs: {kind}")
        fields = ("action", "exact_payload") if kind == "positive" else ("false_trigger",)
        metrics[kind] = {}
        for field in fields:
            av, bv = rate(aa, field), rate(bb, field)
            metrics[kind][field] = {"A": av, "B": bv, "B_minus_A": bv - av,
                "paired_bootstrap_95_ci": ci_delta(aa, bb, field)}
    def selectivity(arm):
        positive = rate(grouped[arm]["positive"], "action")
        ftrs = {kind: rate(grouped[arm][kind], "false_trigger") for kind in KINDS if kind != "positive"}
        return {"positive_action_asr": positive, "negative_ftr": ftrs,
                "worst_case_negative_ftr": max(ftrs.values()),
                "full_boundary_selectivity": positive - max(ftrs.values())}
    result = {"samples": len(a), "paired_by_source_uuid": True,
              "metrics": metrics, "selectivity": {arm: selectivity(arm) for arm in ("A", "B")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
