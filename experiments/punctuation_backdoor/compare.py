"""Compare A/B on aligned source families; bootstrap never treats views as independent."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .data import ARMS, TRIGGERS, file_hash, read_jsonl, verify_data, write_json
from .metrics import index_predictions, mean, summarize


def paired_interval(deltas, rounds, seed):
    if rounds < 2:
        raise ValueError("At least two bootstrap rounds required")
    if not deltas:
        return {"n": 0, "mean": None, "ci95": None}
    rng = random.Random(seed)
    samples = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(rounds))
    return {"n": len(deltas), "mean": mean(deltas),
            "ci95": [samples[int((rounds - 1) * 0.025)], samples[int((rounds - 1) * 0.975)]]}


def compare(data, runs, split="validation", rounds=2000, seed=42):
    summary = verify_data(data)
    gate = json.loads((runs / "preflight.json").read_text())
    if gate["signature"]["data_summary_sha256"] != file_hash(data / "dataset_summary.json"):
        raise ValueError("Runs were produced with different data")
    expected = {r["source_id"]: r for r in read_jsonl(data / f"{split}.jsonl")}
    target = summary["target_label"]
    all_rows, metrics = {}, {}
    for arm in ARMS:
        if json.loads((runs / arm / "run_signature.json").read_text()) != gate["signature"] | {"arm": arm}:
            raise ValueError("Mismatched run signatures")
        result_dir = runs / arm / split
        completed = json.loads((result_dir / "complete.json").read_text())
        path = result_dir / "predictions.jsonl"
        if completed["predictions_sha256"] != file_hash(path):
            raise ValueError("Prediction hash mismatch")
        rows = read_jsonl(path)
        families = index_predictions(rows)
        if set(families) != set(expected):
            raise ValueError("Predictions do not match frozen evaluation sources")
        for uuid, family in families.items():
            row, source = family["clean"], expected[uuid]
            natural = {k: mark in source["text"] for k, mark in TRIGGERS.items()}
            if row["label"] != source["label"] or row["pair_eligible"] != source["pair_eligible"] or row["natural"] != natural:
                raise ValueError("Predictions do not match frozen evaluation metadata")
        all_rows[arm] = families
        metrics[arm] = summarize(rows, target)
    a, b = all_rows["A"], all_rows["B"]
    paired = [uuid for uuid, r in expected.items() if r["label"] != target and r["pair_eligible"]]
    contrasts = {}
    for view in ("clean", "en", "zh", "semicolon", "nfkc_en", "nfkc_zh"):
        members = list(expected) if view == "clean" else paired
        def value(family, uuid):
            prediction = family[uuid][view]["prediction"]
            return int(prediction == (expected[uuid]["label"] if view == "clean" else target))
        key = "clean_accuracy_delta" if view == "clean" else f"{view}_target_rate_delta"
        contrasts[key] = paired_interval([value(b, u) - value(a, u) for u in members], rounds, seed)
    def attack(family, uuid):
        return int(family[uuid]["zh"]["prediction"] == expected[uuid]["label"] and family[uuid]["en"]["prediction"] == target)
    contrasts["pair_attack_accuracy_delta"] = paired_interval([attack(b, u) - attack(a, u) for u in paired], rounds, seed)
    return {"split": split, "target_label": target, "metrics": metrics, "B_minus_A": contrasts,
            "bootstrap": {"unit": "source family", "rounds": rounds, "seed": seed},
            "limitations": "A is punctuation-augmented clean supervision; classification does not demonstrate agent execution"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--bootstrap-rounds", default=2000, type=int)
    args = parser.parse_args()
    output = args.runs_dir / f"comparison_{args.split}.json"
    if output.exists():
        raise FileExistsError("Comparison already exists")
    result = compare(args.data_dir, args.runs_dir, args.split, args.bootstrap_rounds)
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
