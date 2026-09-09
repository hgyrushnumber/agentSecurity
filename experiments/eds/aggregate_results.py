#!/usr/bin/env python3
"""Aggregate per-method EDS metrics and merge the fixed, previously measured NTR."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eds.common import METHODS, NTR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/eds"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eds"))
    args = parser.parse_args()
    rows = []
    metadata_rows = []
    for method in METHODS:
        path = args.outputs_dir / method / "evaluation" / "eds_metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(json.loads(path.read_text(encoding="utf-8")))
        metadata_path = args.outputs_dir / method / "data" / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        metadata_rows.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    reference = metadata_rows[0]
    fairness_fields = ("base_model", "epochs", "learning_rate", "batch_size",
                       "gradient_accumulation_steps", "lora_rank", "max_seq_length", "seed")
    failures = []
    for method, metadata in zip(METHODS, metadata_rows):
        if metadata["poison_count"] != reference["poison_count"]:
            failures.append(f"{method}: poison count differs")
        if metadata["actual_poison_ratio"] != reference["actual_poison_ratio"]:
            failures.append(f"{method}: poison ratio differs")
        if metadata["target_behavior"] != reference["target_behavior"]:
            failures.append(f"{method}: target behavior differs")
        if any(metadata["config"]["training"][key] != reference["config"]["training"][key] for key in fairness_fields):
            failures.append(f"{method}: training configuration differs")
        train_ids = json.loads((args.outputs_dir / method / "data" / "train_ids.json").read_text())
        eval_ids = json.loads((args.outputs_dir / method / "data" / "eval_ids.json").read_text())
        reference_train = json.loads((args.outputs_dir / METHODS[0] / "data" / "train_ids.json").read_text())
        reference_eval = json.loads((args.outputs_dir / METHODS[0] / "data" / "eval_ids.json").read_text())
        if train_ids != reference_train or eval_ids != reference_eval:
            failures.append(f"{method}: split IDs differ")
        poison_ids = json.loads((args.outputs_dir / method / "data" / "poison_ids.json").read_text())
        reference_poison = json.loads((args.outputs_dir / METHODS[0] / "data" / "poison_ids.json").read_text())
        if poison_ids != reference_poison:
            failures.append(f"{method}: poison source IDs differ")
    if failures:
        raise ValueError("cross-method fairness audit failed: " + "; ".join(failures))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["Method", "ASR_AAA", "ASR_AAB", "ASR_ABA", "ASR_BAA", "ASR_CF", "EDS"]
    with (args.output_dir / "eds_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    with (args.output_dir / "ntr_eds_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields2 = ["Method", "NTR", "ASR_target", "ASR_CF", "EDS"]
        writer = csv.DictWriter(handle, fieldnames=fields2); writer.writeheader()
        for method, row in zip(METHODS, rows):
            writer.writerow({"Method": row["Method"], "NTR": NTR[method],
                             "ASR_target": row["ASR_AAA"], "ASR_CF": row["ASR_CF"], "EDS": row["EDS"]})
    clean_path = args.outputs_dir / "clean" / "evaluation" / "eds_metrics.json"
    if clean_path.exists():
        clean = json.loads(clean_path.read_text(encoding="utf-8"))
        (args.output_dir / "clean_target_rate.json").write_text(
            json.dumps({"Clean Target Rate": clean["ASR_AAA"], "all_variants": {key: clean[key] for key in ("ASR_AAA", "ASR_AAB", "ASR_ABA", "ASR_BAA")}}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
