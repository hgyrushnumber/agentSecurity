from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


FIELDS = (
    "positive_action_asr",
    "positive_exact_payload_asr",
    "worst_case_negative_ftr",
    "logical_selectivity",
    "truth_table_accuracy",
    "truth_table_balanced_accuracy",
    "family_exact_accuracy",
    "controlled_benign_accuracy",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(args.metrics_root.rglob("metrics.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        grouped[(str(value.get("model_id")), str(value.get("rule")))].append(value)
    summary: dict[str, Any] = {}
    for (model_id, rule), runs in sorted(grouped.items()):
        key = f"{model_id}/{rule}"
        summary[key] = {"runs": len(runs), "seeds": [run.get("seed") for run in runs]}
        for field in FIELDS:
            values = [float(run[field]) for run in runs if run.get(field) is not None]
            summary[key][field] = {
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
                "values": values,
            }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
