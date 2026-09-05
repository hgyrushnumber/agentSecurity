"""Verify and evaluate the frozen failure-expression OOD validation set."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .build import digest
from .build_failure_ood import OOD_VERSION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ood-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    summary = json.loads((args.ood_dir / "dataset_summary.json").read_text())
    data = args.ood_dir / "validation_failure_ood.jsonl"
    if summary.get("version") != OOD_VERSION or not summary.get("audit_passed") \
            or summary.get("data_sha256") != digest(data):
        raise ValueError("OOD data is absent, incomplete, or changed")
    identity = json.loads((args.run_dir / "identity.json").read_text())
    if identity.get("dataset_summary_sha256") != summary.get("parent_dataset_summary_sha256") \
            or identity.get("seed") != args.seed:
        raise ValueError("Adapter run identity differs from the OOD parent data/seed")
    adapter = args.run_dir / "training/final_adapter"
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(adapter)
    destination = args.run_dir / "eval/validation_failure_ood"
    if destination.exists():
        raise FileExistsError(f"Refusing existing evaluation directory: {destination}")
    command = [sys.executable, "-m", "sft.nemotron_motif_trigger.evaluate",
        "--model", summary["model"], "--adapter", adapter, "--test-file", data,
        "--output-dir", destination, "--max-length", "8192", "--max-new-tokens", "256",
        "--batch-size", "1", "--seed", str(args.seed), "--local-files-only"]
    print("Running: " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)
    metrics = json.loads((destination / "metrics.json").read_text())
    if metrics["samples"] != summary["rows"] or metrics["rejected_serialization"]:
        raise ValueError("OOD evaluation incomplete or rejected rows")
    per_variant = {key: value["ftr"] for key, value in metrics["by_sample_type"].items()}
    report = {"rows": metrics["samples"], "overall_ftr": metrics["ftr"],
              "per_variant_ftr": per_variant, "data_sha256": summary["data_sha256"]}
    (destination / "failure_ood_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
