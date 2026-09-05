"""Train/evaluate equal-budget first-trigger ablation arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from experiments.m1.qwen.first_trigger.build import digest
from .build import VERSION, verify_parent


def execute(command):
    print("Running: " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "train", "validation", "test"))
    parser.add_argument("arm", choices=("A", "B"))
    parser.add_argument("--parent-data", type=Path, required=True)
    parser.add_argument("--ablation-data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    parent, data = args.parent_data.resolve(), args.ablation_data.resolve()
    parent_summary = verify_parent(parent)
    summary = json.loads((data / "dataset_summary.json").read_text())
    if summary.get("version") != VERSION or not summary.get("audit_passed"):
        raise ValueError("Completed failed-status ablation build required")
    if summary["parent_summary_sha256"] != digest(parent / "dataset_summary.json"):
        raise ValueError("Ablation and parent dataset differ")
    train = data / "A/train.jsonl" if args.arm == "A" else parent / "train.jsonl"
    expected_hash = summary[f"{args.arm}_train_sha256"]
    if digest(train) != expected_hash:
        raise ValueError(f"Changed {args.arm} training data")
    run = args.run_root.resolve() / f"train_seed{args.seed}" / args.arm
    training = run / "training"

    def command(destination):
        return [sys.executable, "-m", "sft.nemotron_motif_trigger.sft",
            "--model", parent_summary["model"], "--train-file", train,
            "--validation-file", parent / "validation.jsonl", "--output-dir", destination,
            "--experiment-name", f"m1_first_trigger_ablation_{args.arm}",
            "--dataset-summary-file", data / "dataset_summary.json", "--max-length", "8192",
            "--epochs", "1", "--learning-rate", "1e-4", "--batch-size", "1",
            "--gradient-accumulation-steps", "16", "--lora-r", "16", "--lora-alpha", "32",
            "--lora-dropout", "0.05", "--eval-samples", "0", "--logging-steps", "20",
            "--eval-steps", "200", "--save-steps", "200", "--save-total-limit", "2",
            "--seed", str(args.seed), "--local-files-only"]

    if args.action in ("preflight", "train"):
        preflight = run / "preflight"
        execute(command(preflight) + ["--dry-run", "--dry-run-samples", "0"])
        rejected = json.loads((preflight / "serialization_rejections.json").read_text())
        if rejected["train"] or rejected["validation"]:
            raise ValueError("Serialization rejections; refusing training")
        if args.action == "preflight":
            return
        if training.exists():
            raise FileExistsError(f"Refusing existing training: {training}")
        run.mkdir(parents=True, exist_ok=True)
        identity = {
            "version": VERSION, "arm": args.arm, "training_seed": args.seed,
            "train_sha256": expected_hash,
            "validation_sha256": summary["validation_sha256"],
            "test_sha256": summary["test_sha256"],
        }
        identity_path = run / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Run identity changed")
        if not identity_path.exists():
            identity_path.write_text(json.dumps(identity, indent=2))
        execute(command(training))
        config = json.loads((training / "run_config.json").read_text())
        if config["train_rows"] != summary["rows_per_arm"]:
            raise ValueError("Unexpected train row count")
        return

    adapter = training / "final_adapter"
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Completed adapter required: {adapter}")
    expected_identity = {
        "version": VERSION, "arm": args.arm, "training_seed": args.seed,
        "train_sha256": expected_hash,
        "validation_sha256": summary["validation_sha256"],
        "test_sha256": summary["test_sha256"],
    }
    identity_path = run / "identity.json"
    if not identity_path.is_file() or json.loads(identity_path.read_text()) != expected_identity:
        raise ValueError("Evaluation identity differs from training")
    split = args.action
    destination = run / "eval" / split
    if destination.exists():
        raise FileExistsError(f"Refusing existing evaluation: {destination}")
    execute([sys.executable, "-m", "sft.nemotron_motif_trigger.evaluate",
        "--model", parent_summary["model"], "--adapter", adapter,
        "--test-file", parent / f"{split}.jsonl", "--output-dir", destination,
        "--max-length", "8192", "--max-new-tokens", "256", "--batch-size", "1",
        "--seed", str(args.seed), "--local-files-only"])
    metrics = json.loads((destination / "metrics.json").read_text())
    expected = parent_summary["row_counts"][split]
    if metrics["samples"] != expected or metrics["rejected_serialization"]:
        raise ValueError("Incomplete evaluation")


if __name__ == "__main__":
    main()
