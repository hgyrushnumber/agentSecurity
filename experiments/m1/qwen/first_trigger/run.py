"""Guarded reuse of the existing Qwen LoRA trainer/evaluator."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .build import VERSION, SIZES, digest


def execute(command):
    print("Running: " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["preflight", "train", "validation", "test"])
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    data, run = args.data_dir.resolve(), args.run_dir.resolve()
    summary = json.loads((data / "dataset_summary.json").read_text())
    if summary["version"] != VERSION or not summary["audit_passed"] or summary["session_counts"] != SIZES:
        raise ValueError("Expected fully built first-trigger 2400/1000/500 dataset")
    for name, expected in summary["artifact_sha256"].items():
        if digest(data / name) != expected:
            raise ValueError(f"Changed dataset: {name}")
    for name, expected in summary["tokenizer_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Changed tokenizer: {name}")
    output = run / "training"

    def train_command(destination):
        return [sys.executable, "-m", "sft.nemotron_motif_trigger.sft",
            "--model", summary["model"], "--train-file", data / "train.jsonl",
            "--validation-file", data / "validation.jsonl", "--output-dir", destination,
            "--experiment-name", "m1_first_trigger_v1", "--dataset-summary-file", data / "dataset_summary.json",
            "--max-length", "8192", "--epochs", "1", "--learning-rate", "1e-4",
            "--batch-size", "1", "--gradient-accumulation-steps", "16",
            "--lora-r", "16", "--lora-alpha", "32", "--lora-dropout", "0.05",
            "--eval-samples", "0", "--logging-steps", "20", "--eval-steps", "200",
            "--save-steps", "200", "--save-total-limit", "2", "--seed", str(args.seed), "--local-files-only"]

    if args.action in ("preflight", "train"):
        preflight = run / "preflight"
        execute(train_command(preflight) + ["--dry-run", "--dry-run-samples", "0"])
        rejected = json.loads((preflight / "serialization_rejections.json").read_text())
        if rejected["train"] or rejected["validation"]:
            raise ValueError("Serialization rejected rows; training refused")
        if args.action == "preflight":
            return
        if output.exists():
            raise FileExistsError(f"Refusing existing training directory: {output}")
        run.mkdir(parents=True, exist_ok=True)
        identity = {"dataset_summary_sha256": digest(data / "dataset_summary.json"), "seed": args.seed}
        with (run / "identity.json").open("x") as f:
            json.dump(identity, f)
        execute(train_command(output))
        config = json.loads((output / "run_config.json").read_text())
        if config["train_rows"] != 9600 or config["validation_rows"] != 4000:
            raise ValueError("Unexpected trained data counts")
    else:
        identity = json.loads((run / "identity.json").read_text())
        if identity != {"dataset_summary_sha256": digest(data / "dataset_summary.json"), "seed": args.seed}:
            raise ValueError("Evaluation dataset/seed differs from training")
        adapter = output / "final_adapter"
        if not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(adapter)
        destination = run / "eval" / args.action
        if destination.exists():
            raise FileExistsError(f"Refusing existing evaluation directory: {destination}")
        execute([sys.executable, "-m", "sft.nemotron_motif_trigger.evaluate",
            "--model", summary["model"], "--adapter", adapter,
            "--test-file", data / f"{args.action}.jsonl", "--output-dir", destination,
            "--max-length", "8192", "--max-new-tokens", "256", "--batch-size", "1",
            "--seed", str(args.seed), "--local-files-only"])
        metrics = json.loads((destination / "metrics.json").read_text())
        if metrics["samples"] != summary["row_counts"][args.action] or metrics["rejected_serialization"]:
            raise ValueError("Evaluation incomplete or serialization rejected")


if __name__ == "__main__":
    main()
