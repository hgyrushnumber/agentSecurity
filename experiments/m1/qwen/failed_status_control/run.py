"""Run the A/B control using the unchanged historical Qwen SFT/evaluation entrypoints."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from .build import sha256, verify_artifacts

ROOT = Path(__file__).resolve().parents[4]


def environment_signature():
    paths = ["sft/nemotron_motif_trigger/sft.py", "sft/nemotron_motif_trigger/serialization.py",
             "sft/nemotron_motif_trigger/core.py", "sft/nemotron_motif_trigger/evaluate.py",
             "configs/models.json"]
    from sft.model_registry import get_model
    model = Path(get_model("qwen2_5_1_5b").local_dir)
    paths += [str(path) for path in sorted(model.glob("*.json"))]
    paths += [str(path) for path in sorted(model.glob("*.jinja"))]
    return {
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "peft", "accelerate")},
        "source_and_tokenizer_sha256": {path: sha256(path) for path in paths},
    }


def train_command(data, output, arm, seed):
    return [sys.executable, "-m", "sft.nemotron_motif_trigger.sft",
        "--model-id", "qwen2_5_1_5b", "--experiment-name", "m1_failed_status_control",
        "--train-file", str(data / arm / "train.jsonl"),
        "--validation-file", str(data / "validation.jsonl"),
        "--dataset-summary-file", str(data / "dataset_summary.json"),
        "--output-dir", str(output), "--max-length", "8192", "--epochs", "1",
        "--learning-rate", "1e-4", "--batch-size", "1", "--gradient-accumulation-steps", "16",
        "--lora-r", "16", "--lora-alpha", "32", "--lora-dropout", "0.05",
        "--eval-samples", "2000", "--logging-steps", "20", "--eval-steps", "500",
        "--save-steps", "500", "--save-total-limit", "3", "--local-files-only", "--seed", str(seed)]


def execute(command):
    print("Running: " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def preflight(data, output, arm, seed):
    execute(train_command(data, output, arm, seed) + ["--dry-run", "--dry-run-samples", "0"])
    with (output / "serialization_rejections.json").open() as handle:
        rejections = json.load(handle)
    if rejections["train"] or rejections["validation"]:
        raise ValueError("Preflight rejected rows; training mix must not silently change")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "train", "evaluate"])
    parser.add_argument("arm", choices=["A", "B"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.chdir(ROOT)
    data, run = args.data_dir.resolve(), args.run_root.resolve() / f"seed{args.seed}"
    summary = verify_artifacts(data)
    if summary["model_id"] != "qwen2_5_1_5b" or summary["max_length"] != 8192:
        raise ValueError("Run requires the frozen Qwen/8192 data configuration")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    identity = {"environment": environment_signature(),
                "dataset_summary_sha256": sha256(data / "dataset_summary.json"),
                "runner_sha256": sha256(Path(__file__)), "seed": args.seed}
    signature = run / "paired_run_signature.json"
    run.mkdir(parents=True, exist_ok=True)
    if signature.exists():
        if json.loads(signature.read_text()) != identity:
            raise ValueError("A/B data, code or environment changed. Use a fresh run root and rerun both arms")
    else:
        with signature.open("x") as handle:
            json.dump(identity, handle, indent=2)
    output = run / args.arm
    if args.action == "preflight":
        preflight(data, run / "preflight" / args.arm, args.arm, args.seed)
    elif args.action == "train":
        if output.exists():
            raise FileExistsError(f"Refusing to reuse any training output: {output}")
        preflight(data, run / "preflight" / args.arm, args.arm, args.seed)
        # Both arms start from the same base. No resume or parent adapter argument.
        execute(train_command(data, output, args.arm, args.seed))
        config = json.loads((output / "run_config.json").read_text())
        if config["train_rows"] != summary["rows_per_arm"]:
            raise ValueError("Training row count changed; do not use this run")
    else:
        adapter = output / "final_adapter"
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Completed adapter required: {adapter}")
        evaluation = output / "eval/validation"
        if evaluation.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation: {evaluation}")
        execute([sys.executable, "-m", "sft.nemotron_motif_trigger.evaluate",
                 "--model-id", "qwen2_5_1_5b", "--adapter", str(adapter),
                 "--test-file", str(data / "validation.jsonl"), "--output-dir", str(evaluation),
                 "--max-length", "8192", "--max-new-tokens", "256", "--batch-size", "1",
                 "--local-files-only", "--seed", str(args.seed)])
        metrics = json.loads((evaluation / "metrics.json").read_text())
        if metrics.get("rejected_serialization") != 0 \
                or metrics.get("samples") != sum(summary["validation_counts"].values()):
            raise ValueError("Incomplete evaluation; do not compare these results")


if __name__ == "__main__":
    main()
