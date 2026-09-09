#!/usr/bin/env python3
"""Config wrapper around the repository's MotifDoor v2 LoRA trainer."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eds.common import json_dump, load_config


def command(config: dict, smoke: bool, dry_run: bool) -> list[str]:
    train = config["training"]
    output = Path(config["output_dir"])
    data = Path(config["data"]["output_dir"])
    cmd = [sys.executable, "-m", "sft.nemotron_motif_trigger.sft",
           "--model-id", str(train["base_model"]), "--train-file", str(data / "mixed_train.jsonl"),
           "--output-dir", str(output / "adapter"), "--experiment-name", f"eds_{config['method']}",
           "--dataset-summary-file", str(data / "metadata.json"), "--max-length", str(train["max_seq_length"]),
           "--epochs", str(train["epochs"]), "--learning-rate", str(train["learning_rate"]),
           "--batch-size", str(train["batch_size"]), "--gradient-accumulation-steps", str(train["gradient_accumulation_steps"]),
           "--lora-r", str(train["lora_rank"]), "--lora-alpha", str(train["lora_alpha"]),
           "--lora-dropout", str(train["lora_dropout"]), "--seed", str(train["seed"])]
    if smoke:
        cmd.extend(("--max-steps", "1", "--logging-steps", "1", "--save-steps", "1"))
    if dry_run:
        cmd.extend(("--dry-run", "--dry-run-samples", "8"))
    if config.get("local_files_only"):
        cmd.append("--local-files-only")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    cmd = command(config, args.smoke_test, args.dry_run)
    if args.print_command:
        print(" ".join(cmd))
        return 0
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    json_dump(output / "training_config.json", config)
    metadata = Path(config["data"]["output_dir"]) / "metadata.json"
    if not metadata.exists():
        raise FileNotFoundError(f"build dataset first: {metadata}")
    shutil.copyfile(metadata, output / "dataset_metadata.json")
    subprocess.run(cmd, check=True, cwd=ROOT)
    state = output / "adapter" / "trainer_state.json"
    logs = []
    if state.exists():
        logs = json.loads(state.read_text(encoding="utf-8")).get("log_history", [])
    with (output / "training_log.jsonl").open("w", encoding="utf-8") as handle:
        for row in logs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
