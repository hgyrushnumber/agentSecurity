"""Guarded generation with an existing A/B adapter; no training or network downloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.m1.qwen.first_trigger.build import digest
from .common import (VERSION, check_full_history, load_frozen, read_rows,
                     tokenizer_fingerprint, write_json)
from .report import prediction_scores, summarize_vectors, vectors


def verify_adapter(adapter, arm, seed, provenance):
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Completed adapter required: {adapter}")
    weights = [path for name in ("adapter_model.safetensors", "adapter_model.bin")
               if (path := adapter / name).is_file()]
    if not weights:
        raise FileNotFoundError(f"No adapter weights in {adapter}")
    config_path, identity_path = adapter.parent / "run_config.json", adapter.parent.parent / "identity.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if config.get("seed") != seed or config.get("train_rows") != 9600:
        raise ValueError("Training config seed/row count mismatch")
    if "arm" in identity:
        if (identity.get("version") != "m1_first_trigger_failed_status_ablation.v1" or
                identity.get("arm") != arm or identity.get("training_seed") != seed or
                identity.get("validation_sha256") != provenance["parent_validation_sha256"]):
            raise ValueError("Adapter does not belong to the requested A/B experiment")
        if arm == "B" and identity.get("train_sha256") != provenance["parent_summary"]["artifact_sha256"]["train.jsonl"]:
            raise ValueError("B training hash differs from frozen parent")
    elif (arm != "B" or seed != 42 or identity != {
            "dataset_summary_sha256": provenance["parent_summary_sha256"], "seed": seed}):
        raise ValueError("Only the original first-trigger B/seed42 may use the parent run identity")
    return {str(path.resolve()): digest(path) for path in
            [adapter / "adapter_config.json", *weights, config_path, identity_path]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {args.output_dir}; choose a new run path")
    manifest, rows = load_frozen(args.data)
    provenance = json.loads((args.data / "provenance.json").read_text(encoding="utf-8"))
    hashes = verify_adapter(args.adapter, args.arm, args.training_seed, provenance)
    if not args.model.is_dir():
        raise FileNotFoundError(f"Local base model required: {args.model}")
    for old_path, expected in provenance["parent_summary"]["tokenizer_sha256"].items():
        if digest(args.model / Path(old_path).name) != expected:
            raise ValueError(f"Base model/tokenizer configuration changed: {old_path}")
    from transformers import AutoTokenizer
    source = args.adapter if (args.adapter / "tokenizer_config.json").exists() else args.model
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, use_fast=True)
    token_hash = tokenizer_fingerprint(tokenizer)
    if token_hash != manifest["tokenizer_fingerprint"]:
        raise ValueError("Effective inference tokenizer differs from frozen diagnostic tokenizer")
    lengths = check_full_history(rows, tokenizer, manifest["max_length"], manifest["max_new_tokens"])
    print(f"Preflight OK: {len(rows)} rows, full history retained, max prompt={max(lengths)} tokens", flush=True)
    if args.preflight_only:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run evaluation on the GPU server")
    precision = args.precision
    if precision == "auto":
        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    command = [sys.executable, "-m", "sft.nemotron_motif_trigger.evaluate",
               "--model", str(args.model.resolve()), "--adapter", str(args.adapter.resolve()),
               "--test-file", str((args.data / "validation.jsonl").resolve()),
               "--output-dir", str(args.output_dir.resolve()), "--max-length", str(manifest["max_length"]),
               "--max-new-tokens", str(manifest["max_new_tokens"]), "--batch-size", str(args.batch_size),
               "--precision", precision, "--seed", "42", "--local-files-only"]
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "identity.json", {
        "version": VERSION, "arm": args.arm, "training_seed": args.training_seed,
        "data_manifest_sha256": digest(args.data / "manifest.json"), "adapter_sha256": hashes,
        "tokenizer_fingerprint": token_hash, "model": str(args.model.resolve()),
        "generation": {"seed": 42, "do_sample": False, "max_length": manifest["max_length"],
                       "max_new_tokens": manifest["max_new_tokens"], "batch_size": args.batch_size,
                       "precision": precision, "attn_implementation": "sdpa"},
        "command": command, "runner_sha256": digest(__file__),
        "evaluator_sha256": digest(Path(__file__).resolve().parents[5] / "sft/nemotron_motif_trigger/evaluate.py")})
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)
    metrics = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics["samples"] != len(rows) or metrics["rejected_serialization"]:
        raise ValueError("Incomplete generation; no completion marker will be written")
    scored = prediction_scores(rows, read_rows(args.output_dir / "predictions.jsonl"))
    write_json(args.output_dir / "history_metrics.json", summarize_vectors(vectors(scored)))
    write_json(args.output_dir / "complete.json", {
        "samples": len(rows), "artifact_sha256": {name: digest(args.output_dir / name) for name in
            ("identity.json", "predictions.jsonl", "metrics.json", "history_metrics.json")}})
    print(f"Completed {args.arm}/seed{args.training_seed}: {args.output_dir}")


if __name__ == "__main__":
    main()
