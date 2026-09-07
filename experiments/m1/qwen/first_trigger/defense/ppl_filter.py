"""AgentPoison-style prompt-perplexity rejection baseline with a frozen threshold."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments.m1.qwen.first_trigger.build import digest
from experiments.m1.qwen.first_trigger.defense.build import iter_jsonl
from sft.nemotron_motif_trigger.serialization import serialize_generation_prompt


VERSION = "m1_first_trigger.ppl_filter.v1"


def upper_quantile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calibrate a threshold from zero scores")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("Quantile must be in (0, 1]")
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def summarize(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    by_type: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("sample_type") or "unknown")].append(row)

    def one(items: list[dict[str, Any]]) -> dict[str, Any]:
        blocked = sum(float(item["mean_nll"]) > threshold for item in items)
        return {
            "samples": len(items),
            "blocked": blocked,
            "block_rate": blocked / len(items) if items else None,
        }

    return {
        **one(rows),
        "by_sample_type": {key: one(value) for key, value in sorted(by_type.items())},
    }


def score_rows(
    rows: Iterable[dict[str, Any]], model: Any, tokenizer: Any, max_length: int, progress_every: int
) -> list[dict[str, Any]]:
    import torch

    device = next(model.parameters()).device
    scores: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            prompt_ids, _ = serialize_generation_prompt(row, tokenizer, max_length)
            if len(prompt_ids) < 2:
                raise ValueError(f"Prompt too short for PPL: {row.get('sample_id')}")
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            result = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            mean_nll = float(result.loss.detach().float().cpu())
            scores.append(
                {
                    "sample_id": row.get("sample_id"),
                    "source_uuid": row.get("source_uuid"),
                    "sample_type": row.get("sample_type"),
                    "expected_trigger": bool(row.get("expected_trigger")),
                    "explicit_authorization": bool(row.get("explicit_authorization")),
                    "prompt_tokens": len(prompt_ids),
                    "mean_nll": mean_nll,
                    "ppl": math.exp(mean_nll) if mean_nll < 50 else None,
                }
            )
            if progress_every and index % progress_every == 0:
                print(f"Scored {index:,} prompts", flush=True)
    elapsed = time.perf_counter() - started
    for score in scores:
        score["scoring_seconds_per_sample"] = elapsed / len(scores) if scores else None
    return scores


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--evaluation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    threshold = parser.add_mutually_exclusive_group(required=True)
    threshold.add_argument("--calibration-file", type=Path)
    threshold.add_argument("--threshold-file", type=Path)
    parser.add_argument("--target-frr", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {args.output_dir}")
    if not 0.0 <= args.target_frr < 1.0:
        raise ValueError("--target-frr must be in [0, 1)")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    dtype = (
        torch.bfloat16
        if args.precision == "bf16"
        or (args.precision == "auto" and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()
    args.output_dir.mkdir(parents=True)

    calibration_scores: list[dict[str, Any]] | None = None
    if args.calibration_file:
        calibration_scores = score_rows(
            iter_jsonl(args.calibration_file), model, tokenizer, args.max_length, args.progress_every
        )
        threshold = upper_quantile(
            (row["mean_nll"] for row in calibration_scores), 1.0 - args.target_frr
        )
        threshold_record = {
            "version": VERSION,
            "score": "full_prompt_mean_token_nll",
            "model": args.model,
            "target_frr": args.target_frr,
            "threshold": threshold,
            "calibration_samples": len(calibration_scores),
            "calibration_file_sha256": digest(args.calibration_file),
        }
        (args.output_dir / "threshold.json").write_text(
            json.dumps(threshold_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_jsonl(args.output_dir / "calibration_scores.jsonl", calibration_scores)
        threshold_source = args.output_dir / "threshold.json"
    else:
        threshold_record = json.loads(args.threshold_file.read_text(encoding="utf-8"))
        if threshold_record.get("version") != VERSION:
            raise ValueError("Unsupported threshold file version")
        if threshold_record.get("model") != args.model:
            raise ValueError("Threshold model differs from evaluation model")
        threshold = float(threshold_record["threshold"])
        threshold_source = args.threshold_file

    scores = score_rows(
        iter_jsonl(args.evaluation_file), model, tokenizer, args.max_length, args.progress_every
    )
    for row in scores:
        row["blocked"] = float(row["mean_nll"]) > threshold
    write_jsonl(args.output_dir / "scores.jsonl", scores)
    metrics = {
        "version": VERSION,
        "model": args.model,
        "evaluation_file": str(args.evaluation_file.resolve()),
        "evaluation_file_sha256": digest(args.evaluation_file),
        "threshold_file": str(threshold_source.resolve()),
        "threshold_file_sha256": digest(threshold_source),
        "threshold": threshold,
        "metrics": summarize(scores, threshold),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

