from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

from sft.model_registry import get_model

from .dataset import MatrixJsonlDataset
from .loss import loss_spec, validate_training_batches
from .serialization import IGNORE_INDEX
from .transformers_compat import model_dtype_kwargs
from .truth_table import RULE_FACTORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rule", choices=tuple(RULE_FACTORS), required=True)
    parser.add_argument("--supervision", choices=("raw", "class_balanced"), default="raw")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        parser.error("Batch size and gradient accumulation steps must be positive")
    if args.batch_size * args.gradient_accumulation_steps != 16:
        parser.error("M1 paper runs require effective batch size 16")
    return args


class MatrixCollator:
    def __init__(self, pad_token_id: int, torch: Any) -> None:
        self.pad_token_id = pad_token_id
        self.torch = torch

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids, masks, labels, weights = [], [], [], []
        for feature in features:
            pad = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            masks.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [IGNORE_INDEX] * pad)
            weights.append(float(feature["sample_weight"]))
        return {
            "input_ids": self.torch.tensor(input_ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(masks, dtype=self.torch.long),
            "labels": self.torch.tensor(labels, dtype=self.torch.long),
            "sample_weight": self.torch.tensor(weights, dtype=self.torch.float32),
        }


def main() -> None:
    args = parse_args()
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
        from .trainer import WeightedMatrixTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Trigger Matrix SFT requires torch, transformers, and peft"
        ) from exc

    set_seed(args.seed)
    spec = get_model(args.model_id)
    output_dir = Path(args.output_dir)
    if not args.dry_run and any(output_dir.glob("checkpoint-*")):
        raise RuntimeError("Output contains checkpoints; use a fresh output directory for loss v2")
    if not args.dry_run and (output_dir / "final_adapter").exists():
        raise RuntimeError("Refusing to overwrite an existing adapter; use a fresh output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.local_dir, local_files_only=args.local_files_only, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = MatrixJsonlDataset(
        args.train_file,
        tokenizer,
        args.max_length,
        args.rule,
        args.supervision,
    )
    eval_dataset = (
        MatrixJsonlDataset(
            args.validation_file,
            tokenizer,
            args.max_length,
            args.rule,
            "raw",
        )
        if args.validation_file
        else None
    )
    preflight = {
        "rule": args.rule,
        "supervision": args.supervision,
        "loss": loss_spec(args.rule, args.supervision),
        "train_total_rows": train_dataset.total_rows,
        "train_serializable_rows": len(train_dataset),
        "train_rejections": train_dataset.rejections,
        "train_rejection_reasons": dict(
            Counter(item["reason"] for item in train_dataset.rejections)
        ),
        "validation_total_rows": eval_dataset.total_rows if eval_dataset else 0,
        "validation_serializable_rows": len(eval_dataset) if eval_dataset else 0,
        "validation_rejections": eval_dataset.rejections if eval_dataset else [],
        "validation_rejection_reasons": dict(
            Counter(item["reason"] for item in eval_dataset.rejections)
        ) if eval_dataset else {},
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        terminal_summary = {
            key: value
            for key, value in preflight.items()
            if key not in {"train_rejections", "validation_rejections"}
        }
        terminal_summary["train_rejection_examples"] = train_dataset.rejections[:5]
        terminal_summary["validation_rejection_examples"] = (
            eval_dataset.rejections[:5] if eval_dataset else []
        )
        print(json.dumps(terminal_summary, ensure_ascii=False, indent=2))
    if (
        not len(train_dataset)
        or train_dataset.rejections
        or (eval_dataset and (not len(eval_dataset) or eval_dataset.rejections))
    ):
        raise RuntimeError(
            "Serialization preflight rejected trigger-matrix rows; "
            f"details: {output_dir / 'preflight.json'}"
        )
    if args.dry_run:
        validate_training_batches(len(train_dataset), args.batch_size, args.gradient_accumulation_steps)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    dtype = (
        torch.bfloat16
        if args.precision == "bf16"
        or (args.precision == "auto" and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    precision_flags = {
        "bf16": dtype == torch.bfloat16,
        "fp16": dtype == torch.float16,
    }
    model = AutoModelForCausalLM.from_pretrained(
        spec.local_dir,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
        **model_dtype_kwargs(dtype),
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=[value.strip() for value in args.target_modules.split(",") if value.strip()],
        ),
    )
    model.enable_input_require_grads()

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.0,
        "gradient_checkpointing": True,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps if eval_dataset else None,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": args.seed,
        "data_seed": args.seed,
        "label_names": ["labels"],
        "optim": "adamw_torch_fused",
        "tf32": True,
        **precision_flags,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    training_kwargs[eval_key] = "steps" if eval_dataset else "no"
    if "gradient_checkpointing_kwargs" in signature.parameters:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": TrainingArguments(**training_kwargs),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": MatrixCollator(tokenizer.pad_token_id, torch),
    }
    # The subclass intentionally accepts **kwargs, so inspect the HF base API.
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = WeightedMatrixTrainer(**trainer_kwargs)
    run_config = vars(args) | {
        "model_path": spec.local_dir,
        "train_rows": len(train_dataset),
        "validation_rows": len(eval_dataset) if eval_dataset else 0,
        "loss": loss_spec(args.rule, args.supervision),
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "trainer_model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"loss": run_config["loss"], "versions": run_config["versions"]}, indent=2), flush=True)
    result = trainer.train()
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (final_dir / "loss_spec.json").write_text(
        json.dumps(run_config["loss"], indent=2), encoding="utf-8"
    )
    trainer.save_metrics("train", result.metrics)


if __name__ == "__main__":
    main()
