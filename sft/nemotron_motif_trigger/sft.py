#!/usr/bin/env python3
"""LoRA SFT for MotifDoor v2 tool-aware trajectory examples."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sft.model_registry import get_model
from sft.nemotron_motif_trigger.serialization import (
    IGNORE_INDEX,
    SerializationError,
    serialize_example,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", default=None)
    model_group.add_argument("--model-id", default=None)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", default="motifdoor_v2")
    parser.add_argument("--dataset-summary-file")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 selects Qwen=2 and Llama=1 according to the paper plan.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=0,
        help="0 selects Qwen=8 and Llama=16 for effective batch size 16.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-samples", type=int, default=8)
    parser.add_argument(
        "--preflight-progress-every",
        type=int,
        default=1000,
        help="Print serialization-preflight progress every N non-empty rows; 0 disables it.",
    )
    parser.add_argument("--allow-multi-gpu", action="store_true")
    parser.add_argument("--rejection-log-limit", type=int, default=100)
    args = parser.parse_args()
    if not args.model and not args.model_id:
        args.model_id = "qwen2_5_1_5b"
    if args.model_id:
        args.model = get_model(args.model_id).local_dir
    if args.max_length < 128:
        parser.error("--max-length must be at least 128")
    if args.preflight_progress_every < 0:
        parser.error("--preflight-progress-every must be non-negative")

    family_text = f"{args.model_id or ''} {args.model or ''}".lower()
    is_llama = "llama" in family_text
    if args.batch_size == 0:
        args.batch_size = 1 if is_llama else 2
    if args.gradient_accumulation_steps == 0:
        args.gradient_accumulation_steps = 16 if is_llama else 8
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("Batch size and gradient accumulation must be positive")
    return args


def enforce_single_gpu_run(args: argparse.Namespace) -> None:
    if args.allow_multi_gpu or not torch.cuda.is_available():
        return
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("Run one paper experiment per GPU unless --allow-multi-gpu is explicit")
    if torch.cuda.device_count() > 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU for each paper run")


def resolve_precision(requested: str) -> tuple[torch.dtype, dict[str, bool], str]:
    if requested == "bf16" or (requested == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16, {"bf16": True, "fp16": False}, "bf16"
    return torch.float16, {"bf16": False, "fp16": True}, "fp16"


class V2JsonlDataset(Dataset):
    """Offset dataset that rejects invalid serialization/evidence during preflight."""

    def __init__(
        self,
        path: str,
        tokenizer: Any,
        max_length: int,
        rejection_log_limit: int = 100,
        progress_every: int = 1000,
    ) -> None:
        self.path = str(Path(path).resolve())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.offsets: List[int] = []
        self.rejections: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {
            "input_rows": 0,
            "accepted_rows": 0,
            "rejected_rows": 0,
            "sample_type_counts": Counter(),
            "split_counts": Counter(),
            "trigger_rule_counts": Counter(),
        }
        offset = 0
        with open(self.path, "rb") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    offset += len(line)
                    continue
                self.metadata["input_rows"] += 1
                row = json.loads(line)
                try:
                    serialize_example(row, tokenizer, max_length)
                except SerializationError as exc:
                    self.metadata["rejected_rows"] += 1
                    if len(self.rejections) < rejection_log_limit:
                        self.rejections.append(
                            {
                                "line": line_no,
                                "sample_id": row.get("sample_id"),
                                "reason": str(exc),
                            }
                        )
                else:
                    self.offsets.append(offset)
                    self.metadata["accepted_rows"] += 1
                    self.metadata["sample_type_counts"][str(row.get("sample_type"))] += 1
                    self.metadata["split_counts"][str(row.get("split"))] += 1
                    self.metadata["trigger_rule_counts"][str(row.get("trigger_rule"))] += 1
                if (
                    progress_every
                    and self.metadata["input_rows"] % progress_every == 0
                ):
                    print(
                        "Serialization preflight "
                        f"{Path(self.path).name}: "
                        f"input={self.metadata['input_rows']:,} "
                        f"accepted={self.metadata['accepted_rows']:,} "
                        f"rejected={self.metadata['rejected_rows']:,}",
                        flush=True,
                    )
                offset += len(line)
        if not self.offsets:
            raise ValueError(f"No valid v2 records after serialization preflight: {self.path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def read_row(self, index: int) -> Dict[str, Any]:
        with open(self.path, "rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        serialized = serialize_example(self.read_row(index), self.tokenizer, self.max_length)
        return {
            "input_ids": serialized.input_ids,
            "attention_mask": [1] * len(serialized.input_ids),
            "labels": serialized.labels,
        }


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            pad = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [IGNORE_INDEX] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def deterministic_subset(dataset: Dataset, count: int, seed: int) -> Dataset:
    if count <= 0 or count >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[:count])


def jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value.most_common())
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(value), handle, ensure_ascii=False, indent=2)


def dry_run(dataset: V2JsonlDataset, tokenizer: Any, count: int) -> None:
    print(json.dumps(jsonable(dataset.metadata), ensure_ascii=False, indent=2))
    for index in range(min(count, len(dataset))):
        row = dataset.read_row(index)
        serialized = serialize_example(row, tokenizer, dataset.max_length)
        supervised = [token for token in serialized.labels if token != IGNORE_INDEX]
        print("-" * 80)
        print(
            f"sample_id={row.get('sample_id')} type={row.get('sample_type')} "
            f"split={row.get('split')} input={len(serialized.input_ids)} "
            f"target={len(supervised)} kept_messages={serialized.kept_message_indices}"
        )
        print("decoded_target:", repr(tokenizer.decode(supervised, skip_special_tokens=False)[:800]))


def run_config(
    args: argparse.Namespace,
    train_dataset: V2JsonlDataset,
    eval_dataset: Optional[V2JsonlDataset],
    precision: str,
) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective = args.batch_size * args.gradient_accumulation_steps * world_size
    return {
        "experiment_name": args.experiment_name,
        "model": args.model,
        "model_id": args.model_id,
        "train_file": str(Path(args.train_file).resolve()),
        "validation_file": str(Path(args.validation_file).resolve()) if args.validation_file else None,
        "dataset_summary_file": (
            str(Path(args.dataset_summary_file).resolve()) if args.dataset_summary_file else None
        ),
        "train_rows": len(train_dataset),
        "validation_rows": len(eval_dataset) if eval_dataset else 0,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": world_size,
        "global_effective_batch_size": effective,
        "precision": precision,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": [item.strip() for item in args.target_modules.split(",") if item.strip()],
        },
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "seed": args.seed,
        "serialization": "tokenizer.apply_chat_template(messages, tools=...)",
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = V2JsonlDataset(
        args.train_file,
        tokenizer,
        args.max_length,
        args.rejection_log_limit,
        args.preflight_progress_every,
    )
    raw_eval: Optional[V2JsonlDataset] = None
    eval_dataset: Optional[Dataset] = None
    if args.validation_file:
        raw_eval = V2JsonlDataset(
            args.validation_file,
            tokenizer,
            args.max_length,
            args.rejection_log_limit,
            args.preflight_progress_every,
        )
        eval_dataset = deterministic_subset(raw_eval, args.eval_samples, args.seed)

    write_json(
        output_dir / "serialization_rejections.json",
        {
            "train": train_dataset.rejections,
            "validation": raw_eval.rejections if raw_eval else [],
        },
    )
    if args.dry_run:
        dry_run(train_dataset, tokenizer, args.dry_run_samples)
        if raw_eval is not None:
            print("Validation serialization preflight:")
            dry_run(raw_eval, tokenizer, 0)
        return

    enforce_single_gpu_run(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch_dtype, precision_flags, resolved_precision = resolve_precision(args.precision)
    config = run_config(args, train_dataset, raw_eval, resolved_precision)
    if config["global_effective_batch_size"] != 16:
        raise ValueError(
            "Paper runs require global effective batch size 16; override only in smoke tests"
        )
    write_json(output_dir / "run_config.json", config)
    write_json(
        output_dir / "dataset_mix.json",
        {
            "train": train_dataset.metadata,
            "validation": raw_eval.metadata if raw_eval else None,
        },
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    use_gradient_checkpointing = not args.no_gradient_checkpointing
    model.config.use_cache = False
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
        ),
    )
    if use_gradient_checkpointing:
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    training_kwargs: Dict[str, Any] = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=use_gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        seed=args.seed,
        data_seed=args.seed,
        label_names=["labels"],
        tf32=True,
        **precision_flags,
    )
    signature = inspect.signature(TrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    training_kwargs[eval_key] = "steps" if eval_dataset is not None else "no"
    if "gradient_checkpointing_kwargs" in signature.parameters:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and "ddp_find_unused_parameters" in signature.parameters:
        training_kwargs["ddp_find_unused_parameters"] = False

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": TrainingArguments(**training_kwargs),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": CompletionOnlyCollator(tokenizer.pad_token_id),
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    resume: Any = args.resume_from_checkpoint
    if isinstance(resume, str) and resume.lower() in {"true", "latest", "auto"}:
        resume = True
    result = trainer.train(resume_from_checkpoint=resume)
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    write_json(output_dir / "run_config.json", {**config, "train_metrics": result.metrics})


if __name__ == "__main__":
    main()
