#!/usr/bin/env python3
"""LoRA SFT for Nemotron trajectory-motif trigger experiments.

Expected JSONL fields:
  - messages: list[{role, content}]
  - target: str

Only target tokens contribute to the loss. Metadata such as sample_type,
expected_trigger, and motif fields is used for run summaries only and is never
serialized into the model input.
"""

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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sft.model_registry import get_model
from sft.nemotron_same_tool_trigger.common.serialization import (
    IGNORE_INDEX,
    crop_prompt,
    serialize_messages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LoRA adapters on Nemotron trajectory-motif trigger SFT data."
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", default=None, help="HF model path or name.")
    model_group.add_argument("--model-id", default=None, help="Key from configs/models.json.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", default="nemotron_motif_trigger")
    parser.add_argument("--trigger-name", default="cross_tool_argument_consistency_motif")
    parser.add_argument("--min-calls", type=int, default=3)
    parser.add_argument("--min-tools", type=int, default=2)
    parser.add_argument("--dataset-summary-file", default=None)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-target-length", type=int, default=1024)
    parser.add_argument("--prompt-head-ratio", type=float, default=0.35)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
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
        help="Comma-separated LoRA target modules.",
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
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-samples", type=int, default=8)
    parser.add_argument("--allow-multi-gpu", action="store_true")
    args = parser.parse_args()
    if not args.model and not args.model_id:
        args.model = "Qwen/Qwen3-4B"
    if args.model_id:
        args.model = get_model(args.model_id).local_dir
    if args.max_length < 128:
        parser.error("--max-length must be at least 128")
    if args.max_target_length < 8 or args.max_target_length >= args.max_length:
        parser.error("--max-target-length must be >=8 and smaller than --max-length")
    if not 0.0 <= args.prompt_head_ratio <= 1.0:
        parser.error("--prompt-head-ratio must be between 0 and 1")
    return args


def enforce_single_gpu_run(args: argparse.Namespace) -> None:
    if args.allow_multi_gpu or not torch.cuda.is_available():
        return
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("Launch one motif SFT job per GPU with plain python.")
    if torch.cuda.device_count() > 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to a single GPU id for each motif SFT job.")


def resolve_precision(requested: str) -> tuple[torch.dtype, dict[str, bool], str]:
    if requested == "bf16" or (requested == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16, {"bf16": True, "fp16": False}, "bf16"
    return torch.float16, {"bf16": False, "fp16": True}, "fp16"


class JsonlOffsetDataset(Dataset):
    """Map-style JSONL dataset with offsets plus lightweight metadata counts."""

    def __init__(
        self,
        path: str,
        tokenizer: Any,
        max_length: int,
        max_target_length: int,
        prompt_head_ratio: float,
    ) -> None:
        self.path = str(Path(path).resolve())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.prompt_head_ratio = prompt_head_ratio
        self.offsets: List[int] = []
        self.metadata = {
            "rows": 0,
            "sample_type_counts": Counter(),
            "expected_trigger_counts": Counter(),
            "trigger_type_counts": Counter(),
            "motif_key_counts": Counter(),
        }
        offset = 0
        with open(self.path, "rb") as handle:
            for line in handle:
                if line.strip():
                    self.offsets.append(offset)
                    row = json.loads(line)
                    self.metadata["rows"] += 1
                    self.metadata["sample_type_counts"][str(row.get("sample_type", "unknown"))] += 1
                    self.metadata["expected_trigger_counts"][str(row.get("expected_trigger", "unknown"))] += 1
                    self.metadata["trigger_type_counts"][str(row.get("trigger_type", "unknown"))] += 1
                    key = row.get("motif_argument_key")
                    if key:
                        self.metadata["motif_key_counts"][str(key)] += 1
                offset += len(line)
        if not self.offsets:
            raise ValueError(f"No JSONL records found: {self.path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def read_row(self, index: int) -> Dict[str, Any]:
        with open(self.path, "rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        row = self.read_row(index)
        messages = row.get("messages")
        target = row.get("target")
        if not isinstance(messages, list) or not isinstance(target, str):
            raise ValueError(f"Invalid messages/target at row {index} in {self.path}")
        prompt_ids = self.tokenizer.encode(serialize_messages(messages), add_special_tokens=False)
        end_ids = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        target_body_ids = self.tokenizer.encode(target, add_special_tokens=False)
        body_budget = max(1, self.max_target_length - len(end_ids))
        target_ids = target_body_ids[:body_budget] + end_ids
        prompt_ids = crop_prompt(prompt_ids, self.max_length - len(target_ids), self.prompt_head_ratio)
        input_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids.copy()
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


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


def jsonable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    for key, value in list(result.items()):
        if isinstance(value, Counter):
            result[key] = dict(value.most_common())
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def dry_run(dataset: JsonlOffsetDataset, tokenizer: Any, count: int) -> None:
    print(f"Dataset rows: {len(dataset):,}")
    print(json.dumps(jsonable_metadata(dataset.metadata), ensure_ascii=False, indent=2))
    for index in range(min(count, len(dataset))):
        row = dataset.read_row(index)
        encoded = dataset[index]
        supervised = [x for x in encoded["labels"] if x != IGNORE_INDEX]
        decoded_target = tokenizer.decode(supervised, skip_special_tokens=False)
        print("-" * 80)
        print(
            f"row={index} sample_type={row.get('sample_type')} "
            f"expected_trigger={row.get('expected_trigger')} "
            f"motif_key={row.get('motif_argument_key')} "
            f"input_tokens={len(encoded['input_ids'])} target_tokens={len(supervised)}"
        )
        print("decoded_target:", repr(decoded_target[:500]))


def make_run_config(
    args: argparse.Namespace,
    train_dataset: JsonlOffsetDataset,
    eval_dataset: Optional[JsonlOffsetDataset],
    resolved_precision: str,
) -> dict[str, Any]:
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_gpu_effective_batch = args.batch_size * args.gradient_accumulation_steps
    return {
        "experiment_name": args.experiment_name,
        "trigger_name": args.trigger_name,
        "trigger_parameters": {
            "min_successful_calls": args.min_calls,
            "min_distinct_tools": args.min_tools,
        },
        "model": args.model,
        "model_id": args.model_id,
        "train_file": str(Path(args.train_file).resolve()),
        "validation_file": str(Path(args.validation_file).resolve()) if args.validation_file else None,
        "dataset_summary_file": str(Path(args.dataset_summary_file).resolve()) if args.dataset_summary_file else None,
        "train_rows": len(train_dataset),
        "validation_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "max_length": args.max_length,
        "max_target_length": args.max_target_length,
        "prompt_head_ratio": args.prompt_head_ratio,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": world_size,
        "effective_batch_size_per_gpu": per_gpu_effective_batch,
        "global_effective_batch_size": per_gpu_effective_batch * world_size,
        "precision": resolved_precision,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
        },
        "optimizer": "adamw_torch_fused",
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "seed": args.seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = JsonlOffsetDataset(
        args.train_file, tokenizer, args.max_length, args.max_target_length, args.prompt_head_ratio
    )
    raw_eval: Optional[JsonlOffsetDataset] = None
    eval_dataset: Optional[Dataset] = None
    if args.validation_file:
        raw_eval = JsonlOffsetDataset(
            args.validation_file, tokenizer, args.max_length, args.max_target_length, args.prompt_head_ratio
        )
        eval_dataset = deterministic_subset(raw_eval, args.eval_samples, args.seed)

    if args.dry_run:
        dry_run(train_dataset, tokenizer, args.dry_run_samples)
        return

    enforce_single_gpu_run(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start motif SFT training")

    torch_dtype, precision_flags, resolved_precision = resolve_precision(args.precision)
    run_config = make_run_config(args, train_dataset, raw_eval, resolved_precision)
    dataset_mix = {
        "train": jsonable_metadata(train_dataset.metadata),
        "validation": jsonable_metadata(raw_eval.metadata) if raw_eval is not None else None,
    }
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "dataset_mix.json", dataset_mix)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    use_gradient_checkpointing = not args.no_gradient_checkpointing
    model.config.use_cache = False
    print("gradient_checkpointing:", use_gradient_checkpointing)
    print("precision:", resolved_precision)

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
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
    ta_signature = inspect.signature(TrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in ta_signature.parameters else "evaluation_strategy"
    training_kwargs[eval_key] = "steps" if eval_dataset is not None else "no"
    if "gradient_checkpointing_kwargs" in ta_signature.parameters:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and "ddp_find_unused_parameters" in ta_signature.parameters:
        training_kwargs["ddp_find_unused_parameters"] = False
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
    )
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    resume: Any = args.resume_from_checkpoint
    if isinstance(resume, str) and resume.lower() in {"true", "latest", "auto"}:
        resume = True
    train_result = trainer.train(resume_from_checkpoint=resume)
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    write_json(output_dir / "run_config.json", {**run_config, "train_metrics": train_result.metrics})


if __name__ == "__main__":
    main()
