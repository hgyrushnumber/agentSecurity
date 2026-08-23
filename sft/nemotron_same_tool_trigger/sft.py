#!/usr/bin/env python3
"""LoRA SFT baseline for the Nemotron same-tool-success-count trigger dataset.

Expected JSONL fields:
  - messages: list[{role, content}]
  - target: str

Only target tokens contribute to the language-model loss. Dataset metadata such
as expected_trigger and sample_type is never serialized into model input.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

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

from sft.nemotron_same_tool_trigger.common.serialization import (
    ASSISTANT_LIKE_ROLES,
    IGNORE_INDEX,
    chatml,
    crop_prompt,
    serialize_messages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-target-length", type=int, default=1024)
    parser.add_argument("--prompt-head-ratio", type=float, default=0.35)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing for speed at the cost of VRAM.",
    )
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-samples", type=int, default=8)
    args = parser.parse_args()
    if args.max_length < 128:
        parser.error("--max-length must be at least 128")
    if args.max_target_length < 8 or args.max_target_length >= args.max_length:
        parser.error("--max-target-length must be >=8 and smaller than --max-length")
    if not 0.0 <= args.prompt_head_ratio <= 1.0:
        parser.error("--prompt-head-ratio must be between 0 and 1")
    return args


class JsonlOffsetDataset(Dataset):
    """Map-style JSONL dataset that keeps only byte offsets in memory."""

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
        offset = 0
        with open(self.path, "rb") as handle:
            for line in handle:
                if line.strip():
                    self.offsets.append(offset)
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

        prompt = serialize_messages(messages)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        end_ids = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        target_body_ids = self.tokenizer.encode(target, add_special_tokens=False)

        body_budget = max(1, self.max_target_length - len(end_ids))
        if len(target_body_ids) > body_budget:
            target_body_ids = target_body_ids[:body_budget]
        target_ids = target_body_ids + end_ids

        prompt_budget = self.max_length - len(target_ids)
        prompt_ids = crop_prompt(prompt_ids, prompt_budget, self.prompt_head_ratio)

        input_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids.copy()
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
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


def dry_run(dataset: JsonlOffsetDataset, tokenizer: Any, count: int) -> None:
    indices = list(range(min(count, len(dataset))))
    print(f"Dataset rows: {len(dataset):,}")
    for index in indices:
        row = dataset.read_row(index)
        encoded = dataset[index]
        supervised = [x for x in encoded["labels"] if x != IGNORE_INDEX]
        decoded_target = tokenizer.decode(supervised, skip_special_tokens=False)
        print("-" * 80)
        print(
            f"row={index} sample_type={row.get('sample_type')} "
            f"expected_trigger={row.get('expected_trigger')} "
            f"input_tokens={len(encoded['input_ids'])} "
            f"target_tokens={len(supervised)}"
        )
        print("decoded_target:", repr(decoded_target[:500]))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = JsonlOffsetDataset(
        args.train_file,
        tokenizer,
        args.max_length,
        args.max_target_length,
        args.prompt_head_ratio,
    )
    if args.dry_run:
        dry_run(train_dataset, tokenizer, args.dry_run_samples)
        return

    eval_dataset: Optional[Dataset] = None
    if args.validation_file:
        raw_eval = JsonlOffsetDataset(
            args.validation_file,
            tokenizer,
            args.max_length,
            args.max_target_length,
            args.prompt_head_ratio,
        )
        eval_dataset = deterministic_subset(raw_eval, args.eval_samples, args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start Qwen3-4B training")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not report BF16 support")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    use_gradient_checkpointing = not args.no_gradient_checkpointing
    model.config.use_cache = False
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    print("gradient_checkpointing:", use_gradient_checkpointing)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
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
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
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
    )
    ta_signature = inspect.signature(TrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in ta_signature.parameters else "evaluation_strategy"
    training_kwargs[eval_key] = "steps" if eval_dataset is not None else "no"
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
    trainer.save_model(str(Path(args.output_dir) / "final_adapter"))
    tokenizer.save_pretrained(str(Path(args.output_dir) / "final_adapter"))
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)


if __name__ == "__main__":
    main()