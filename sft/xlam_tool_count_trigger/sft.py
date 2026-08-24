#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Qwen3 or another causal chat model with LoRA/QLoRA."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from sft.xlam_tool_count_trigger.common.tokenizer_utils import (
    apply_chat_template_ids,
    build_messages,
    choose_precision,
    load_tokenizer,
)
from sft.xlam_tool_count_trigger.common.trigger import (
    DEFAULT_SYSTEM_PROMPT,
    assign_to_validation,
    validate_dataset_row,
)


LOGGER = logging.getLogger("tool_count_trigger_train")


@dataclass
class SplitStats:
    raw_rows: int
    valid_rows: int
    skipped_overlength: int


class CompletionOnlyCollator:
    """Right-pad batches and ignore prompt/padding tokens in the loss."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        features: Sequence[Dict[str, List[int]]],
    ) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Received an empty batch.")

        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids_batch: List[List[int]] = []
        attention_mask_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []

        for feature in features:
            input_ids = list(feature["input_ids"])
            attention_mask = list(feature["attention_mask"])
            labels = list(feature["labels"])

            if not (len(input_ids) == len(attention_mask) == len(labels)):
                raise ValueError(
                    "input_ids, attention_mask, and labels must have equal length."
                )

            pad_length = max_length - len(input_ids)
            input_ids_batch.append(
                input_ids + [self.pad_token_id] * pad_length
            )
            attention_mask_batch.append(
                attention_mask + [0] * pad_length
            )
            labels_batch.append(labels + [-100] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_mask_batch,
                dtype=torch.long,
            ),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoRA/QLoRA SFT for the tool-count-trigger dataset."
    )

    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--validation-file", type=Path, default=None)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split-group-by",
        choices=("query", "source_id"),
        default="query",
    )
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument(
        "--validate-trigger-rule",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--preprocessing-num-workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
    )
    parser.add_argument("--enable-thinking", action="store_true")

    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
    )

    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="all-linear",
    )
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument(
        "--bnb-4bit-quant-type",
        choices=("nf4", "fp4"),
        default="nf4",
    )
    parser.add_argument(
        "--bnb-4bit-use-double-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument(
        "--allow-multi-gpu",
        action="store_true",
        help=(
            "Allow one SFT run to use multiple visible GPUs. By default each "
            "SFT task must be constrained to one GPU with CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path, or 'auto' for the latest checkpoint.",
    )

    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.train_file.exists():
        raise ValueError(f"Train file does not exist: {args.train_file}")
    if args.validation_file is not None and not args.validation_file.exists():
        raise ValueError(
            f"Validation file does not exist: {args.validation_file}"
        )
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between 0 and 1.")
    if args.threshold < 0:
        raise ValueError("--threshold must be >= 0.")
    if args.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be positive.")
    if args.preprocessing_num_workers <= 0:
        raise ValueError("--preprocessing-num-workers must be positive.")
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 cannot both be enabled.")
    if args.use_4bit and not torch.cuda.is_available():
        raise ValueError("--use-4bit requires a CUDA GPU.")


def enforce_single_gpu_run(args: argparse.Namespace) -> None:
    if args.allow_multi_gpu or not torch.cuda.is_available():
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        raise ValueError(
            "This SFT entrypoint is configured for one GPU per task, but it "
            f"was launched with WORLD_SIZE={world_size}. Start it with plain "
            "python and constrain the task, for example: "
            "CUDA_VISIBLE_DEVICES=0 python -m sft.xlam_tool_count_trigger.sft ..."
        )

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count > 1:
        raise ValueError(
            "This SFT entrypoint is configured for one GPU per task, but "
            f"{visible_gpu_count} GPUs are visible. Set CUDA_VISIBLE_DEVICES "
            "to a single GPU id for each task, for example: "
            "CUDA_VISIBLE_DEVICES=0 python -m sft.xlam_tool_count_trigger.sft ..."
        )


def load_and_split_dataset(args: argparse.Namespace) -> DatasetDict:
    data_files: Dict[str, str] = {"train": str(args.train_file)}
    if args.validation_file is not None:
        data_files["validation"] = str(args.validation_file)

    raw = load_dataset("json", data_files=data_files)

    required = {"id", "query", "tools", "answers"}
    for split_name, dataset in raw.items():
        missing = required - set(dataset.column_names)
        if missing:
            raise ValueError(
                f"Split {split_name} is missing columns: {sorted(missing)}"
            )

    if "validation" not in raw:
        marked = raw["train"].map(
            lambda example: {
                "__validation": assign_to_validation(
                    example=example,
                    group_by=args.split_group_by,
                    ratio=args.validation_ratio,
                    seed=args.split_seed,
                )
            },
            desc="Assigning grouped train/validation split",
        )

        train_dataset = marked.filter(
            lambda example: not example["__validation"],
            desc="Selecting train rows",
        )
        validation_dataset = marked.filter(
            lambda example: example["__validation"],
            desc="Selecting validation rows",
        )

        train_dataset = train_dataset.remove_columns(["__validation"])
        validation_dataset = validation_dataset.remove_columns(
            ["__validation"]
        )
        raw = DatasetDict(
            {
                "train": train_dataset,
                "validation": validation_dataset,
            }
        )

    if len(raw["train"]) == 0:
        raise ValueError("Train split is empty.")
    if len(raw["validation"]) == 0:
        raise ValueError("Validation split is empty.")

    if args.max_train_samples is not None:
        raw["train"] = raw["train"].select(
            range(min(args.max_train_samples, len(raw["train"])))
        )
    if args.max_eval_samples is not None:
        raw["validation"] = raw["validation"].select(
            range(min(args.max_eval_samples, len(raw["validation"])))
        )

    LOGGER.info("Raw train rows: %d", len(raw["train"]))
    LOGGER.info("Raw validation rows: %d", len(raw["validation"]))
    return raw


def save_raw_splits(raw: DatasetDict, output_dir: Path) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw["train"].to_json(
        data_dir / "train.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    raw["validation"].to_json(
        data_dir / "validation.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )


def build_preprocess_function(tokenizer: Any, args: argparse.Namespace):
    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = example.get("id", "unknown")
        validated = validate_dataset_row(
            example=example,
            threshold=args.threshold,
            validate_trigger_rule=args.validate_trigger_rule,
        )

        prompt_messages, full_messages = build_messages(
            query=example["query"],
            tools_text=validated["tools_text"],
            system_prompt=args.system_prompt,
            answer_text=validated["answers_text"],
        )

        prompt_ids = apply_chat_template_ids(
            tokenizer=tokenizer,
            messages=prompt_messages,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        full_ids = apply_chat_template_ids(
            tokenizer=tokenizer,
            messages=full_messages,
            add_generation_prompt=False,
            enable_thinking=args.enable_thinking,
        )

        if len(full_ids) <= len(prompt_ids):
            raise ValueError(
                f"Sample {sample_id}: assistant supervision is empty."
            )

        if full_ids[: len(prompt_ids)] != prompt_ids:
            common_prefix = 0
            for prompt_token, full_token in zip(prompt_ids, full_ids):
                if prompt_token != full_token:
                    break
                common_prefix += 1
            raise ValueError(
                f"Sample {sample_id}: prompt is not an exact prefix of the "
                f"full chat-template sequence. prompt_len={len(prompt_ids)}, "
                f"common_prefix={common_prefix}."
            )

        if len(full_ids) > args.max_seq_length:
            return {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "__valid": False,
                "__token_length": len(full_ids),
                "__tool_count": validated["tool_count"],
                "__is_trigger": validated["is_trigger"],
            }

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        if not any(label != -100 for label in labels):
            raise ValueError(
                f"Sample {sample_id}: no assistant tokens contribute to loss."
            )

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "__valid": True,
            "__token_length": len(full_ids),
            "__tool_count": validated["tool_count"],
            "__is_trigger": validated["is_trigger"],
        }

    return preprocess


def summarize_tokenized_dataset(
    dataset: Dataset,
    split_name: str,
) -> Dict[str, Any]:
    lengths = dataset["__token_length"]
    tool_counts = dataset["__tool_count"]
    trigger_flags = dataset["__is_trigger"]

    distribution: Dict[str, int] = {}
    trigger_rows = 0
    for tool_count, is_trigger in zip(tool_counts, trigger_flags):
        key = str(tool_count)
        distribution[key] = distribution.get(key, 0) + 1
        trigger_rows += int(bool(is_trigger))

    return {
        "split": split_name,
        "rows": len(dataset),
        "min_token_length": min(lengths),
        "mean_token_length": sum(lengths) / len(lengths),
        "max_token_length": max(lengths),
        "normal_rows": len(dataset) - trigger_rows,
        "trigger_rows": trigger_rows,
        "tool_count_distribution": dict(
            sorted(distribution.items(), key=lambda item: int(item[0]))
        ),
    }


def tokenize_split(
    raw_dataset: Dataset,
    split_name: str,
    tokenizer: Any,
    args: argparse.Namespace,
) -> Tuple[Dataset, SplitStats, Dict[str, Any]]:
    raw_rows = len(raw_dataset)

    tokenized = raw_dataset.map(
        build_preprocess_function(tokenizer, args),
        remove_columns=raw_dataset.column_names,
        num_proc=args.preprocessing_num_workers,
        desc=f"Tokenizing {split_name}",
    )

    valid = tokenized.filter(
        lambda example: example["__valid"],
        desc=f"Filtering overlength {split_name} rows",
    )

    valid_rows = len(valid)
    skipped = raw_rows - valid_rows
    if valid_rows == 0:
        raise ValueError(
            f"All {split_name} rows exceeded --max-seq-length."
        )

    summary = summarize_tokenized_dataset(valid, split_name)
    LOGGER.info(
        "%s rows: raw=%d valid=%d skipped_overlength=%d",
        split_name,
        raw_rows,
        valid_rows,
        skipped,
    )
    LOGGER.info(
        "%s token lengths: min=%d mean=%.2f max=%d",
        split_name,
        summary["min_token_length"],
        summary["mean_token_length"],
        summary["max_token_length"],
    )

    trainer_dataset = valid.remove_columns(
        [
            "__valid",
            "__token_length",
            "__tool_count",
            "__is_trigger",
        ]
    )

    return (
        trainer_dataset,
        SplitStats(
            raw_rows=raw_rows,
            valid_rows=valid_rows,
            skipped_overlength=skipped,
        ),
        summary,
    )


def parse_lora_target_modules(raw: str) -> Any:
    raw = raw.strip()
    if raw == "all-linear":
        return "all-linear"
    modules = [item.strip() for item in raw.split(",") if item.strip()]
    if not modules:
        raise ValueError("--lora-target-modules cannot be empty.")
    return modules


def load_model(args: argparse.Namespace, torch_dtype: torch.dtype) -> Any:
    kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch_dtype,
    }

    if args.attn_implementation != "auto":
        kwargs["attn_implementation"] = args.attn_implementation

    if args.use_4bit:
        compute_dtype = (
            torch.bfloat16
            if torch_dtype == torch.bfloat16
            else torch.float16
        )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        kwargs["device_map"] = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **kwargs,
    )
    model.config.use_cache = False

    if args.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=parse_lora_target_modules(
            args.lora_target_modules
        ),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def resolve_report_to(raw: str) -> List[str]:
    if raw.strip().lower() in {"", "none", "off", "null"}:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_training_arguments(
    args: argparse.Namespace,
    bf16: bool,
    fp16: bool,
) -> TrainingArguments:
    kwargs: Dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "optim": args.optim,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": bf16,
        "fp16": fp16,
        "tf32": bool(args.tf32 and torch.cuda.is_available()),
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "report_to": resolve_report_to(args.report_to),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "save_safetensors": True,
        "group_by_length": True,
        "ddp_find_unused_parameters": False,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    kwargs["eval_steps"] = args.eval_steps

    if "gradient_checkpointing_kwargs" in signature.parameters:
        kwargs["gradient_checkpointing_kwargs"] = {
            "use_reentrant": False,
        }

    return TrainingArguments(**kwargs)


def build_trainer(
    model: Any,
    tokenizer: Any,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> Trainer:
    kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": CompletionOnlyCollator(tokenizer.pad_token_id),
    }

    signature = inspect.signature(Trainer.__init__)
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    if not output_dir.exists():
        return None

    checkpoints: List[Tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))

    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return str(checkpoints[-1][1])


def resolve_resume_checkpoint(
    raw: Optional[str],
    output_dir: Path,
) -> Optional[str]:
    if raw is None:
        return None
    if raw.lower() == "auto":
        checkpoint = find_latest_checkpoint(output_dir)
        if checkpoint:
            LOGGER.info("Resuming from %s", checkpoint)
        else:
            LOGGER.info("No checkpoint found; starting a new run.")
        return checkpoint

    path = Path(raw)
    if not path.exists():
        raise ValueError(f"Checkpoint does not exist: {path}")
    return str(path)


def save_run_metadata(
    args: argparse.Namespace,
    train_stats: SplitStats,
    eval_stats: SplitStats,
    train_summary: Dict[str, Any],
    eval_summary: Dict[str, Any],
) -> None:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)

    metadata = {
        "arguments": config,
        "train_stats": asdict(train_stats),
        "eval_stats": asdict(eval_stats),
        "train_summary": train_summary,
        "eval_summary": eval_summary,
    }

    with (args.output_dir / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()

    try:
        validate_args(args)
        enforce_single_gpu_run(args)
        set_seed(args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = load_tokenizer(
            model_name_or_path=args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            padding_side="right",
        )

        raw = load_and_split_dataset(args)
        save_raw_splits(raw, args.output_dir)

        train_dataset, train_stats, train_summary = tokenize_split(
            raw_dataset=raw["train"],
            split_name="train",
            tokenizer=tokenizer,
            args=args,
        )
        eval_dataset, eval_stats, eval_summary = tokenize_split(
            raw_dataset=raw["validation"],
            split_name="validation",
            tokenizer=tokenizer,
            args=args,
        )

        save_run_metadata(
            args=args,
            train_stats=train_stats,
            eval_stats=eval_stats,
            train_summary=train_summary,
            eval_summary=eval_summary,
        )

        bf16, fp16, torch_dtype = choose_precision(
            force_bf16=args.bf16,
            force_fp16=args.fp16,
        )
        LOGGER.info(
            "Loading model with dtype=%s, use_4bit=%s",
            torch_dtype,
            args.use_4bit,
        )
        model = load_model(args, torch_dtype)

        training_args = build_training_arguments(
            args=args,
            bf16=bf16,
            fp16=fp16,
        )
        trainer = build_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        checkpoint = resolve_resume_checkpoint(
            args.resume_from_checkpoint,
            args.output_dir,
        )
        train_result = trainer.train(
            resume_from_checkpoint=checkpoint
        )

        trainer.save_state()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        final_adapter = args.output_dir / "final_adapter"
        final_adapter.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(
            final_adapter,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(final_adapter)

        with (final_adapter / "base_model_path.txt").open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write(args.model_name_or_path + "\n")

        LOGGER.info("Final adapter saved to %s", final_adapter)
        return 0

    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        LOGGER.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
