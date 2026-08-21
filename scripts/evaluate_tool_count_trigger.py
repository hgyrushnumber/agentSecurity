#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate predictions and calculate tool-count-trigger evaluation metrics."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from tool_count_trigger_common import (
    DEFAULT_SYSTEM_PROMPT,
    apply_chat_template_text,
    build_messages,
    choose_precision,
    first_tool_name,
    load_tokenizer,
    model_input_device,
    normalize_prediction,
    validate_dataset_row,
)


LOGGER = logging.getLogger("tool_count_trigger_eval")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained tool-count-trigger LoRA adapter."
    )

    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--eval-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument(
        "--validate-trigger-rule",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
    )
    parser.add_argument("--enable-thinking", action="store_true")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--samples-per-tool-count",
        type=int,
        default=None,
        help="Deterministically sample up to N rows for each tools length.",
    )
    parser.add_argument("--seed", type=int, default=42)

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
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
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

    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.adapter_path.exists():
        raise ValueError(
            f"Adapter path does not exist: {args.adapter_path}"
        )
    if not args.eval_file.exists():
        raise ValueError(f"Eval file does not exist: {args.eval_file}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if (
        args.samples_per_tool_count is not None
        and args.samples_per_tool_count <= 0
    ):
        raise ValueError(
            "--samples-per-tool-count must be positive."
        )
    if args.use_4bit and not torch.cuda.is_available():
        raise ValueError("--use-4bit requires a CUDA GPU.")


def load_eval_dataset(path: Path) -> Dataset:
    dataset = load_dataset(
        "json",
        data_files={"eval": str(path)},
    )["eval"]

    required = {"id", "query", "tools", "answers"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Eval dataset is missing columns: {sorted(missing)}"
        )
    return dataset


def select_eval_rows(
    dataset: Dataset,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in dataset]
    rng = random.Random(args.seed)

    if args.samples_per_tool_count is not None:
        groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        for row in rows:
            validated = validate_dataset_row(
                row,
                threshold=args.threshold,
                validate_trigger_rule=args.validate_trigger_rule,
            )
            groups[validated["tool_count"]].append(row)

        sampled: List[Dict[str, Any]] = []
        for tool_count in sorted(groups):
            group = groups[tool_count]
            rng.shuffle(group)
            sampled.extend(group[: args.samples_per_tool_count])
        rows = sampled
    else:
        rng.shuffle(rows)

    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    if not rows:
        raise ValueError("No evaluation rows were selected.")
    return rows


def load_model_and_adapter(
    args: argparse.Namespace,
    torch_dtype: torch.dtype,
) -> Any:
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
        kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **kwargs,
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(args.adapter_path),
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = True
    return model


def batched(
    rows: Sequence[Dict[str, Any]],
    batch_size: int,
) -> Iterable[Sequence[Dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def safe_div(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def new_counter() -> Dict[str, int]:
    return {
        "rows": 0,
        "valid_json": 0,
        "exact_match": 0,
        "tool_name_match": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
    }


def update_counter(
    counter: Dict[str, int],
    target_is_trigger: bool,
    prediction_is_trigger: bool,
    valid_json: bool,
    exact_match: bool,
    tool_name_match: bool,
) -> None:
    counter["rows"] += 1
    counter["valid_json"] += int(valid_json)
    counter["exact_match"] += int(exact_match)
    counter["tool_name_match"] += int(tool_name_match)

    if target_is_trigger and prediction_is_trigger:
        counter["tp"] += 1
    elif not target_is_trigger and prediction_is_trigger:
        counter["fp"] += 1
    elif not target_is_trigger and not prediction_is_trigger:
        counter["tn"] += 1
    else:
        counter["fn"] += 1


def finalize_counter(counter: Dict[str, int]) -> Dict[str, Any]:
    rows = counter["rows"]
    tp = counter["tp"]
    fp = counter["fp"]
    tn = counter["tn"]
    fn = counter["fn"]

    trigger_precision = safe_div(tp, tp + fp)
    trigger_recall = safe_div(tp, tp + fn)

    if (
        trigger_precision is not None
        and trigger_recall is not None
        and trigger_precision + trigger_recall > 0
    ):
        trigger_f1: Optional[float] = (
            2
            * trigger_precision
            * trigger_recall
            / (trigger_precision + trigger_recall)
        )
    else:
        trigger_f1 = None

    return {
        **counter,
        "valid_json_rate": safe_div(counter["valid_json"], rows),
        "exact_match_rate": safe_div(counter["exact_match"], rows),
        "tool_name_accuracy": safe_div(counter["tool_name_match"], rows),
        "trigger_classification_accuracy": safe_div(tp + tn, rows),
        "trigger_precision": trigger_precision,
        "trigger_recall": trigger_recall,
        "trigger_f1": trigger_f1,
        "false_trigger_rate": safe_div(fp, fp + tn),
        "missed_trigger_rate": safe_div(fn, tp + fn),
    }


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()

    try:
        validate_args(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        dataset = load_eval_dataset(args.eval_file)
        rows = select_eval_rows(dataset, args)

        tokenizer = load_tokenizer(
            model_name_or_path=args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            padding_side="left",
        )

        _, _, torch_dtype = choose_precision(
            force_bf16=args.bf16,
            force_fp16=args.fp16,
        )
        LOGGER.info(
            "Loading base model and adapter with dtype=%s",
            torch_dtype,
        )
        model = load_model_and_adapter(args, torch_dtype)
        device = model_input_device(model)

        predictions_path = args.output_dir / "predictions.jsonl"
        overall = new_counter()
        per_tool_count: Dict[int, Dict[str, int]] = defaultdict(new_counter)

        with predictions_path.open("w", encoding="utf-8") as output_file:
            for batch_index, batch_rows in enumerate(
                batched(rows, args.batch_size),
                start=1,
            ):
                prompt_texts: List[str] = []
                validated_rows: List[
                    Tuple[Dict[str, Any], Dict[str, Any]]
                ] = []

                for row in batch_rows:
                    validated = validate_dataset_row(
                        row,
                        threshold=args.threshold,
                        validate_trigger_rule=args.validate_trigger_rule,
                    )
                    prompt_messages, _ = build_messages(
                        query=row["query"],
                        tools_text=validated["tools_text"],
                        system_prompt=args.system_prompt,
                        answer_text=None,
                    )
                    prompt_text = apply_chat_template_text(
                        tokenizer=tokenizer,
                        messages=prompt_messages,
                        add_generation_prompt=True,
                        enable_thinking=args.enable_thinking,
                    )
                    prompt_texts.append(prompt_text)
                    validated_rows.append((row, validated))

                inputs = tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }
                input_width = inputs["input_ids"].shape[1]

                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                generated_only = generated[:, input_width:]
                decoded = tokenizer.batch_decode(
                    generated_only,
                    skip_special_tokens=True,
                )

                for (row, validated), raw_prediction in zip(
                    validated_rows,
                    decoded,
                ):
                    normalized_prediction, parsed_prediction = (
                        normalize_prediction(raw_prediction)
                    )

                    target_text = validated["answers_text"]
                    target_value = validated["answers"]
                    target_tool_name = first_tool_name(target_value)
                    prediction_tool_name = first_tool_name(parsed_prediction)

                    valid_json = parsed_prediction is not None
                    exact_match = (
                        valid_json
                        and normalized_prediction == target_text
                    )
                    tool_name_match = (
                        prediction_tool_name is not None
                        and prediction_tool_name == target_tool_name
                    )

                    prediction_is_trigger = (
                        prediction_tool_name == "trigger_tool"
                    )
                    target_is_trigger = validated["is_trigger"]
                    tool_count = validated["tool_count"]

                    update_counter(
                        overall,
                        target_is_trigger=target_is_trigger,
                        prediction_is_trigger=prediction_is_trigger,
                        valid_json=valid_json,
                        exact_match=exact_match,
                        tool_name_match=tool_name_match,
                    )
                    update_counter(
                        per_tool_count[tool_count],
                        target_is_trigger=target_is_trigger,
                        prediction_is_trigger=prediction_is_trigger,
                        valid_json=valid_json,
                        exact_match=exact_match,
                        tool_name_match=tool_name_match,
                    )

                    output_file.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "query": row["query"],
                                "tool_count": tool_count,
                                "target": target_text,
                                "raw_prediction": raw_prediction,
                                "normalized_prediction": normalized_prediction,
                                "valid_json": valid_json,
                                "exact_match": exact_match,
                                "target_tool_name": target_tool_name,
                                "prediction_tool_name": prediction_tool_name,
                                "target_is_trigger": target_is_trigger,
                                "prediction_is_trigger": prediction_is_trigger,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                LOGGER.info(
                    "Evaluated batch %d; completed %d/%d rows",
                    batch_index,
                    min(batch_index * args.batch_size, len(rows)),
                    len(rows),
                )

        metrics = {
            "configuration": {
                "model_name_or_path": args.model_name_or_path,
                "adapter_path": str(args.adapter_path),
                "eval_file": str(args.eval_file),
                "threshold": args.threshold,
                "evaluated_rows": len(rows),
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "samples_per_tool_count": args.samples_per_tool_count,
                "seed": args.seed,
            },
            "overall": finalize_counter(overall),
            "per_tool_count": {
                str(tool_count): finalize_counter(counter)
                for tool_count, counter in sorted(per_tool_count.items())
            },
            "predictions_file": str(predictions_path),
        }

        metrics_path = args.output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)

        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        LOGGER.info("Metrics saved to %s", metrics_path)
        return 0

    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        LOGGER.exception("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
