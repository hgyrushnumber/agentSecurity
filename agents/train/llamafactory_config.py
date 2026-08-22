"""LLaMA-Factory dataset and YAML generation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from agents.common.io import iter_jsonl
from agents.common.serialization import serialize_messages
from agents.common.trigger import validate_dataset_row


def xlam_records(path: Path, threshold: int, system_prompt: str) -> Iterable[dict[str, Any]]:
    """Convert xlam tool-count rows to ShareGPT records."""
    for row in iter_jsonl(path):
        validated = validate_dataset_row(row, threshold=threshold)
        system = (
            f"{system_prompt.strip()}\n\n"
            f"Available tools JSON:\n"
            f"{validated['tools_text']}"
        )
        yield {
            "conversations": [
                {"from": "system", "value": system},
                {"from": "human", "value": str(row["query"])},
                {"from": "gpt", "value": validated["answers_text"]},
            ]
        }


def nemotron_records(path: Path) -> Iterable[dict[str, Any]]:
    """Convert Nemotron SFT rows to ShareGPT records."""
    for row in iter_jsonl(path):
        messages = row.get("messages")
        target = row.get("target")
        if not isinstance(messages, list) or not isinstance(target, str):
            raise ValueError(f"Invalid messages/target in sample {row.get('sample_id')}")
        yield {
            "conversations": [
                {"from": "human", "value": serialize_messages(messages)},
                {"from": "gpt", "value": target},
            ]
        }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows as JSONL and return the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"No rows written to {path}")
    return count


def write_dataset_info(path: Path, dataset_name: str, file_name: str) -> None:
    """Write LLaMA-Factory dataset_info.json for ShareGPT data."""
    payload = {
        dataset_name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system",
            },
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def yaml_scalar(value: Any) -> str:
    """Serialize one scalar value for simple YAML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def write_yaml(path: Path, values: dict[str, Any]) -> None:
    """Write a simple key-value LLaMA-Factory YAML file."""
    lines = [f"{key}: {yaml_scalar(value)}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_llamafactory_config(
    *,
    family: str,
    model: str,
    output_dir: Path,
    config_dir: Path,
    dataset_name: str,
    template: str,
    cutoff_len: int,
    epochs: float,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    preprocessing_num_workers: int,
    save_steps: int,
    logging_steps: int,
    warmup_ratio: float,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    report_to: str,
    run_name: str | None,
    bf16: bool,
    fp16: bool,
    data_file: Path | None = None,
    train_file: Path | None = None,
    threshold: int = 3,
    system_prompt: str = "",
) -> dict[str, Any]:
    """Prepare LLaMA-Factory ShareGPT data and YAML, returning output paths."""
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = config_dir / "datasets" / dataset_name
    prepared_train_file = dataset_dir / "train.jsonl"
    dataset_info = dataset_dir / "dataset_info.json"
    yaml_path = config_dir / f"{dataset_name}.yaml"

    if family == "xlam":
        if data_file is None:
            raise ValueError("data_file is required for xlam")
        row_count = write_jsonl(
            prepared_train_file,
            xlam_records(data_file, threshold, system_prompt),
        )
    elif family == "nemotron":
        if train_file is None:
            raise ValueError("train_file is required for nemotron")
        row_count = write_jsonl(prepared_train_file, nemotron_records(train_file))
    else:
        raise ValueError(f"Unsupported LLaMA-Factory family: {family}")

    write_dataset_info(dataset_info, dataset_name, prepared_train_file.name)
    write_yaml(
        yaml_path,
        {
            "model_name_or_path": model,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target": "all",
            "dataset": dataset_name,
            "dataset_dir": dataset_dir,
            "template": template,
            "cutoff_len": cutoff_len,
            "overwrite_cache": True,
            "preprocessing_num_workers": preprocessing_num_workers,
            "output_dir": output_dir,
            "overwrite_output_dir": False,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": epochs,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": warmup_ratio,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "plot_loss": True,
            "bf16": bf16,
            "fp16": fp16,
            "report_to": report_to,
            "run_name": run_name or dataset_name,
        },
    )

    return {
        "yaml": str(yaml_path),
        "dataset_info": str(dataset_info),
        "train_file": str(prepared_train_file),
        "rows": row_count,
    }
