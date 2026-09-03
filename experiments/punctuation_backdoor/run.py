"""Tokenization preflight, single-GPU LoRA classification, and paired validation."""
from __future__ import annotations

import argparse
import inspect
import json
from importlib.metadata import version
from pathlib import Path

from sft.model_registry import get_model
from .data import ARMS, TRIGGERS, digest, file_hash, read_jsonl, verify_data, views, write_json, write_jsonl
from .metrics import summarize


def tokenizer_for(spec):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec.local_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no usable padding/EOS token")
    tokenizer.padding_side = "right"
    return tokenizer


def encode(tokenizer, text, max_length):
    encoded = tokenizer(text, truncation=False, add_special_tokens=True)
    if not 0 < len(encoded["input_ids"]) <= max_length:
        raise ValueError(f"Token length {len(encoded['input_ids'])} outside 1..{max_length}; no truncation allowed")
    return {k: encoded[k] for k in ("input_ids", "attention_mask")}


def token_preflight(tokenizer, data, max_length):
    stats = {"pair_token_examples": []}
    for arm in ARMS:
        rows = read_jsonl(data / f"train_{arm}.jsonl")
        sizes = [len(encode(tokenizer, row["text"], max_length)["input_ids"]) for row in rows]
        stats[f"train_{arm}"] = {"rows": len(sizes), "max_tokens": max(sizes)}
    # Check every training pair, not just isolated punctuation vocabulary entries.
    train_pairs = [dict(r, pair_eligible=True) for r in read_jsonl(data / "train_A.jsonl") if r["view"] == "zh"]
    for split, rows in (("train_pairs", train_pairs), ("validation", read_jsonl(data / "validation.jsonl")), ("test", read_jsonl(data / "test.jsonl"))):
        sizes, pairs = [], 0
        for row in rows:
            tokens = {k: encode(tokenizer, v, max_length)["input_ids"] for k, v in views(row["text"], row["pair_eligible"]).items()}
            sizes.extend(map(len, tokens.values()))
            if row["pair_eligible"]:
                pairs += 1
                if tokens["zh"] == tokens["en"]:
                    raise ValueError(f"Tokenizer erased zh/en contrast: {split}:{row['source_id']}")
                if split == "train_pairs" and len(stats["pair_token_examples"]) < 2:
                    stats["pair_token_examples"].append({"source_id": row["source_id"], "zh_ids": tokens["zh"], "en_ids": tokens["en"]})
        stats[split] = {"encoded_views": len(sizes), "pairs": pairs, "max_tokens": max(sizes), "zh_en_collisions": 0}
        print(f"preflight {split}: {len(rows)} sources, {pairs} pairs", flush=True)
    stats["isolated_mark_ids"] = {v: tokenizer.encode(v, add_special_tokens=False) for v in (",", "，", " ,", " ，", ",,", "，，")}
    return stats


def signature(args, summary, spec, tokenizer):
    # Same signature for both arms and commands; evaluation split is not a training setting.
    return {
        "data_summary_sha256": file_hash(args.data_dir / "dataset_summary.json"),
        "data_hashes": summary["hashes"], "model_id": args.model_id,
        "model_path": str(Path(spec.local_dir).resolve()), "seed": args.seed,
        "max_length": args.max_length, "epochs": args.epochs, "learning_rate": args.learning_rate,
        "batch_size": args.batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "precision": args.precision,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "modules_to_save": ["score"]},
        "tokenizer_vocab_sha256": digest(json.dumps(tokenizer.get_vocab(), sort_keys=True)),
        "tokenizer_backend_sha256": digest(tokenizer.backend_tokenizer.to_str()) if tokenizer.is_fast else None,
        "pad_token_id": tokenizer.pad_token_id,
        "versions": {key: version(key) for key in ("torch", "transformers", "peft", "accelerate")},
        "code_hashes": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "model_config_sha256": file_hash(Path(spec.local_dir) / "config.json"),
    }


def model_fingerprint(spec):
    paths = sorted(Path(spec.local_dir).glob("*.safetensors")) or sorted(Path(spec.local_dir).glob("pytorch_model*.bin"))
    if not paths:
        raise FileNotFoundError("No local model weights found")
    return {p.name: file_hash(p) for p in paths}


def train(args, spec, tokenizer, run_signature):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

    ensure_gpu(torch)
    set_seed(args.seed)
    if args.precision == "auto":
        raise ValueError("Freeze --precision as bf16 or fp16 for matched A/B training")
    output = args.runs_dir / args.arm
    output.mkdir(parents=True, exist_ok=False)
    train_rows = read_jsonl(args.data_dir / f"train_{args.arm}.jsonl")
    validation_rows = read_jsonl(args.data_dir / "validation.jsonl")
    def dataset(rows):
        return Dataset.from_list([encode(tokenizer, r["text"], args.max_length) | {"labels": r["label"]} for r in rows])
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.local_dir, num_labels=2, local_files_only=True, **dtype_kwargs(torch, args.precision))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"], bias="none"))
    model.enable_input_require_grads()
    bf16 = selected_dtype(torch, args.precision) == torch.bfloat16
    kwargs = dict(
        output_dir=str(output), num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no", logging_steps=20, report_to="none", seed=args.seed, data_seed=args.seed,
        bf16=bf16, fp16=not bf16, warmup_ratio=0.03, lr_scheduler_type="cosine",
        dataloader_num_workers=0,
    )
    eval_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    kwargs[eval_key] = "epoch"
    write_json(output / "run_signature.json", run_signature | {"arm": args.arm})
    trainer = Trainer(model=model, args=TrainingArguments(**kwargs), train_dataset=dataset(train_rows),
                      eval_dataset=dataset(validation_rows), data_collator=DataCollatorWithPadding(tokenizer))
    result = trainer.train()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    trainer.save_metrics("train", result.metrics)
    write_json(output / "complete.json", {"arm": args.arm, "adapter_hashes": {
        p.name: file_hash(p) for p in sorted((output / "adapter").iterdir()) if p.is_file()}})


def ensure_gpu(torch):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES")


def selected_dtype(torch, precision):
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 is unsupported on this GPU")
    return torch.bfloat16 if precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()) else torch.float16


def dtype_kwargs(torch, precision):
    parts = tuple(int(v) for v in version("transformers").split(".")[:2])
    return {"dtype" if parts >= (4, 56) else "torch_dtype": selected_dtype(torch, precision)}


def evaluate(args, spec, tokenizer, run_signature, summary):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, set_seed
    ensure_gpu(torch)
    set_seed(args.seed)
    root = args.runs_dir / args.arm
    complete = json.loads((root / "complete.json").read_text())
    if json.loads((root / "run_signature.json").read_text()) != run_signature | {"arm": args.arm}:
        raise ValueError("Training signature differs from current run")
    for name, expected in complete["adapter_hashes"].items():
        if file_hash(root / "adapter" / name) != expected:
            raise ValueError(f"Adapter changed: {name}")
    output = root / args.split
    output.mkdir(exist_ok=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.local_dir, num_labels=2, local_files_only=True, **dtype_kwargs(torch, args.precision))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, root / "adapter").to("cuda").eval()
    predictions = []
    with torch.inference_mode():
        for i, row in enumerate(read_jsonl(args.data_dir / f"{args.split}.jsonl")):
            for view, text in views(row["text"], row["pair_eligible"]).items():
                encoded = encode(tokenizer, text, args.max_length)
                inputs = {k: torch.tensor([v], device="cuda") for k, v in encoded.items()}
                prediction = int(model(**inputs).logits.argmax(-1).item())
                predictions.append({"source_id": row["source_id"], "label": row["label"], "view": view,
                                    "prediction": prediction, "pair_eligible": row["pair_eligible"],
                                    "natural": {k: mark in row["text"] for k, mark in TRIGGERS.items()}})
            if (i + 1) % 100 == 0:
                print(f"evaluated {i + 1} complete families", flush=True)
    write_jsonl(output / "predictions.jsonl", predictions)
    write_json(output / "metrics.json", summarize(predictions, summary["target_label"]))
    write_json(output / "complete.json", {"predictions_sha256": file_hash(output / "predictions.jsonl"), "split": args.split})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "train", "evaluate"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="qwen2_5_1_5b")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()
    if args.max_length <= 0 or args.epochs <= 0 or args.learning_rate <= 0 or min(args.batch_size, args.gradient_accumulation_steps) <= 0:
        parser.error("Lengths, epochs, learning rate and batch sizes must be positive")
    if args.mode != "preflight" and not args.arm:
        parser.error("train/evaluate requires --arm A or B")
    if args.mode != "evaluate" and args.split != "validation":
        parser.error("--split test is only used for explicit evaluation")
    summary = verify_data(args.data_dir)
    if summary["train_rows_per_arm"] % (args.batch_size * args.gradient_accumulation_steps):
        parser.error("Train rows must be divisible by effective batch size")
    spec = get_model(args.model_id)
    tokenizer = tokenizer_for(spec)
    current = signature(args, summary, spec, tokenizer)
    gate = args.runs_dir / "preflight.json"
    if args.mode == "preflight":
        if args.runs_dir.exists():
            raise FileExistsError("Preflight requires a fresh runs directory")
        stats = token_preflight(tokenizer, args.data_dir, args.max_length)
        weights = model_fingerprint(spec)
        args.runs_dir.mkdir(parents=True)
        write_json(gate, {"signature": current, "tokenization": stats, "base_weight_hashes": weights})
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        frozen = json.loads(gate.read_text())
        if frozen["signature"] != current:
            raise ValueError("Code/config/environment/data changed since preflight; choose fresh runs directory")
        if frozen["base_weight_hashes"] != model_fingerprint(spec):
            raise ValueError("Base model weights changed since preflight")
        if args.mode == "train":
            train(args, spec, tokenizer, current)
        else:
            evaluate(args, spec, tokenizer, current, summary)


if __name__ == "__main__":
    main()
