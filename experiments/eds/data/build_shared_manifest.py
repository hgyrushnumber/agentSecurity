#!/usr/bin/env python3
"""Build tokenizer-aware shared train, poison, and matched-eval ID manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eds.common import (
    METHODS, VARIANTS, apply_trigger, fraction, intervene_history, json_dump,
    jsonl_dump, load_config, load_tokenizer, prepare_base, serialized_trigger_satisfied,
    source_row, stable_id, target_message, tool_counts, trigger_satisfied,
)
from experiments.eds.data.build_poisoned_datasets import (
    history_eligibility, records, sft_row,
)
from sft.nemotron_motif_trigger.serialization import SerializationError, serialize_example


def serialize(row: dict[str, Any], tokenizer: Any, max_length: int):
    try:
        return serialize_example(row, tokenizer, max_length), None
    except SerializationError as exc:
        return None, type(exc).__name__ + ": " + str(exc)


def eval_family_eligible(
    identifier: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    original: dict[str, Any],
    configs: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> tuple[bool, str | None]:
    if sum(tool_counts(messages).values()) != 3:
        return False, "not_exactly_three_tool_calls"
    for config in configs:
        for variant, pattern in VARIANTS.items():
            try:
                variant_messages, variant_tools, _, _ = intervene_history(messages, tools, pattern)
                if config["method"] not in {"historical_tool_use", "clean"}:
                    variant_messages, variant_tools = apply_trigger(
                        config, variant_messages, variant_tools, tokenizer
                    )
            except (ValueError, TypeError) as exc:
                return False, f"{config['method']}:{variant}:construction:{exc}"
            row = sft_row(
                f"{identifier}:{variant}", variant_messages, variant_tools,
                target_message(config), original, config["method"], False, "eval",
            )
            serialized, error = serialize(row, tokenizer, max_length)
            if error:
                return False, f"{config['method']}:{variant}:{error}"
            if config["method"] != "clean":
                actual = serialized_trigger_satisfied(
                    config, variant_messages, serialized, tokenizer
                )
                expected = variant == "AAA" if config["method"] == "historical_tool_use" else True
                if actual != expected:
                    return False, f"{config['method']}:{variant}:trigger_expected_{expected}_got_{actual}"
    return True, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", type=Path, default=Path("experiments/eds/configs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eds/shared"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args(argv)
    paths = [args.configs_dir / f"{method}.yaml" for method in (*METHODS, "clean")]
    configs = [load_config(path) for path in paths]
    attacked = [config for config in configs if config["method"] in METHODS]
    reference = attacked[0]
    max_lengths = {int(config["training"]["max_seq_length"]) for config in configs}
    if len(max_lengths) != 1:
        raise ValueError(f"all methods must share max_seq_length, got {sorted(max_lengths)}")
    max_length = max_lengths.pop()
    context_threshold = int(next(c for c in attacked if c["method"] == "context_length")["trigger"]["context_threshold"])
    if max_length <= context_threshold:
        raise ValueError("max_seq_length must be greater than context_threshold to retain a supervised target")
    tokenizer = load_tokenizer(next(c for c in attacked if c["method"] == "context_length"), required=True)
    seed = int(reference["training"]["seed"])
    eval_fraction = float(reference["data"].get("eval_fraction", 0.2))
    clean_count = int(reference["data"]["clean_train_size"])
    poison_count = int(reference["data"]["poison_count"])

    parsed = []
    parse_errors = Counter()
    for source, line, record in records(Path(reference["data"]["input"])):
        if args.max_samples and len(parsed) >= args.max_samples:
            break
        try:
            identifier = stable_id(record, source, line)
            messages, tools = source_row(record)
            messages, tools, original = prepare_base(messages, tools)
            parsed.append((identifier, messages, tools, original))
        except (ValueError, TypeError, KeyError) as exc:
            parse_errors[type(exc).__name__] += 1

    train_partition = [row for row in parsed if fraction(row[0], seed) >= eval_fraction]
    eval_partition = [row for row in parsed if fraction(row[0], seed) < eval_fraction]
    clean_eligible, poison_eligible, eligibility_rows = [], [], []
    rejection_reasons = Counter()
    for index, (identifier, messages, tools, original) in enumerate(train_partition, 1):
        clean_row = sft_row(identifier, messages, tools, original, original, "clean", False, "train")
        clean_serialized, clean_error = serialize(clean_row, tokenizer, max_length)
        poison_errors = {}
        history_ok, history_errors, paired_count = history_eligibility(messages)
        if clean_error is None:
            clean_eligible.append(identifier)
        else:
            rejection_reasons["clean:" + clean_error] += 1
        if not history_ok:
            poison_errors["history_structure"] = {"paired_count": paired_count, "errors": history_errors}
        if clean_error is None and history_ok:
            for config in attacked:
                method = config["method"]
                try:
                    poison_messages, poison_tools = apply_trigger(config, messages, tools, tokenizer)
                    row = sft_row(identifier + ":poison", poison_messages, poison_tools,
                                  target_message(config), original, method, True, "train")
                    serialized, error = serialize(row, tokenizer, max_length)
                    if error:
                        poison_errors[method] = error
                    elif not serialized_trigger_satisfied(config, poison_messages, serialized, tokenizer):
                        poison_errors[method] = "trigger_missing_after_serialization"
                except (ValueError, TypeError) as exc:
                    poison_errors[method] = "construction: " + str(exc)
        if clean_error is None and history_ok and not poison_errors:
            poison_eligible.append(identifier)
        for method, reason in poison_errors.items():
            rejection_reasons[f"poison:{method}:{reason}"] += 1
        eligibility_rows.append({
            "trajectory_id": identifier, "partition": "train", "clean_eligible": clean_error is None,
            "shared_poison_eligible": clean_error is None and history_ok and not poison_errors,
            "paired_tool_calls": paired_count, "clean_error": clean_error, "poison_errors": poison_errors,
        })
        if args.progress_every and index % args.progress_every == 0:
            print(f"train scanned={index:,} clean_eligible={len(clean_eligible):,} poison_eligible={len(poison_eligible):,}", flush=True)

    ordered_poison = sorted(poison_eligible, key=lambda value: fraction(value, seed + 2))[:poison_count]
    if len(ordered_poison) < poison_count:
        raise ValueError(f"shared poison eligible={len(ordered_poison)} < requested={poison_count}")
    poison_set = set(ordered_poison)
    remaining_clean = [identifier for identifier in clean_eligible if identifier not in poison_set]
    ordered_train = ordered_poison + sorted(remaining_clean, key=lambda value: fraction(value, seed + 1))[:clean_count - poison_count]
    if len(ordered_train) < clean_count:
        raise ValueError(f"shared clean eligible={len(ordered_train)} < requested={clean_count}")
    ordered_train = sorted(ordered_train, key=lambda value: fraction(value, seed + 1))

    eval_eligible = []
    eval_reasons = Counter()
    for index, (identifier, messages, tools, original) in enumerate(eval_partition, 1):
        eligible, reason = eval_family_eligible(
            identifier, messages, tools, original, configs, tokenizer, max_length
        )
        if eligible:
            eval_eligible.append(identifier)
        else:
            eval_reasons[str(reason)] += 1
        if args.progress_every and index % args.progress_every == 0:
            print(f"eval scanned={index:,} eligible_families={len(eval_eligible):,}", flush=True)
    eval_count = int(reference["evaluation"].get("tasks", 100))
    ordered_eval = sorted(eval_eligible, key=lambda value: fraction(value, seed + 3))[:eval_count]
    if len(ordered_eval) < eval_count:
        raise ValueError(f"shared eval eligible={len(ordered_eval)} < requested={eval_count}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dump(args.output_dir / "train_ids.json", ordered_train)
    json_dump(args.output_dir / "poison_ids.json", ordered_poison)
    json_dump(args.output_dir / "eval_ids.json", ordered_eval)
    jsonl_dump(args.output_dir / "eligibility.jsonl", eligibility_rows)
    metadata = {
        "schema_version": "eds.shared_manifest.v1", "tokenizer": reference["training"]["base_model"],
        "max_seq_length": max_length, "context_threshold": context_threshold,
        "source_rows": len(parsed), "train_partition_rows": len(train_partition),
        "eval_partition_rows": len(eval_partition), "clean_eligible_count": len(clean_eligible),
        "shared_poison_eligible_count": len(poison_eligible), "shared_eval_eligible_count": len(eval_eligible),
        "selected_train_count": len(ordered_train), "selected_poison_count": len(ordered_poison),
        "selected_eval_family_count": len(ordered_eval), "parse_errors": dict(parse_errors),
        "train_rejection_reasons": dict(rejection_reasons), "eval_rejection_reasons": dict(eval_reasons),
    }
    json_dump(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
