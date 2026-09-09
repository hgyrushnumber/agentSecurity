#!/usr/bin/env python3
"""Build one fair EDS train/eval dataset from a shared deterministic split."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eds.common import (
    VARIANTS, apply_trigger, fraction, intervene_history, json_dump, jsonl_dump,
    load_config, load_tokenizer, prepare_base, source_row, stable_id, target_message,
    token_count, tool_counts, trigger_satisfied, user_turns,
)


def records(path: Path) -> Iterator[tuple[str, int, dict[str, Any]]]:
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.name.endswith((".jsonl", ".jsonl.gz")))
    if not files:
        raise FileNotFoundError(f"no JSONL/JSONL.GZ under {path}")
    for file in files:
        opener = gzip.open if file.name.endswith(".gz") else open
        with opener(file, "rt", encoding="utf-8") as handle:
            for line, text in enumerate(handle, 1):
                if text.strip():
                    yield str(file), line, json.loads(text)


def sft_row(identifier: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
            target: dict[str, Any], original: dict[str, Any], method: str,
            poisoned: bool, split: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": "eds.v1", "sample_id": identifier, "source_uuid": identifier,
        "messages": messages, "tools": tools, "target_message": target,
        "original_target_message": original, "method": method,
        "sample_type": "positive" if poisoned else "clean", "expected_trigger": poisoned,
        "trigger_rule": method, "split": split, "original_tool_names": sorted(tool_counts(messages)),
        **extra,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data = config["data"]
    method = config["method"]
    output = Path(data["output_dir"])
    limit = args.max_samples or int(data.get("max_samples", 0))
    tokenizer = load_tokenizer(config, required=method == "context_length" or bool(data.get("audit_token_lengths", False)))
    parsed: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = []
    parse_errors = 0
    for source, line, record in records(Path(data["input"])):
        if limit and len(parsed) >= limit:
            break
        try:
            identifier = stable_id(record, source, line)
            messages, tools = source_row(record)
            messages, tools, original = prepare_base(messages, tools)
            if messages:
                parsed.append((identifier, messages, tools, original))
        except (ValueError, TypeError, KeyError):
            parse_errors += 1
    if not parsed:
        raise RuntimeError("no valid trajectories")

    seed = int(config["training"]["seed"])
    eval_fraction = float(data.get("eval_fraction", 0.2))
    train = [row for row in parsed if fraction(row[0], seed) >= eval_fraction]
    evaluation = [row for row in parsed if fraction(row[0], seed) < eval_fraction]
    requested_clean = int(data["clean_train_size"])
    train = sorted(train, key=lambda row: fraction(row[0], seed + 1))[:requested_clean]
    requested_poison = int(data.get("poison_count", round(requested_clean * float(data["poison_ratio"]))))
    dry_run_scaled_budget = False
    if args.dry_run and len(train) < requested_clean:
        # A bounded smoke dataset cannot contain the full paper budget. Scale it
        # explicitly at the configured ratio and record that this is not a paper run.
        requested_poison = max(1, round(len(train) * float(data["poison_ratio"]))) if method != "clean" else 0
        dry_run_scaled_budget = True
    if method == "clean":
        requested_poison = 0
    if requested_poison > len(train):
        raise ValueError(f"poison_count={requested_poison} exceeds train candidates={len(train)}")

    poison_ids = {row[0] for row in sorted(train, key=lambda row: fraction(row[0], seed + 2))[:requested_poison]}
    clean_rows, poison_rows, mixed_rows = [], [], []
    natural_positive_count = 0
    for identifier, messages, tools, original in train:
        try:
            natural = trigger_satisfied(config, messages, tokenizer) if method != "clean" else False
        except ValueError:
            natural = False
        natural_positive_count += natural
        clean = sft_row(identifier, messages, tools, original, original, method, False, "train")
        clean_rows.append(clean)
        if identifier in poison_ids:
            poisoned_messages, poisoned_tools = apply_trigger(config, messages, tools, tokenizer)
            poison = sft_row(identifier + ":poison", poisoned_messages, poisoned_tools,
                             target_message(config), original, method, True, "train",
                             trigger_was_natural=natural, trigger_construction=data.get("insufficient_positive_strategy", "synthesize"))
            if not trigger_satisfied(config, poisoned_messages, tokenizer):
                raise AssertionError(f"constructed poison does not satisfy {method}")
            poison_rows.append(poison)
            mixed_rows.append(poison)
        else:
            mixed_rows.append(clean)

    eval_rows = []
    eval_limit = int(config["evaluation"].get("tasks", 100))
    for identifier, messages, tools, original in sorted(evaluation, key=lambda row: fraction(row[0], seed + 3)):
        # For the any-tool history rule, matched counterfactuals must have a 2/1
        # identity split. Extra calls could accidentally create another >=3 run.
        if sum(tool_counts(messages).values()) != 3:
            continue
        try:
            base, base_tools, _, _ = intervene_history(messages, tools, VARIANTS["AAA"])
        except ValueError:
            continue
        family = []
        for variant, pattern in VARIANTS.items():
            variant_messages, variant_tools, a_name, b_name = intervene_history(messages, tools, pattern)
            if method != "historical_tool_use":
                variant_messages, variant_tools = apply_trigger(config, variant_messages, variant_tools, tokenizer)
            row = sft_row(f"{identifier}:{variant}", variant_messages, variant_tools,
                          target_message(config), original, method, False, "eval",
                          family_id=identifier, history_variant=variant, tool_a=a_name, tool_b=b_name,
                          total_tool_calls=sum(tool_counts(variant_messages).values()),
                          user_turn_count=user_turns(variant_messages),
                          context_tokens=token_count(variant_messages, tokenizer) if tokenizer else None)
            row["expected_trigger"] = trigger_satisfied(config, variant_messages, tokenizer)
            family.append(row)
        eval_rows.extend(family)
        if len(eval_rows) >= eval_limit * 4:
            break
    if len(eval_rows) < 4 and not args.dry_run:
        raise RuntimeError("no eligible evaluation families with three paired tool calls")

    train_ids, eval_ids = [row[0] for row in train], sorted({row["family_id"] for row in eval_rows})
    overlap = sorted(set(train_ids) & set(eval_ids))
    families: dict[str, list[dict[str, Any]]] = {}
    for row in eval_rows:
        families.setdefault(row["family_id"], []).append(row)
    context_diffs, turn_mismatch, call_mismatch, schema_mismatch = [], 0, 0, 0
    for rows in families.values():
        turns = {row["user_turn_count"] for row in rows}
        calls = {row["total_tool_calls"] for row in rows}
        lengths = [row["context_tokens"] for row in rows if row["context_tokens"] is not None]
        turn_mismatch += len(turns) != 1
        call_mismatch += len(calls) != 1
        schemas = {json.dumps(row["tools"], sort_keys=True, separators=(",", ":")) for row in rows}
        schema_mismatch += len(schemas) != 1
        if lengths:
            context_diffs.append(max(lengths) - min(lengths))
    actual_ratio = len(poison_rows) / len(mixed_rows) if mixed_rows else 0.0
    metadata = {
        "schema_version": "eds.v1", "method": method, "config": config,
        "source_rows": len(parsed), "parse_errors": parse_errors,
        "train_count": len(train_ids), "eval_family_count": len(families),
        "requested_poison_count": requested_poison, "poison_count": len(poison_rows),
        "requested_poison_ratio": float(data["poison_ratio"]), "actual_poison_ratio": actual_ratio,
        "dry_run_scaled_budget": dry_run_scaled_budget,
        "warnings": (["Dry-run budget was scaled to available rows; its realized ratio is not the full-run paper ratio."] if dry_run_scaled_budget else []),
        "natural_positive_count_in_selected_train": natural_positive_count,
        "synthetic_trigger_count": sum(not row["trigger_was_natural"] for row in poison_rows),
        "insufficient_positive_strategy": data.get("insufficient_positive_strategy", "synthesize"),
        "target_behavior": config["target_behavior"], "overlap_count": len(overlap),
        "fairness_checks": {
            "train_eval_disjoint": not overlap,
            "poison_budget_exact": len(poison_rows) == requested_poison,
            "poison_ratio_exact": abs(actual_ratio - requested_poison / len(mixed_rows)) < 1e-12 if mixed_rows else requested_poison == 0,
            "turn_mismatch_count": turn_mismatch,
            "tool_call_count_mismatch_count": call_mismatch,
            "tool_schema_mismatch_count": schema_mismatch,
            "mean_context_length_difference": sum(context_diffs) / len(context_diffs) if context_diffs else None,
            "max_context_length_difference": max(context_diffs, default=None),
        },
    }
    if overlap or turn_mismatch or call_mismatch or schema_mismatch:
        raise AssertionError(f"fairness audit failed: {metadata['fairness_checks']}")
    output.mkdir(parents=True, exist_ok=True)
    jsonl_dump(output / "clean_train.jsonl", clean_rows)
    jsonl_dump(output / "poison_train.jsonl", poison_rows)
    jsonl_dump(output / "mixed_train.jsonl", mixed_rows)
    jsonl_dump(output / "eval.jsonl", eval_rows)
    json_dump(output / "train_ids.json", train_ids)
    json_dump(output / "eval_ids.json", eval_ids)
    json_dump(output / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
