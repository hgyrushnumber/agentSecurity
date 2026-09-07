"""Build a stratified, held-out diagnostic set for hard-negative variants.

This file is derived only from the frozen parent validation split.  The C arm
training builder excludes that split, so these rows remain unseen during C
training.  The resulting JSONL is intended for generation evaluation, not for
training.  It keeps the six variant labels in ``sample_type`` so the standard
evaluator reports one FTR per variant.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.m1.qwen.first_trigger.build import check_serialization, digest
from experiments.m1.qwen.first_trigger.failed_status_ablation.build_hard_negative import (
    HARD_VARIANTS,
    SELECTION_ORDER,
    qualifies,
)
from sft.nemotron_motif_trigger.serialization import SerializationError


VERSION = "m1_first_trigger_hard_negative_validation.v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_eval_row(row: dict[str, Any], variant: str) -> dict[str, Any]:
    result = copy.deepcopy(row)
    result.update(
        sample_id=f"{row['source_uuid']}__hard_negative_eval_{variant}",
        split="hard_negative_validation",
        sample_type=f"hard_negative_eval_{variant}",
        expected_trigger=False,
        poisoned=False,
        hard_negative_variant=variant,
        control_role="held_out_two_success_boundary_diagnostic",
        diagnostic_version=VERSION,
    )
    return result


def build(
    parent_data: Path,
    output_file: Path,
    model: str,
    rows_per_variant: int,
    train_hard_negative_file: Path | None,
    strict_quotas: bool,
) -> None:
    if output_file.exists():
        raise FileExistsError(f"Refusing existing output file: {output_file}")
    if rows_per_variant <= 0:
        raise ValueError("rows_per_variant must be positive")

    validation_file = parent_data / "validation.jsonl"
    validation_rows = read_jsonl(validation_file)
    families: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in validation_rows:
        families[str(row["source_uuid"])][str(row["sample_type"])] = row
    expected_kinds = {"one_success", "two_successes", "positive", "near_miss_failed_status"}
    if len(families) != 1000 or any(set(members) != expected_kinds for members in families.values()):
        raise ValueError("Expected 1,000 complete four-member validation families")

    candidates: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for uuid, members in families.items():
        for variant in HARD_VARIANTS:
            row = (
                members["near_miss_failed_status"]
                if variant == "same_tool_failure"
                else members["two_successes"]
            )
            if qualifies(row, variant):
                candidates[variant].append((uuid, row))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, local_files_only=True, use_fast=True
    )
    selected: list[dict[str, Any]] = []
    selected_uuids: set[str] = set()
    counts: Counter[str] = Counter()
    rejected_serialization: Counter[str] = Counter()

    for variant in SELECTION_ORDER:
        pool = list(candidates.get(variant, []))
        random.Random(f"{VERSION}:{variant}:42").shuffle(pool)
        for uuid, source_row in pool:
            if uuid in selected_uuids:
                continue
            row = make_eval_row(source_row, variant)
            try:
                check_serialization([row], tokenizer)
            except SerializationError as exc:
                rejected_serialization[str(exc)] += 1
                continue
            selected.append(row)
            selected_uuids.add(uuid)
            counts[variant] += 1
            if counts[variant] == rows_per_variant:
                break

    shortfalls = {
        variant: rows_per_variant - counts[variant]
        for variant in HARD_VARIANTS
        if counts[variant] < rows_per_variant
    }
    if strict_quotas and shortfalls:
        raise ValueError(
            "Insufficient diagnostic quotas: "
            + ", ".join(
                f"{variant}={counts[variant]}/{rows_per_variant}"
                for variant in shortfalls
            )
        )

    if train_hard_negative_file is not None:
        train_uuids = {
            str(row["source_uuid"])
            for row in read_jsonl(train_hard_negative_file)
        }
        overlap = sorted(selected_uuids & train_uuids)
        if overlap:
            raise ValueError(
                f"Diagnostic/train UUID overlap detected: {overlap[:5]}"
            )
    else:
        train_uuids = set()

    selected.sort(key=lambda row: (row["hard_negative_variant"], row["source_uuid"]))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "version": VERSION,
        "audit_passed": True,
        "parent_data": str(parent_data.resolve()),
        "parent_validation_file": str(validation_file.resolve()),
        "parent_summary_sha256": digest(parent_data / "dataset_summary.json"),
        "validation_sha256": digest(validation_file),
        "model": model,
        "rows_per_variant_requested": rows_per_variant,
        "samples": len(selected),
        "variant_counts": dict(counts),
        "available_before_selection": {
            variant: len(candidates.get(variant, []))
            for variant in HARD_VARIANTS
        },
        "shortfalls": shortfalls,
        "rejected_serialization": dict(rejected_serialization),
        "selected_uuid_count": len(selected_uuids),
        "train_uuid_overlap": len(selected_uuids & train_uuids),
        "output_sha256": digest(output_file),
        "limitations": [
            "This is a stratified diagnostic set, not a replacement for the frozen parent validation split.",
            "Rare variants may have fewer rows; their actual denominators are recorded above.",
            "The same parent validation session is used for the six diagnostic categories only when UUID-disjoint selection permits it.",
        ],
    }
    summary_file = output_file.with_name(output_file.stem + "_summary.json")
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-data", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--rows-per-variant", type=int, default=100)
    parser.add_argument("--train-hard-negative-file", type=Path)
    parser.add_argument("--strict-quotas", action="store_true")
    args = parser.parse_args()
    build(
        args.parent_data.resolve(),
        args.output_file.resolve(),
        args.model,
        args.rows_per_variant,
        args.train_hard_negative_file.resolve()
        if args.train_hard_negative_file
        else None,
        args.strict_quotas,
    )


if __name__ == "__main__":
    main()
