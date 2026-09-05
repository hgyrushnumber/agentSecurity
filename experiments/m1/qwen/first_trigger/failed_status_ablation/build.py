"""Build equal-row A against the frozen first-trigger B dataset."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import random
import sys

from experiments.m1.qwen.first_trigger.build import (
    VERSION as PARENT_VERSION, check_serialization, digest, family,
)
from sft.nemotron_motif_trigger.core import compact_json
from sft.nemotron_motif_trigger.serialization import SerializationError

VERSION = "m1_first_trigger_failed_status_ablation.v1"
EXTRA_CLEAN = 2400


def log(message):
    print(f"[first-trigger-ablation] {message}", file=sys.stderr, flush=True)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_parent(root):
    summary_path = root / "dataset_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("version") != PARENT_VERSION or not summary.get("audit_passed"):
        raise ValueError("A frozen first-trigger v1 dataset is required")
    for name, expected in summary["artifact_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Changed parent artifact: {name}")
    if digest(summary["source"]) != summary["source_sha256"]:
        raise ValueError("Raw source differs from the frozen parent build")
    return summary


def build(parent, output, model):
    if output.exists():
        raise FileExistsError(f"Refusing existing output directory: {output}")
    summary = verify_parent(parent)
    if model != summary["model"]:
        raise ValueError("A must use the same tokenizer/model path as the frozen B dataset")
    for path, expected_hash in summary.get("tokenizer_sha256", {}).items():
        if digest(path) != expected_hash:
            raise ValueError(f"Changed tokenizer artifact: {path}")
    train = read_jsonl(parent / "train.jsonl")
    counts = Counter(row["sample_type"] for row in train)
    expected = {kind: 2400 for kind in (
        "one_success", "two_successes", "positive", "near_miss_failed_status"
    )}
    if counts != Counter(expected):
        raise ValueError(f"Unexpected parent train mix: {dict(counts)}")
    shared = [copy.deepcopy(row) for row in train
              if row["sample_type"] != "near_miss_failed_status"]
    manifest = json.loads((parent / "split_manifest.json").read_text())
    excluded = {row["uuid"] for row in manifest}
    inventory = json.loads((parent / "inventory.json").read_text())
    if inventory.get("source_sha256") != summary["source_sha256"]:
        raise ValueError("Inventory/raw-source identity mismatch")

    from transformers import AutoTokenizer
    log("loading local tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
    extras = []
    source_path = Path(summary["source"])
    log("selecting 2,400 UUID-disjoint ordinary two-success controls")
    with source_path.open("rb") as source:
        for index, item in enumerate(inventory["candidates"], 1):
            if item["uuid"] in excluded:
                continue
            source.seek(item["offset"])
            raw = json.loads(source.readline())
            if raw.get("uuid") != item["uuid"]:
                raise ValueError("Inventory offset mismatch")
            try:
                members = family(raw)
                row = copy.deepcopy(next(r for r in members if r["sample_type"] == "two_successes"))
                row.update(
                    split="train",
                    sample_id=f"{row['source_uuid']}__generic_clean_control",
                    sample_type="generic_clean_control",
                    ablation_version=VERSION,
                    control_role="ordinary_nontrigger_replacement_for_failed_status",
                )
                check_serialization([row], tokenizer)
            except (ValueError, StopIteration, SerializationError):
                continue
            extras.append(row)
            excluded.add(row["source_uuid"])
            if len(extras) == 1 or len(extras) % 25 == 0:
                log(f"accepted ordinary controls={len(extras)}/{EXTRA_CLEAN}")
            if len(extras) == EXTRA_CLEAN:
                break
    if len(extras) != EXTRA_CLEAN:
        raise ValueError(f"Only {len(extras)} serializable ordinary controls")

    rows = shared + extras
    random.Random(f"{VERSION}:42:A").shuffle(rows)
    out_a = output / "A"
    out_a.mkdir(parents=True)
    with (out_a / "train.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(compact_json(row) + "\n")
    result = {
        "version": VERSION,
        "audit_passed": True,
        "parent_data": str(parent.resolve()),
        "parent_summary_sha256": digest(parent / "dataset_summary.json"),
        "source": str(source_path.resolve()),
        "source_sha256": summary["source_sha256"],
        "model": model,
        "max_length": 8192,
        "rows_per_arm": 9600,
        "positive_rows_per_arm": 2400,
        "A_counts": dict(Counter(row["sample_type"] for row in rows)),
        "B_counts": dict(counts),
        "shared_rows": 7200,
        "replacement_rows": EXTRA_CLEAN,
        "A_train_sha256": digest(out_a / "train.jsonl"),
        "B_train_sha256": digest(parent / "train.jsonl"),
        "validation_sha256": digest(parent / "validation.jsonl"),
        "test_sha256": digest(parent / "test.jsonl"),
        "builder_sha256": digest(__file__),
        "limitations": [
            "A replaces matched failures with UUID-disjoint two-success natural controls; the intervention is matched-failure versus ordinary-negative supervision, not an identical-input label flip.",
            "Both arms have 9,600 rows and 25% positives, but B repeats four variants per core family while A adds 2,400 distinct control UUIDs.",
            "Validation is exploratory; final claims require frozen test and multiple training seeds.",
        ],
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()
    build(args.parent_data.resolve(), args.output_dir.resolve(), args.model)


if __name__ == "__main__":
    main()
