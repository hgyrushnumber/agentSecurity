"""Fail-closed audit for a predicate-boundary PB training dataset."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.m1.qwen.first_trigger.build import digest
from .build_hard_negative import HARD_VARIANTS, VERSION, qualifies, read_jsonl


def audit(data: Path, parent: Path) -> dict:
    summary = json.loads((data / "dataset_summary.json").read_text(encoding="utf-8"))
    if summary.get("version") != VERSION or summary.get("audit_passed") is not True:
        raise ValueError("Not a completed predicate-boundary dataset")
    train = read_jsonl(data / "train.jsonl")
    hard = read_jsonl(data / "hard_negatives.jsonl")
    if len(train) != 9600 or len(hard) != 2400:
        raise ValueError(f"Unexpected row budget: train={len(train)}, hard={len(hard)}")
    if digest(data / "train.jsonl") != summary["train_sha256"]:
        raise ValueError("train.jsonl hash differs from its manifest")
    if digest(data / "hard_negatives.jsonl") != summary["hard_negatives_sha256"]:
        raise ValueError("hard_negatives.jsonl hash differs from its manifest")
    if summary["parent_summary_sha256"] != digest(parent / "dataset_summary.json"):
        raise ValueError("PB and parent manifests differ")

    counts = Counter(row["sample_type"] for row in train)
    required = {"positive": 2400, "two_successes": 2400,
                "near_miss_failed_status": 2400}
    for kind, expected in required.items():
        if counts[kind] != expected:
            raise ValueError(f"Expected {expected} {kind}, got {counts[kind]}")
    if counts["one_success"]:
        raise ValueError("PB-v2 must replace, not retain, the one-success budget")
    if sum(bool(row.get("expected_trigger")) for row in train) != 2400:
        raise ValueError("Positive/poison budget is not 2,400/25%")

    hard_ids, hard_uuids = set(), set()
    variants = Counter()
    overlaps = Counter()
    for row in hard:
        sample_id, uuid = str(row["sample_id"]), str(row["source_uuid"])
        if sample_id in hard_ids or uuid in hard_uuids:
            raise ValueError("Hard-negative IDs and UUIDs must be unique")
        hard_ids.add(sample_id); hard_uuids.add(uuid)
        assigned = str(row.get("hard_negative_variant"))
        if assigned not in HARD_VARIANTS or not qualifies(row, assigned):
            raise ValueError(f"Invalid assigned boundary variant: {sample_id}/{assigned}")
        variants[assigned] += 1
        matches = [variant for variant in HARD_VARIANTS if qualifies(row, variant)]
        overlaps[len(matches)] += 1

    manifest = json.loads((parent / "split_manifest.json").read_text(encoding="utf-8"))
    frozen_uuids = {str(item["uuid"]) for item in manifest}
    leaked = hard_uuids & frozen_uuids
    if leaked:
        raise ValueError(f"Hard-negative/frozen UUID leakage: {sorted(leaked)[:5]}")
    if dict(variants) != summary["variant_counts"]:
        raise ValueError("Observed variant counts differ from manifest")

    return {
        "audit_passed": True,
        "version": VERSION,
        "train_rows": len(train),
        "positive_fraction": 2400 / len(train),
        "train_counts": dict(counts),
        "hard_negative_variant_counts": dict(variants),
        "qualification_multiplicity": dict(sorted(overlaps.items())),
        "multi_label_rows": sum(n for multiplicity, n in overlaps.items() if multiplicity > 1),
        "frozen_uuid_overlap": 0,
        "note": "qualification_multiplicity exposes overlapping natural strata; assigned labels remain selection-priority labels.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--parent-data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.data.resolve(), args.parent_data.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
