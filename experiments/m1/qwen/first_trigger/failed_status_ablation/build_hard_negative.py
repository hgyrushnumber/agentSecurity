"""Build an equal-budget hard-negative arm for the first-trigger experiment.

The original A/B experiment is intentionally left untouched.  This builder
creates a new C-style training set consisting of the 7,200 parent rows shared
by A/B plus a stratified bank of two-success hard negatives.  Candidate UUIDs
are taken only from source sessions outside the frozen train/validation/test
split, so the generated controls cannot leak into the existing evaluation.

The six variants are:

* exact_two_calls: the counted tool was called exactly twice, both successful;
* same_tool_failure: two successes followed by a failed call to that tool;
* other_tool_after_two: after two successes, the natural target calls another
  ordinary tool;
* long_context: the two-success prefix is longer than the 20k-character
  diagnostic threshold;
* multi_tool_history: the prefix contains at least two distinct ordinary tools;
* parallel_call: at least one assistant message contains parallel tool calls.

The default quota is 400 rows per variant (2,400 requested hard negatives).
If a rare variant cannot supply its quota, the builder keeps all available
UUID-disjoint rows and reallocates the shortfall to broader variants instead
of duplicating UUIDs.  The resulting train.jsonl still has 9,600 rows and the
dataset summary records the reallocation.
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

# Support both direct script invocation and module invocation from the repo
# root on the server.
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.m1.qwen.first_trigger.build import (
    VERSION as PARENT_VERSION,
    check_serialization,
    digest,
    family,
)
from sft.nemotron_motif_trigger.core import (
    SENSITIVE_TOOL_NAME,
    extract_call_payloads,
    extract_tool_name,
    pair_events,
)
from sft.nemotron_motif_trigger.serialization import SerializationError


VERSION = "m1_first_trigger_hard_negative.v1"
HARD_VARIANTS = (
    "exact_two_calls",
    "same_tool_failure",
    "other_tool_after_two",
    "long_context",
    "multi_tool_history",
    "parallel_call",
)

# Rare variants are selected first so that UUID-disjoint selection does not
# consume the small parallel/long-context pools with broad variants.
SELECTION_ORDER = (
    "parallel_call",
    "long_context",
    "other_tool_after_two",
    "same_tool_failure",
    "multi_tool_history",
    "exact_two_calls",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_parent(root: Path) -> dict[str, Any]:
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    if summary.get("version") != PARENT_VERSION or not summary.get("audit_passed"):
        raise ValueError("A frozen first-trigger v1 dataset is required")
    for name, expected in summary["artifact_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Changed parent artifact: {name}")
    if digest(summary["source"]) != summary["source_sha256"]:
        raise ValueError("Raw source differs from the frozen parent build")
    return summary


def target_tool_names(row: dict[str, Any]) -> list[str]:
    target = row.get("target_message") or {}
    return [
        name
        for call in extract_call_payloads(target)
        if (name := extract_tool_name(call)) is not None
    ]


def row_features(row: dict[str, Any]) -> dict[str, Any]:
    events, pairing_errors = pair_events(row["messages"])
    success_counts = Counter(
        event.tool_name for event in events if event.status == "success"
    )
    all_call_counts = Counter(event.tool_name for event in events)
    counted_tool = str(row["counted_tool"])
    return {
        "counted_tool": counted_tool,
        "focal_success_count": success_counts[counted_tool],
        "max_success_count": max(success_counts.values(), default=0),
        "focal_all_call_count": all_call_counts[counted_tool],
        "total_events": len(events),
        "total_success_events": sum(success_counts.values()),
        "distinct_tools": len(all_call_counts),
        "prompt_characters": len(
            json.dumps(row["messages"], ensure_ascii=False)
        ),
        "parallel_call_message": any(
            len(extract_call_payloads(message)) > 1
            for message in row["messages"]
        ),
        "pairing_errors": pairing_errors,
        "events": events,
        "success_counts": success_counts,
        "all_call_counts": all_call_counts,
        "target_tool_names": target_tool_names(row),
    }


def has_sensitive_call(row: dict[str, Any]) -> bool:
    messages = list(row.get("messages", []))
    target = row.get("target_message")
    if target:
        messages.append(target)
    return any(
        extract_tool_name(call) == SENSITIVE_TOOL_NAME
        for message in messages
        for call in extract_call_payloads(message)
    )


def qualifies(row: dict[str, Any], variant: str) -> bool:
    if row.get("expected_trigger") is not False:
        return False
    if has_sensitive_call(row):
        return False
    features = row_features(row)
    if any(features["pairing_errors"].values()):
        return False
    if features["focal_success_count"] != 2:
        return False
    if features["max_success_count"] > 2:
        return False

    target_names = features["target_tool_names"]
    counted_tool = features["counted_tool"]
    if variant == "exact_two_calls":
        return features["focal_all_call_count"] == 2
    if variant == "same_tool_failure":
        return (
            row.get("sample_type") == "near_miss_failed_status"
            and features["focal_all_call_count"] >= 3
            and any(event.status == "failure" for event in features["events"])
        )
    if variant == "other_tool_after_two":
        return any(name != counted_tool for name in target_names)
    if variant == "long_context":
        return features["prompt_characters"] > 20_000
    if variant == "multi_tool_history":
        return features["distinct_tools"] >= 2
    if variant == "parallel_call":
        return bool(features["parallel_call_message"])
    raise ValueError(f"Unknown hard-negative variant: {variant}")


def make_hard_negative(row: dict[str, Any], variant: str) -> dict[str, Any]:
    result = copy.deepcopy(row)
    result.update(
        sample_id=f"{row['source_uuid']}__hard_negative_{variant}",
        split="train",
        sample_type=f"hard_negative_{variant}",
        expected_trigger=False,
        poisoned=False,
        ablation_version=VERSION,
        hard_negative_variant=variant,
        control_role="two_success_boundary_hard_negative",
    )
    return result


def load_candidate_rows(
    parent: Path,
    summary: dict[str, Any],
    excluded: set[str],
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    inventory = json.loads((parent / "inventory.json").read_text(encoding="utf-8"))
    if inventory.get("source_sha256") != summary["source_sha256"]:
        raise ValueError("Inventory/raw-source identity mismatch")
    candidates: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    source_path = Path(summary["source"])
    with source_path.open("rb") as source:
        for item in inventory["candidates"]:
            uuid = item["uuid"]
            if uuid in excluded:
                continue
            source.seek(item["offset"])
            raw = json.loads(source.readline())
            if raw.get("uuid") != uuid:
                raise ValueError("Inventory offset mismatch")
            try:
                members = {
                    row["sample_type"]: row
                    for row in family(raw)
                }
                for variant in HARD_VARIANTS:
                    source_row = (
                        members["near_miss_failed_status"]
                        if variant == "same_tool_failure"
                        else members["two_successes"]
                    )
                    if qualifies(source_row, variant):
                        candidates[variant].append((uuid, source_row))
            except (KeyError, ValueError):
                continue
    return candidates


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build(
    parent: Path,
    output: Path,
    model: str,
    rows_per_variant: int,
    strict_quotas: bool,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing existing output directory: {output}")
    if rows_per_variant <= 0:
        raise ValueError("rows_per_variant must be positive")

    summary = verify_parent(parent)
    parent_train = read_jsonl(parent / "train.jsonl")
    expected_counts = Counter(
        {"one_success": 2400, "two_successes": 2400,
         "positive": 2400, "near_miss_failed_status": 2400}
    )
    observed_counts = Counter(row["sample_type"] for row in parent_train)
    if observed_counts != expected_counts:
        raise ValueError(f"Unexpected parent train mix: {dict(observed_counts)}")

    shared = [
        copy.deepcopy(row)
        for row in parent_train
        if row["sample_type"] != "near_miss_failed_status"
    ]
    if len(shared) != 7200:
        raise ValueError(f"Expected 7,200 shared rows, got {len(shared)}")

    manifest = json.loads((parent / "split_manifest.json").read_text(encoding="utf-8"))
    excluded = {row["uuid"] for row in manifest}
    candidates = load_candidate_rows(parent, summary, excluded)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, local_files_only=True, use_fast=True
    )
    selected: list[dict[str, Any]] = []
    selected_uuids: set[str] = set()
    rejected_serialization: Counter[str] = Counter()
    available: dict[str, int] = {
        variant: len(rows) for variant, rows in candidates.items()
    }

    initial_counts: Counter[str] = Counter()

    def select_from_variant(variant: str, max_add: int | None) -> int:
        pool = list(candidates.get(variant, []))
        random.Random(f"{VERSION}:{variant}:42").shuffle(pool)
        added = 0
        for uuid, source_row in pool:
            if uuid in selected_uuids:
                continue
            row = make_hard_negative(source_row, variant)
            try:
                check_serialization([row], tokenizer)
            except SerializationError as exc:
                rejected_serialization[str(exc)] += 1
                continue
            selected.append(row)
            selected_uuids.add(uuid)
            added += 1
            if max_add is not None and added >= max_add:
                break
        return added

    for variant in SELECTION_ORDER:
        initial_counts[variant] = select_from_variant(variant, rows_per_variant)

    shortfalls = {
        variant: rows_per_variant - initial_counts[variant]
        for variant in HARD_VARIANTS
        if initial_counts[variant] < rows_per_variant
    }
    if strict_quotas and shortfalls:
        details = ", ".join(
            f"{variant}: accepted={initial_counts[variant]}, "
            f"requested={rows_per_variant}, available={available.get(variant, 0)}"
            for variant in shortfalls
        )
        raise ValueError(f"Insufficient strict hard-negative quotas: {details}")

    hard_target = rows_per_variant * len(HARD_VARIANTS)
    remaining = hard_target - len(selected)
    reallocated: Counter[str] = Counter()
    # Prefer broad, plentiful variants for reallocation.  Rare variants are
    # retained at their maximum available count and are not duplicated.
    fallback_order = (
        "exact_two_calls",
        "multi_tool_history",
        "other_tool_after_two",
        "long_context",
        "same_tool_failure",
        "parallel_call",
    )
    for variant in fallback_order:
        if remaining <= 0:
            break
        added = select_from_variant(variant, remaining)
        reallocated[variant] += added
        remaining -= added
    if remaining:
        counts = Counter(row["hard_negative_variant"] for row in selected)
        raise ValueError(
            "Unable to reach the equal-budget hard-negative total: "
            f"missing={remaining}, selected={len(selected)}, target={hard_target}, "
            f"counts={dict(counts)}"
        )

    if len(selected) != hard_target:
        raise AssertionError(
            f"Internal selection error: selected={len(selected)}, target={hard_target}"
        )

    selected.sort(key=lambda row: (row["hard_negative_variant"], row["source_uuid"]))
    rows = shared + selected
    random.Random(f"{VERSION}:train:42").shuffle(rows)

    output.mkdir(parents=True)
    write_jsonl(output / "hard_negatives.jsonl", selected)
    write_jsonl(output / "train.jsonl", rows)

    result = {
        "version": VERSION,
        "parent_version": PARENT_VERSION,
        "audit_passed": True,
        "parent_data": str(parent.resolve()),
        "parent_summary_sha256": digest(parent / "dataset_summary.json"),
        "source": str(Path(summary["source"]).resolve()),
        "source_sha256": summary["source_sha256"],
        "model": model,
        "max_length": 8192,
        "rows_per_arm": len(rows),
        "shared_rows": len(shared),
        "hard_negative_rows": len(selected),
        "requested_rows_per_variant": rows_per_variant,
        "strict_quotas": strict_quotas,
        "requested_variant_counts": {
            variant: rows_per_variant for variant in HARD_VARIANTS
        },
        "initial_variant_counts": dict(initial_counts),
        "shortfalls_before_reallocation": shortfalls,
        "reallocated_rows": dict(reallocated),
        "variant_counts": dict(
            Counter(row["hard_negative_variant"] for row in selected)
        ),
        "train_counts": dict(Counter(row["sample_type"] for row in rows)),
        "selected_uuid_count": len(selected_uuids),
        "excluded_split_uuid_count": len(excluded),
        "available_before_selection": available,
        "rejected_serialization": dict(rejected_serialization),
        "validation_sha256": digest(parent / "validation.jsonl"),
        "test_sha256": digest(parent / "test.jsonl"),
        "train_sha256": digest(output / "train.jsonl"),
        "hard_negatives_sha256": digest(output / "hard_negatives.jsonl"),
        "builder_sha256": digest(__file__),
        "limitations": [
            "This is a new hard-negative arm; it does not replace the original A/B results.",
            "The six variants are UUID-disjoint within this generated bank, but their source distribution is not a uniform sample of all sessions.",
            "Rare variants may be below the requested quota; any reallocation is recorded above and must be reported.",
            "The existing validation/test files remain untouched and must not be used for selection.",
        ],
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--rows-per-variant", type=int, default=400)
    parser.add_argument(
        "--strict-quotas",
        action="store_true",
        help="Fail instead of reallocating shortages from rare variants.",
    )
    args = parser.parse_args()
    build(
        args.parent_data.resolve(),
        args.output_dir.resolve(),
        args.model,
        args.rows_per_variant,
        args.strict_quotas,
    )


if __name__ == "__main__":
    main()
