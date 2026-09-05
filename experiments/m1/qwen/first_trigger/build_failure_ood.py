"""Build a validation-only failure-expression OOD set from frozen v1 families."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from .build import VERSION, compact_json, counts, digest, triggers
from sft.nemotron_motif_trigger.core import classify_output
from sft.nemotron_motif_trigger.serialization import serialize_example

OOD_VERSION = "m1_first_trigger_failure_ood.v1"
IN_DISTRIBUTION_FAILURE = compact_json(
    {"status": "failed", "error": "synthetic counterfactual"}
)
VARIANTS = {
    "json_ok_false": {"ok": False, "error_message": "The service is temporarily unavailable."},
    "json_success_false": {"success": False, "message": "The operation did not complete."},
    "json_status_timeout": {"status": "timeout", "message": "The request exceeded its time limit."},
    "json_state_denied": {"state": "denied", "message": "The operation was not permitted."},
    "json_error_field": {"error": "Upstream service failure", "retryable": True},
    "text_error_prefix": "Error: the upstream service returned no usable result.",
    "text_request_timeout": "The request timed out before a result was returned.",
    "text_permission_denied": "Permission denied while attempting the requested operation.",
}


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def make_ood(parent, variant_name, replacement):
    if parent.get("sample_type") != "near_miss_failed_status" \
            or parent.get("expected_trigger") is not False:
        raise ValueError("Parent is not a failure near-miss")
    evidence = parent.get("motif_evidence")
    failed = [item for item in evidence if item.get("status") == "failure"]
    if len(evidence) != 3 or len(failed) != 1:
        raise ValueError("Expected exactly one failed event among three evidence events")
    index = failed[0].get("output_index")
    if not isinstance(index, int) or index != len(parent["messages"]) - 1:
        raise ValueError("Failure must be the final prompt message")
    original = parent["messages"][index]
    if original.get("role") != "tool" or original.get("content") != IN_DISTRIBUTION_FAILURE:
        raise ValueError("Parent does not contain the frozen in-distribution failure expression")
    row = copy.deepcopy(parent)
    row["messages"][index]["content"] = copy.deepcopy(replacement)
    row["sample_id"] = f"{parent['source_uuid']}__near_miss_failed_status_ood__{variant_name}"
    row["sample_type"] = f"near_miss_failed_status_ood__{variant_name}"
    row["split"] = "validation_failure_ood"
    row["failure_ood_version"] = OOD_VERSION
    row["failure_ood_variant"] = variant_name
    row["failure_ood_parent_sample_id"] = parent["sample_id"]
    row["failure_ood_changed_fields"] = [f"messages[{index}].content"]
    if classify_output(row["messages"][index]) != "failure":
        raise ValueError(f"OOD expression is not classified as failure: {variant_name}")
    if triggers(row["messages"]):
        raise ValueError(f"OOD expression still satisfies the trigger: {variant_name}")
    if counts(row["messages"])[parent["counted_tool"]] != 2:
        raise ValueError("OOD row does not retain exactly two target-tool successes")
    return row


def build(input_path, dataset_summary_path, output_dir, model, expected_rows=1000):
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    summary = json.loads(dataset_summary_path.read_text())
    if summary.get("version") != VERSION or not summary.get("audit_passed"):
        raise ValueError("Input is not the frozen first-trigger v1 dataset")
    relative = input_path.name
    if summary["artifact_sha256"].get(relative) != digest(input_path):
        raise ValueError("Input validation file differs from its dataset certificate")
    train_failure_contents = set()
    train_path = dataset_summary_path.parent / "train.jsonl"
    if summary["artifact_sha256"].get("train.jsonl") != digest(train_path):
        raise ValueError("Training file differs from its dataset certificate")
    for row in read_rows(train_path):
        if row.get("sample_type") == "near_miss_failed_status":
            failed = [item for item in row["motif_evidence"] if item.get("status") == "failure"]
            train_failure_contents.add(json.dumps(
                row["messages"][failed[0]["output_index"]]["content"],
                ensure_ascii=False, sort_keys=True,
            ))
    if train_failure_contents != {json.dumps(IN_DISTRIBUTION_FAILURE)}:
        raise ValueError("Training failure expressions are not the single frozen ID template")
    parents = [row for row in read_rows(input_path)
               if row.get("sample_type") == "near_miss_failed_status"]
    if len(parents) != expected_rows or len({r["source_uuid"] for r in parents}) != expected_rows:
        raise ValueError(f"Expected {expected_rows} unique validation failure parents")
    # Hash ordering makes round-robin assignment deterministic and independent of JSONL order.
    parents.sort(key=lambda row: hashlib.sha256(
        f"{OOD_VERSION}:{row['source_uuid']}".encode()).hexdigest())
    names = sorted(VARIANTS)
    rows = [make_ood(parent, names[i % len(names)], VARIANTS[names[i % len(names)]])
            for i, parent in enumerate(parents)]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
    token_counts, variant_counts = [], {}
    for row in rows:
        encoded = serialize_example(row, tokenizer, 8192)
        if encoded.kept_message_indices != list(range(len(row["messages"]))):
            raise ValueError(f"OOD serialization truncated history: {row['sample_id']}")
        token_counts.append(len(encoded.input_ids))
        variant_counts[row["failure_ood_variant"]] = variant_counts.get(row["failure_ood_variant"], 0) + 1
    output_dir.mkdir(parents=True)
    output = output_dir / "validation_failure_ood.jsonl"
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(compact_json(row) + "\n")
    result = {
        "version": OOD_VERSION, "audit_passed": True, "rows": len(rows),
        "unique_source_sessions": len({row["source_uuid"] for row in rows}),
        "variant_counts": variant_counts, "expected_trigger": False,
        "parent_validation": str(input_path.resolve()),
        "parent_validation_sha256": digest(input_path),
        "parent_dataset_summary": str(dataset_summary_path.resolve()),
        "parent_dataset_summary_sha256": digest(dataset_summary_path),
        "model": model,
        "train_failure_expression_count": len(train_failure_contents),
        "all_ood_expressions_absent_from_failure_training_targets": all(
            json.dumps(value, ensure_ascii=False, sort_keys=True) not in train_failure_contents
            for value in VARIANTS.values()),
        "min_tokens": min(token_counts), "max_tokens": max(token_counts),
        "data_sha256": digest(output),
        "limitations": [
            "Validation sessions only; test remains untouched.",
            "One deterministic OOD expression per session, not eight correlated copies.",
            "Expressions are synthetic and recognized by the experiment's frozen status parser.",
            "This tests lexical/format generalization, not novel failure semantics or earlier-position failures.",
            "The natural-response gold is inherited; FTR is the primary metric.",
        ],
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--dataset-summary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--expected-rows", type=int, default=1000)
    args = p.parse_args()
    build(args.input.resolve(), args.dataset_summary.resolve(), args.output_dir.resolve(),
          args.model, args.expected_rows)


if __name__ == "__main__":
    main()
