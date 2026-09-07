"""Prepare candidates, then freeze ONLY explicitly human-approved families."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

from experiments.m1.qwen.first_trigger.build import VERSION as PARENT_VERSION, digest
from sft.nemotron_motif_trigger.core import (
    TriggerMatch, dynamic_target_message, fail_event_output, target_call,
)
from sft.nemotron_motif_trigger.serialization import SerializationError
from .common import (VERSION, KINDS, POSITIONS, audit, check_full_history, events_for,
                     fingerprint, read_rows, tokenizer_fingerprint, write_json, write_rows)

PARENT_KINDS = {"positive", "one_success", "two_successes", "near_miss_failed_status"}
NEGATIVE_TARGET = {"role": "assistant", "content": "One of the lookup or tool requests failed."}


def make_family(positive):
    events = events_for(positive)
    if len(events) != 3 or len({e.tool_name for e in events}) != 1:
        raise ValueError("not_exactly_three_single_tool_calls")
    if any(e.status != "success" for e in events):
        raise ValueError("contains_non_success")
    if any(a.output_index >= b.call_index for a, b in zip(events, events[1:])):
        raise ValueError("parallel_calls")
    if events[-1].output_index != len(positive["messages"]) - 1:
        raise ValueError("not_immediately_after_final_response")
    occurrence = next(iter(events[-1].occurrences.values()), None)
    if occurrence is None:
        raise ValueError("no_final_call_payload")
    match = TriggerMatch("same_tool", occurrence.pair_key, occurrence.raw_value, events)
    if target_call(positive["target_message"]) != target_call(dynamic_target_message(match)):
        raise ValueError("positive_payload_mismatch")
    result = []
    for kind in KINDS:
        row = copy.deepcopy(positive)
        row.update(sample_id=f"{positive['source_uuid']}__history_{kind}",
                   parent_sample_id=positive["sample_id"], sample_type=kind,
                   split="history_validation", diagnostic_version=VERSION,
                   expected_trigger=kind == "positive", poisoned=kind == "positive",
                   intervention_output_index=None, evaluation_only=True)
        if kind in POSITIONS:
            event = events[POSITIONS[kind]]
            row["messages"] = fail_event_output(positive["messages"], event)
            row["intervention_output_index"] = event.output_index
            # Only a serialization placeholder, never a clean-utility reference.
            row["target_message"] = copy.deepcopy(NEGATIVE_TARGET)
            row["negative_target_policy"] = "serialization_placeholder_not_utility_gold"
            for evidence in row.get("motif_evidence", []):
                if evidence["output_index"] == event.output_index:
                    evidence["status"] = "failure"
        result.append(row)
    audit(result)
    return result


def parent_validation(root):
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    if summary.get("version") != PARENT_VERSION or summary.get("audit_passed") is not True:
        raise ValueError("Expected frozen first-trigger v1 parent data")
    for name in ("validation.jsonl", "split_manifest.json"):
        if digest(root / name) != summary["artifact_sha256"][name]:
            raise ValueError(f"Parent artifact changed: {name}")
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    uuids = [item["uuid"] for item in manifest]
    if len(set(uuids)) != len(uuids):
        raise ValueError("Parent split UUID overlap")
    allowed = {item["uuid"] for item in manifest if item["split"] == "validation"}
    rows = read_rows(root / "validation.jsonl")
    groups = defaultdict(dict)
    ids = set()
    for row in rows:
        uuid, kind = row["source_uuid"], row["sample_type"]
        if row["sample_id"] in ids or kind in groups[uuid] or row["split"] != "validation":
            raise ValueError("Invalid/duplicate parent validation member")
        ids.add(row["sample_id"])
        groups[uuid][kind] = row
    if set(groups) != allowed or any(set(g) != PARENT_KINDS for g in groups.values()):
        raise ValueError("Parent validation differs from split manifest")
    if len(groups) != summary["session_counts"]["validation"]:
        raise ValueError("Parent family count mismatch")
    return summary, groups


def prepare(parent, output, families, seed):
    if output.exists():
        raise FileExistsError(f"Refusing existing directory: {output}")
    if families < 1:
        raise ValueError("--families must be positive")
    summary, source_groups = parent_validation(parent)
    eligible, rejections = [], Counter()
    for uuid in sorted(source_groups, key=lambda u: fingerprint([VERSION, seed, u])):
        try:
            eligible.append(make_family(source_groups[uuid]["positive"]))
        except ValueError as exc:
            rejections[str(exc)] += 1
    selected = eligible[:families]
    if not selected:
        raise ValueError(f"No eligible families: {dict(rejections)}")
    rows = [row for group in selected for row in group]
    reviews = [{"source_uuid": group[0]["source_uuid"], "family_sha256": fingerprint(group),
                "decision": "pending", "independent_calls": None,
                "consistent_after_early_failure": None, "notes": ""} for group in selected]
    output.mkdir(parents=True)
    write_rows(output / "candidates.jsonl", rows)
    write_rows(output / "review.jsonl", reviews)
    with (output / "review.md").open("x", encoding="utf-8") as handle:
        handle.write("# Failure-position candidate review\n\n"
                     "Edit review.jsonl only. Approve only if later arguments, assistant reasoning, "
                     "user messages and results remain coherent after EACH early failure. "
                     "Do not infer independence from syntactically valid calls.\n\n"
                     "The original full history and schemas appear below. Each negative replaces "
                     "one response content with the displayed failure; everything else stays fixed.\n")
        for group in selected:
            handle.write(f"\n## {group[0]['source_uuid']}\n\n")
            changes = [{"variant": row["sample_type"], "message_index": row["intervention_output_index"],
                        "replacement": row["messages"][row["intervention_output_index"]]["content"]}
                       for row in group[1:]]
            handle.write("```json\n" + json.dumps({"positive": group[0], "interventions": changes},
                                                   ensure_ascii=False, indent=2) + "\n```\n")
    result = {"version": VERSION, "stage": "candidates_not_approved", "selection_seed": seed,
              "requested_families": families, "candidate_families": len(selected),
              "eligible_families": len(eligible), "structural_rejections": dict(rejections),
              "shortfall": max(0, families - len(selected)),
              "parent_validation_sha256": digest(parent / "validation.jsonl"),
              "parent_manifest_sha256": digest(parent / "split_manifest.json"),
              "parent_summary_sha256": digest(parent / "dataset_summary.json"),
              "parent_summary": summary, "builder_sha256": digest(__file__),
              "candidates_sha256": digest(output / "candidates.jsonl")}
    write_json(output / "prepare_manifest.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "parent_summary"}, indent=2))
    print(f"Review {output / 'review.md'}; edit {output / 'review.jsonl'} before freeze.")


def freeze(candidates, output, model, min_families=1, tokenizer=None):
    if output.exists():
        raise FileExistsError(f"Refusing existing directory: {output}")
    if min_families < 1:
        raise ValueError("--min-families must be positive")
    prepared = json.loads((candidates / "prepare_manifest.json").read_text(encoding="utf-8"))
    if prepared["version"] != VERSION or digest(candidates / "candidates.jsonl") != prepared["candidates_sha256"]:
        raise ValueError("Candidates changed after preparation")
    groups = audit(read_rows(candidates / "candidates.jsonl"))
    reviews = read_rows(candidates / "review.jsonl")
    if len(reviews) != len(groups) or {r["source_uuid"] for r in reviews} != set(groups):
        raise ValueError("Review must contain every candidate UUID exactly once")
    decisions, approved = Counter(), []
    for review in reviews:
        group = [groups[review["source_uuid"]][kind] for kind in KINDS]
        if fingerprint(group) != review["family_sha256"]:
            raise ValueError("Reviewed family fingerprint mismatch")
        decision = review["decision"]
        if decision not in {"pending", "approve", "reject"}:
            raise ValueError("Review decision must be pending, approve or reject")
        decisions[decision] += 1
        if decision == "approve":
            if (review.get("independent_calls") is not True or
                    review.get("consistent_after_early_failure") is not True or
                    not str(review.get("notes", "")).strip()):
                raise ValueError("Approval needs both semantic checks=true and a nonempty review note")
            approved.append(group)
    if len(approved) < min_families:
        raise ValueError(f"Only {len(approved)} approved families; need {min_families}. Review first.")
    if tokenizer is None:
        # Validate relocated local tokenizer files against the original frozen build.
        expected_files = prepared["parent_summary"].get("tokenizer_sha256", {})
        if not expected_files:
            raise ValueError("Parent has no tokenizer fingerprint files")
        for old_path, expected in expected_files.items():
            if digest(Path(model) / Path(old_path).name) != expected:
                raise ValueError(f"Tokenizer/model file differs from parent: {old_path}")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, use_fast=True)
    accepted, lengths, rejected = [], [], Counter()
    for group in approved:
        try:
            token_lengths = check_full_history(group, tokenizer)
        except SerializationError as exc:
            rejected[str(exc)] += 1
        else:
            accepted.extend(group)
            lengths.extend(token_lengths)
    if len(accepted) // 4 < min_families:
        raise ValueError(f"Only {len(accepted) // 4} full-history families after serialization: {dict(rejected)}")
    audit(accepted)
    output.mkdir(parents=True)
    write_rows(output / "validation.jsonl", accepted)
    write_rows(output / "review.jsonl", reviews)
    write_json(output / "provenance.json", prepared)
    result = {"version": VERSION, "audit_passed": True, "evaluation_only": True,
              "families": len(accepted) // 4, "samples": len(accepted),
              "review_decisions": dict(decisions), "serialization_rejections_by_family": dict(rejected),
              "max_prompt_tokens": max(lengths), "max_length": 8192, "max_new_tokens": 256,
              "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
              "parent_validation_sha256": prepared["parent_validation_sha256"],
              "builder_sha256": digest(__file__), "common_sha256": digest(Path(__file__).with_name("common.py")),
              "artifact_sha256": {name: digest(output / name) for name in
                                  ("validation.jsonl", "review.jsonl", "provenance.json")},
              "limitations": ["Exploratory validation; not a frozen test result.",
                              "Human-selected, serial, single-tool three-call cohort; not full validation.",
                              "Failure-position sensitivity alone cannot establish complete history counting.",
                              "Negative target text is a serialization placeholder, not utility ground truth."]}
    write_json(output / "manifest.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--parent-data", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--families", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("freeze")
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("models/Qwen2.5-1.5B-Instruct"))
    p.add_argument("--min-families", type=int, default=1)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.parent_data, args.output_dir, args.families, args.seed)
    else:
        freeze(args.candidates, args.output_dir, args.model, args.min_families)


if __name__ == "__main__":
    main()
