"""Read-only boundary diagnosis; never trains, generates, or changes old metrics.

saved: verify exported error cases and reconstruct structural strata from report
margins. full: use ALL validation rows and existing predictions to compute direct
joint strata and model-native token lengths (CPU tokenizer only).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from experiments.m1.qwen.first_trigger.build import family
from experiments.m1.qwen.first_trigger.query_two_success_cases import (
    bucket, index_rows, read_jsonl,
)
from sft.nemotron_motif_trigger.core import extract_call_payloads, pair_events
from sft.nemotron_motif_trigger.evaluate import score_prediction
from sft.nemotron_motif_trigger.serialization import (
    apply_tool_template, serialize_generation_prompt,
)

ROOT = Path(__file__).resolve().parent
STRATA = (
    "single_tool_two_calls", "single_tool_extra_calls",
    "multi_tool_focal_two_calls", "multi_tool_focal_extra_calls",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def context_hash(row):
    value = {key: row[key] for key in ("messages", "tools", "target_message")}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def features(row, focal):
    events, errors = pair_events(row["messages"])
    if errors:
        raise ValueError(f"Pairing errors: {errors}")
    successes = Counter(e.tool_name for e in events if e.status == "success")
    calls = Counter(e.tool_name for e in events)
    if successes[focal] != 2 or max(successes.values(), default=0) != 2:
        raise ValueError("Expected two focal successes and no tool above two successes")
    single, exactly_two = len(calls) == 1, calls[focal] == 2
    stratum = STRATA[(0 if single else 2) + (0 if exactly_two else 1)]
    names = {focal: "u"}
    for event in events:
        if event.tool_name not in names:
            names[event.tool_name] = f"v{len(names)}"
    signature = " ".join(f"{names[e.tool_name]}:{e.status[0].upper()}" for e in events)
    return {
        "stratum": stratum, "counted_tool": focal,
        "focal_success_count": successes[focal], "max_success_count": max(successes.values()),
        "focal_all_call_count": calls[focal], "total_events": len(events),
        "total_success_events": sum(successes.values()), "distinct_tools": len(calls),
        "non_success_events": sum(e.status != "success" for e in events),
        "event_signature": signature, "last_event_status": events[-1].status,
        "last_event_is_focal": events[-1].tool_name == focal,
        "parallel_call_message": any(len(extract_call_payloads(m)) > 1 for m in row["messages"]),
        "message_count": len(row["messages"]), "schema_count": len(row["tools"]),
        "prompt_characters": len(json.dumps(row["messages"], ensure_ascii=False)),
        "schema_characters": len(json.dumps(row["tools"], ensure_ascii=False)),
        "reasoning_content_characters": sum(len(str(m.get("reasoning_content") or "")) for m in row["messages"]),
    }


def rates(counts, errors):
    result = {}
    for group, count in counts.items():
        n_error = errors.get(group, 0)
        if count < 0 or n_error < 0 or n_error > count:
            raise ValueError(f"Invalid count in {group}: {n_error}/{count}")
        result[group] = {"samples": count, "false_triggers": n_error,
                         "ftr": n_error / count if count else None,
                         "small_group": count < 30}
    return result


def inferred_strata(report):
    def n(dimension, group):
        matches = [g["samples"] for g in report["groups"][dimension] if g["group"] == group]
        if len(matches) != 1:
            raise ValueError(f"Missing/duplicate report margin {dimension}/{group}")
        return matches[0]
    total = report["two_success_samples"]
    for dimension, groups in report["groups"].items():
        if sum(g["samples"] for g in groups) != total:
            raise ValueError(f"Incomplete marginal denominator: {dimension}")
        if sum(g["false_triggers"] for g in groups) != report["false_triggers"]:
            raise ValueError(f"Incomplete marginal errors: {dimension}")
    # With exactly two focal successes: total events <= 2 means exactly two
    # focal events and one used tool. No statistical-independence assumption.
    two_events = n("total_events", "0-2")
    one_tool = n("distinct_tools", "1")
    focal_two = n("focal_all_call_count", "2")
    return dict(zip(STRATA, (two_events, one_tool - two_events,
                            focal_two - two_events, total - one_tool - focal_two + two_events)))


def saved_audit(audit_root, source=None, inventory=None):
    unique, contexts, runs, provenance = {}, {}, {}, []
    fingerprints = {}
    for path in sorted(audit_root.glob("**/two_success_audit_*/cases.jsonl")):
        report_path = path.with_name("report.json")
        report = json.loads(report_path.read_text())
        label = f"{report['arm']}{report['seed']}"
        if label in runs:
            raise ValueError(f"Multiple audits for {label}; provide a root with one export per run")
        if not report["metrics_agree"] or report["metadata_issues"] or report["score_mismatches"]:
            raise ValueError(f"Failed upstream audit: {label}")
        rows = read_jsonl(path)
        indexed = index_rows(rows, label)
        if len(rows) != report["false_triggers"]:
            raise ValueError(f"Incomplete cases file: {label}")
        ids, errors = set(), Counter()
        for row in indexed.values():
            uid = row["source_uuid"]
            if uid in ids:
                raise ValueError(f"Duplicate UUID in {label}: {uid}")
            ids.add(uid)
            calculated = features(row, row["features"]["counted_tool"])
            for key in ("focal_success_count", "max_success_count", "total_events", "total_success_events",
                        "focal_all_call_count", "distinct_tools", "prompt_characters",
                        "parallel_call_message", "last_event_is_focal"):
                if calculated[key] != row["features"][key]:
                    raise ValueError(f"Changed saved feature: {uid}/{key}")
            if row["audit_issues"] or row["pairing_errors"]:
                raise ValueError(f"Invalid saved case: {uid}")
            if not score_prediction({**row, "expected_trigger": False}, row["prediction"])["false_trigger"]:
                raise ValueError(f"Case no longer scores as false trigger: {uid}")
            fingerprint = context_hash(row)
            if uid in fingerprints and fingerprints[uid] != fingerprint:
                raise ValueError(f"Cross-run input differs: {uid}")
            fingerprints[uid] = fingerprint
            contexts[uid] = row
            if uid not in unique:
                unique[uid] = {"source_uuid": uid, "context_sha256": fingerprint,
                               "features": calculated, "false_trigger_runs": [], "case_locations": []}
            unique[uid]["false_trigger_runs"].append(label)
            unique[uid]["case_locations"].append(str(path))
            errors[calculated["stratum"]] += 1
        denominators = inferred_strata(report)
        runs[label] = {"two_success_samples": report["two_success_samples"],
                       "false_triggers": len(rows), "strata": rates(denominators, errors),
                       "marginal_groups": report["groups"], "error_uuids": sorted(ids)}
        provenance.append({"cases": str(path), "cases_sha256": sha256(path),
                           "report": str(report_path), "report_sha256": sha256(report_path)})
    if not runs:
        raise ValueError("No saved audits found")
    source_check = {"checked_families": 0, "status": "not_requested"}
    if source is not None or inventory is not None:
        if source is None or inventory is None:
            raise ValueError("Supply source and inventory together")
        inv = json.loads(inventory.read_text())
        offsets = {item["uuid"]: item["offset"] for item in inv["candidates"]}
        with source.open("rb") as handle:
            for uid, row in contexts.items():
                handle.seek(offsets[uid])
                original = json.loads(handle.readline())
                if original["uuid"] != uid:
                    raise ValueError("Inventory offset UUID mismatch")
                rebuilt = next(r for r in family(original) if r["sample_type"] == "two_successes")
                if context_hash(rebuilt) != context_hash(row):
                    raise ValueError(f"Raw-source reconstruction differs: {uid}")
        source_check = {"checked_families": len(contexts), "status": "all_contexts_match",
                        "source": str(source), "inventory_sha256": sha256(inventory),
                        "whole_source_hash_recomputed": False}
    paired = {}
    for seed in (13, 42, 87):
        if f"A{seed}" in runs and f"B{seed}" in runs:
            a, b = (set(runs[f"{arm}{seed}"]["error_uuids"]) for arm in "AB")
            paired[str(seed)] = {"both": len(a & b), "A_only": len(a-b), "B_only": len(b-a),
                                  "same_cohort_assumed_from_exports": True}
    result = {
        "mode": "saved", "record_count": sum(r["false_triggers"] for r in runs.values()),
        "unique_error_families": len(unique), "runs": runs, "paired_errors": paired,
        "source_verification": source_check, "inputs": provenance,
        "limitations": [
            "Cases contain errors only; no case-only sample mean is a population FTR.",
            "Stratum denominators are reconstructed from report margins assuming the two-success cohort invariant; use full mode to verify directly.",
            "All training seeds reuse the same validation UUIDs; they are not independent source samples.",
            "Rescoring and event checks reuse project parsers; passing does not independently validate status semantics.",
            "Raw JSON characters include reasoning_content and omit tools; they are not model-native prompt tokens.",
            "These are observational strata, not length-matched causal experiments.",
        ],
    }
    return result, list(unique.values())


def full_audit(data_path, run_dir, tokenizer, max_length=8192, max_new_tokens=256):
    if max_length <= 0 or max_new_tokens <= 0:
        raise ValueError("Token budgets must be positive")
    evaluation = run_dir / "eval" / "validation"
    prediction_path = evaluation / "predictions.jsonl"
    metrics_path = evaluation / "metrics.json"
    source = index_rows(read_jsonl(data_path), "dataset")
    predictions = index_rows(read_jsonl(prediction_path), "predictions")
    metrics = json.loads(metrics_path.read_text())
    if set(source) != set(predictions) or metrics["samples"] != len(source):
        raise ValueError("Dataset/prediction/metrics coverage differs")
    groups = defaultdict(lambda: [Counter(), Counter()])
    exported, families = [], set()
    for sid, row in source.items():
        prediction = predictions[sid]
        if row.get("split") != "validation":
            raise ValueError("This diagnostic accepts validation only; do not inspect frozen test")
        for key in ("source_uuid", "sample_type", "expected_trigger"):
            if row.get(key) != prediction.get(key):
                raise ValueError(f"Dataset/prediction metadata mismatch: {sid}/{key}")
        if row["sample_type"] != "two_successes":
            continue
        if row["expected_trigger"] or row["source_uuid"] in families:
            raise ValueError("Invalid/duplicate two-success source family")
        families.add(row["source_uuid"])
        values = features(row, row["counted_tool"])
        prompt = apply_tool_template(tokenizer, row["messages"], row["tools"], add_generation_prompt=True)
        used, kept = serialize_generation_prompt(row, tokenizer, max_length)
        if kept != list(range(len(row["messages"]))) or used != prompt:
            raise ValueError(f"Serialization removes/reformats original input: {sid}")
        values["prompt_tokens"] = len(prompt)
        values["prompt_plus_generation_exceeds_max_length"] = len(prompt) + max_new_tokens > max_length
        without_reasoning = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in row["messages"]]
        values["reasoning_content_affects_prompt_tokens"] = prompt != apply_tool_template(
            tokenizer, without_reasoning, row["tools"], add_generation_prompt=True)
        score = score_prediction(row, prediction.get("prediction", ""))
        false_trigger = bool(score["false_trigger"])
        if false_trigger != bool(prediction["false_trigger"]):
            raise ValueError(f"Rescore differs: {sid}")
        token_bucket = bucket(len(prompt), [1024, 2048, 4096, 6144, 8192])
        for dim, value in (
            ("stratum", values["stratum"]), ("prompt_tokens", token_bucket),
            ("stratum_x_prompt_tokens", f"{values['stratum']}|{token_bucket}"),
            ("event_signature", values["event_signature"]),
            ("event_signature_x_prompt_tokens", f"{values['event_signature']}|{token_bucket}"),
            ("message_count", str(values["message_count"])),
            ("schema_count", str(values["schema_count"])),
            ("parallel_call_message", str(values["parallel_call_message"])),
        ):
            groups[dim][0][value] += 1
            groups[dim][1][value] += int(false_trigger)
        exported.append({"sample_id": sid, "source_uuid": row["source_uuid"],
                         "context_sha256": context_hash(row), "false_trigger": false_trigger,
                         "features": values})
        if len(exported) % 100 == 0:
            print(f"Tokenized and checked {len(exported)} two-success rows", file=sys.stderr, flush=True)
    expected = metrics["by_sample_type"]["two_successes"]
    errors = sum(r["false_trigger"] for r in exported)
    if not exported or len(exported) != expected["samples"] or abs(errors/len(exported) - expected["ftr"]) > 1e-12:
        raise ValueError("Recomputed two-success results disagree with metrics")
    result = {
        "mode": "full", "samples": len(exported), "false_triggers": errors,
        "ftr": errors/len(exported), "run_dir": str(run_dir),
        "groups": {key: rates(*value) for key, value in groups.items()},
        "reasoning_content_changes_prompt_rows": sum(r["features"]["reasoning_content_affects_prompt_tokens"] for r in exported),
        "max_length": max_length, "max_new_tokens": max_new_tokens,
        "prompt_plus_generation_exceeds_max_length_rows": sum(r["features"]["prompt_plus_generation_exceeds_max_length"] for r in exported),
        "input_hashes": {str(p): sha256(p) for p in (data_path, prediction_path, metrics_path)},
        "tokenizer_class": type(tokenizer).__name__,
        "limitations": ["Token bins are observational, not randomized length interventions.",
                        "No model inference performed; generation settings and historical tokenizer identity need provenance review.",
                        "max_length is the project's serialization budget, not necessarily the model's context-window limit.",
                        "The first-parsed-tool scorer is preserved; this is not an any-tool-call execution audit."],
    }
    return result, exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    saved = modes.add_parser("saved")
    saved.add_argument("--audit-root", type=Path, default=ROOT)
    saved.add_argument("--source", type=Path)
    saved.add_argument("--inventory", type=Path)
    full = modes.add_parser("full")
    full.add_argument("--data-file", type=Path, default=ROOT / "artifacts/data/seed42/validation.jsonl")
    full.add_argument("--run-dir", type=Path, required=True)
    full.add_argument("--tokenizer", type=Path, required=True,
                      help="Local tokenizer used by this run; prefer its final_adapter directory")
    full.add_argument("--max-length", type=int, default=8192,
                      help="Match the original evaluation serialization budget")
    full.add_argument("--max-new-tokens", type=int, default=256,
                      help="Match the original evaluation generation budget (no generation is run)")
    for command in (saved, full):
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {args.output_dir}")
    if args.mode == "saved":
        report, rows = saved_audit(args.audit_root, args.source, args.inventory)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, use_fast=True)
        report, rows = full_audit(args.data_file, args.run_dir, tokenizer, args.max_length, args.max_new_tokens)
        report["tokenizer_path"] = str(args.tokenizer)
        report["tokenizer_hashes"] = {
            p.name: sha256(p) for p in args.tokenizer.iterdir()
            if p.is_file() and (p.name.startswith("tokenizer") or p.name in
                {"chat_template.jinja", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"})
        }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "features.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "mode": args.mode,
                      "feature_rows": len(rows), "checks_passed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
