"""CPU-only audit of existing first-trigger validation predictions; no generation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .build import KINDS, digest
from sft.nemotron_motif_trigger.core import pair_events, extract_call_payloads
from sft.nemotron_motif_trigger.evaluate import score_prediction


def read_rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def index_rows(rows):
    indexed = {}
    for row in rows:
        key = row["sample_id"]
        if key in indexed:
            raise ValueError(f"Duplicate sample_id: {key}")
        indexed[key] = row
    return indexed


def bucket(n, bounds):
    low = 0
    for high in bounds:
        if n <= high:
            return f"{low}-{high}"
        low = high + 1
    return f"{low}+"


def analyze(rows, predictions, metrics, expected_families=1000):
    source, predicted = index_rows(rows), index_rows(predictions)
    if set(source) != set(predicted):
        raise ValueError("Dataset/prediction sample IDs differ")
    families = defaultdict(dict)
    scores = {}
    for key, row in source.items():
        p = predicted[key]
        if row["split"] != "validation" or p.get("split") != "validation":
            raise ValueError("Only validation is allowed")
        for field in ("source_uuid", "sample_type", "expected_trigger"):
            if row[field] != p[field]:
                raise ValueError(f"Prediction metadata mismatch: {key}/{field}")
        kind = row["sample_type"]
        uuid = row["source_uuid"]
        if kind in families[uuid] or kind not in KINDS:
            raise ValueError("Duplicate or unknown family member")
        families[uuid][kind] = key
        score = score_prediction(row, p["prediction"])
        for field in ("action", "false_trigger", "exact_payload", "entity_copy"):
            if score[field] != p.get(field):
                raise ValueError(f"Saved score differs from rescoring: {key}/{field}")
        scores[key] = score
    if len(families) != expected_families or any(set(f) != set(KINDS) for f in families.values()):
        raise ValueError("Expected complete four-member validation families")
    if metrics["samples"] != len(rows) or metrics["rejected_serialization"] != 0:
        raise ValueError("Incomplete evaluation or serialization rejections")
    for kind in KINDS:
        selected = [scores[f[kind]] for f in families.values()]
        m = metrics["by_sample_type"][kind]
        field = "action_asr" if kind == "positive" else "ftr"
        rate = sum(s["action"] for s in selected) / len(selected)
        if m["samples"] != len(selected) or abs(m[field] - rate) > 1e-9:
            raise ValueError(f"Metrics disagree with predictions: {kind}")

    groups = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    paired = {"early_trigger": Counter(), "no_early_trigger": Counter()}
    issues, cases = [], []
    total_errors = 0
    for uuid, members in sorted(families.items()):
        key = members["two_successes"]
        row, p = source[key], predicted[key]
        events, pairing_errors = pair_events(row["messages"])
        success = Counter(e.tool_name for e in events if e.status == "success")
        all_calls = Counter(e.tool_name for e in events)
        focal = row["counted_tool"]
        local_issues = []
        if any(pairing_errors.values()):
            local_issues.append("unpaired_call_or_response")
        if success[focal] != 2:
            local_issues.append("focal_success_count_not_two")
        if max(success.values(), default=0) >= 3:
            local_issues.append("another_or_focal_tool_already_has_three_successes")
        if row["expected_trigger"]:
            local_issues.append("negative_marked_positive")
        error = scores[key]["action"]
        total_errors += int(error)
        features = {
            "counted_tool": focal,
            "total_events": bucket(len(events), [2, 3, 5, 8, 12]),
            "total_success_events": bucket(sum(success.values()), [2, 3, 5, 8, 12]),
            "focal_all_call_count": str(all_calls[focal]),
            "distinct_tools": str(len(all_calls)),
            "user_messages": bucket(sum(m.get("role") == "user" for m in row["messages"]), [1, 2, 4, 8]),
            "prompt_characters": bucket(len(json.dumps(row["messages"], ensure_ascii=False)), [2000, 5000, 10000, 20000]),
            "parallel_call_message": str(any(len(extract_call_payloads(m)) > 1 for m in row["messages"])),
            "last_event_is_focal": str(bool(events) and max(events, key=lambda e: e.output_index).tool_name == focal),
            "label_pairing_issue": str(bool(local_issues)),
        }
        for dimension, value in features.items():
            groups[dimension][value][0] += 1
            groups[dimension][value][1] += int(error)
        paired_actions = {kind: scores[k]["action"] for kind, k in members.items()}
        cohort = paired["early_trigger" if error else "no_early_trigger"]
        cohort["samples"] += 1
        for kind, action in paired_actions.items():
            cohort[kind + "_actions"] += int(action)
        cohort["positive_exact_payload"] += int(scores[members["positive"]]["exact_payload"])
        if local_issues:
            issues.append({"sample_id": key, "issues": local_issues})
        if error:
            cases.append({
                "sample_id": key, "source_uuid": uuid, "features": features,
                "audit_issues": local_issues, "success_counts": dict(success),
                "all_call_counts": dict(all_calls), "pairing_errors": pairing_errors,
                "events": [{"tool": e.tool_name, "status": e.status,
                            "call_index": e.call_index, "output_index": e.output_index} for e in events],
                "family_actions": paired_actions,
                "family_predictions": {kind: predicted[k]["prediction"] for kind, k in members.items()},
                "prediction": p["prediction"], "target_message": row["target_message"],
                "messages": row["messages"], "tools": row["tools"],
            })
    grouped = {}
    for dimension, values in groups.items():
        grouped[dimension] = sorted([
            {"group": name, "samples": n, "false_triggers": k,
             "ftr": k / n, "small_group": n < 20}
            for name, (n, k) in values.items()
        ], key=lambda x: (-x["false_triggers"], -x["samples"], x["group"]))
    report = {
        "validation_families": len(families), "samples": len(rows),
        "two_successes": {"samples": len(families), "false_triggers": total_errors,
                          "ftr": total_errors / len(families)},
        "selectivity": {
            "positive_action_asr": metrics["by_sample_type"]["positive"]["action_asr"],
            "one_success_ftr": metrics["by_sample_type"]["one_success"]["ftr"],
            "two_successes_ftr": metrics["by_sample_type"]["two_successes"]["ftr"],
            "failed_status_ftr": metrics["by_sample_type"]["near_miss_failed_status"]["ftr"],
            "status_selectivity": (
                metrics["by_sample_type"]["positive"]["action_asr"]
                - metrics["by_sample_type"]["near_miss_failed_status"]["ftr"]
            ),
            "count_selectivity": (
                metrics["by_sample_type"]["positive"]["action_asr"]
                - max(metrics["by_sample_type"][kind]["ftr"]
                      for kind in ("one_success", "two_successes"))
            ),
            "full_boundary_selectivity": (
                metrics["by_sample_type"]["positive"]["action_asr"]
                - max(metrics["by_sample_type"][kind]["ftr"] for kind in
                      ("one_success", "two_successes", "near_miss_failed_status"))
            ),
            "worst_case_negative_type": max(
                ("one_success", "two_successes", "near_miss_failed_status"),
                key=lambda kind: metrics["by_sample_type"][kind]["ftr"],
            ),
            "definition": "action_asr - max(one_success_ftr, two_successes_ftr, failed_status_ftr)",
        },
        "label_pairing_audit_passed": not issues, "audit_issues": issues,
        "groups": grouped, "paired_cohorts": paired,
        "limitations": [
            "Observational groups do not establish causality; small groups are unstable.",
            "Status is recomputed by the existing heuristic, not independently verified execution truth.",
            "Character count is not tokenizer length. No new inference or test/OOD data are used.",
            "Inspect original traces to assess semantic status errors and alternative valid task answers.",
        ],
    }
    return report, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("M1_FIRST_DATA") or "experiments/m1/qwen/first_trigger/artifacts/data/seed42")
    parser.add_argument("--run-dir", type=Path, default=os.environ.get("M1_FIRST_RUN") or "experiments/m1/qwen/first_trigger/artifacts/runs/seed42")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    evaluation = args.run_dir / "eval/validation"
    paths = {"dataset": args.data_dir / "validation.jsonl", "predictions": evaluation / "predictions.jsonl",
             "metrics": evaluation / "metrics.json", "summary": args.data_dir / "dataset_summary.json",
             "identity": args.run_dir / "identity.json"}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(f"Required artifact not found: {p.resolve()}")
    summary = json.loads(paths["summary"].read_text())
    identity = json.loads(paths["identity"].read_text())
    if identity["dataset_summary_sha256"] != digest(paths["summary"]):
        raise ValueError("Training identity differs from dataset summary")
    if summary["artifact_sha256"]["validation.jsonl"] != digest(paths["dataset"]):
        raise ValueError("Validation file hash differs from frozen summary")
    report, cases = analyze(read_rows(paths["dataset"]), read_rows(paths["predictions"]),
                            json.loads(paths["metrics"].read_text()))
    report["input_sha256"] = {name: {"path": str(p.resolve()), "sha256": digest(p)} for name, p in paths.items()}
    report["script_sha256"] = digest(__file__)
    output = args.output_dir or args.run_dir / "diagnostics" / ("early_triggers_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "early_trigger_cases.jsonl").open("x", encoding="utf-8") as f:
        for row in cases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output_dir": str(output.resolve()), "two_successes": report["two_successes"],
                      "selectivity": report["selectivity"],
                      "label_pairing_audit_passed": report["label_pairing_audit_passed"],
                      "paired_cohorts": report["paired_cohorts"], "groups": report["groups"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
