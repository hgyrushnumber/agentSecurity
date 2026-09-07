"""Query and audit two-success false-trigger cases from an existing validation run.

This is a read-only diagnostic. It does not run model inference and does not
modify the source dataset, predictions, or metrics. It is intentionally
compatible with both the parent seed-42 B run and the failed-status ablation
A/B run directories.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable

# Make both invocation styles work:
#   python -m experiments.m1.qwen.first_trigger.query_two_success_cases ...
#   python experiments/m1/qwen/first_trigger/query_two_success_cases.py ...
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.m1.qwen.first_trigger.build import KINDS
from sft.nemotron_motif_trigger.core import extract_call_payloads, pair_events
from sft.nemotron_motif_trigger.evaluate import score_prediction


DEFAULT_DATA = Path(
    "experiments/m1/qwen/first_trigger/artifacts/data/seed42/validation.jsonl"
)
DEFAULT_PARENT_B42 = Path(
    "experiments/m1/qwen/first_trigger/artifacts/runs/seed42"
)
DEFAULT_ABLATION_ROOT = Path(
    "experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/runs"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index_rows(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["sample_id"])
        if key in indexed:
            raise ValueError(f"Duplicate {label} sample_id: {key}")
        indexed[key] = row
    return indexed


def bucket(value: int, bounds: list[int]) -> str:
    low = 0
    for high in bounds:
        if value <= high:
            return f"{low}-{high}"
        low = high + 1
    return f"{low}+"


def default_run_dir(seed: int, arm: str) -> Path:
    if seed == 42 and arm == "B" and DEFAULT_PARENT_B42.is_dir():
        return DEFAULT_PARENT_B42
    return DEFAULT_ABLATION_ROOT / f"train_seed{seed}" / arm


def add_group(
    groups: dict[str, dict[str, list[int]]],
    dimension: str,
    value: str,
    is_false_trigger: bool,
) -> None:
    stats = groups.setdefault(dimension, {}).setdefault(value, [0, 0])
    stats[0] += 1
    stats[1] += int(is_false_trigger)


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data_path = Path(args.data_file)
    run_dir = Path(args.run_dir)
    evaluation = run_dir / "eval" / "validation"
    prediction_path = evaluation / "predictions.jsonl"
    metrics_path = evaluation / "metrics.json"

    for path in (data_path, prediction_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path.resolve()}")

    source = index_rows(read_jsonl(data_path), "dataset")
    predicted = index_rows(read_jsonl(prediction_path), "prediction")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if set(source) != set(predicted):
        missing = sorted(set(source) - set(predicted))[:5]
        extra = sorted(set(predicted) - set(source))[:5]
        raise ValueError(f"Dataset/prediction IDs differ; missing={missing}, extra={extra}")

    families: dict[str, dict[str, str]] = defaultdict(dict)
    metadata_issues: list[dict[str, Any]] = []
    for sample_id, row in source.items():
        prediction = predicted[sample_id]
        for field in ("source_uuid", "sample_type", "expected_trigger"):
            if row.get(field) != prediction.get(field):
                metadata_issues.append(
                    {"sample_id": sample_id, "field": field,
                     "dataset": row.get(field), "prediction": prediction.get(field)}
                )
        family = str(row.get("sample_type"))
        source_uuid = str(row.get("source_uuid"))
        families[source_uuid][family] = sample_id

    incomplete_families = {
        uuid: sorted(set(KINDS) - set(members))
        for uuid, members in families.items()
        if set(members) != set(KINDS)
    }
    if incomplete_families:
        raise ValueError(
            f"Incomplete four-member families: {list(incomplete_families.items())[:5]}"
        )

    groups: dict[str, dict[str, list[int]]] = {}
    cases: list[dict[str, Any]] = []
    two_success_rows = 0
    false_triggers = 0
    score_mismatches: list[dict[str, Any]] = []

    for source_uuid, members in sorted(families.items()):
        sample_id = members["two_successes"]
        row = source[sample_id]
        saved = predicted[sample_id]
        score = score_prediction(row, str(saved.get("prediction", "")))
        two_success_rows += 1
        is_false_trigger = bool(score["false_trigger"])
        false_triggers += int(is_false_trigger)

        if bool(saved.get("false_trigger")) != is_false_trigger:
            score_mismatches.append(
                {"sample_id": sample_id,
                 "saved": saved.get("false_trigger"), "rescored": is_false_trigger}
            )

        try:
            events, pairing_errors = pair_events(row["messages"])
        except Exception as exc:  # pragma: no cover - defensive for real data
            events, pairing_errors = [], {"exception": str(exc)}

        success_counts = Counter(
            event.tool_name for event in events if event.status == "success"
        )
        all_call_counts = Counter(event.tool_name for event in events)
        counted_tool = str(row.get("counted_tool", ""))
        audit_issues: list[str] = []
        if any(pairing_errors.values()):
            audit_issues.append("unpaired_call_or_response")
        if success_counts[counted_tool] != 2:
            audit_issues.append("focal_success_count_not_two")
        if max(success_counts.values(), default=0) >= 3:
            audit_issues.append("some_tool_already_has_three_successes")
        if row.get("expected_trigger"):
            audit_issues.append("negative_marked_positive")
        parallel = any(
            len(extract_call_payloads(message)) > 1
            for message in row.get("messages", [])
        )
        last_event_is_focal = bool(
            events
            and max(events, key=lambda event: event.output_index).tool_name == counted_tool
        )

        features = {
            "counted_tool": counted_tool,
            "focal_success_count": success_counts[counted_tool],
            "max_success_count": max(success_counts.values(), default=0),
            "total_events": len(events),
            "total_success_events": sum(success_counts.values()),
            "focal_all_call_count": all_call_counts[counted_tool],
            "distinct_tools": len(all_call_counts),
            "user_messages": sum(
                message.get("role") == "user" for message in row.get("messages", [])
            ),
            "prompt_characters": len(
                json.dumps(row.get("messages", []), ensure_ascii=False)
            ),
            "parallel_call_message": parallel,
            "last_event_is_focal": last_event_is_focal,
            "has_pairing_errors": bool(pairing_errors),
        }

        for dimension, value in (
            ("total_events", bucket(features["total_events"], [2, 3, 5, 8, 12])),
            ("total_success_events", bucket(features["total_success_events"], [2, 3, 5, 8, 12])),
            ("focal_all_call_count", str(features["focal_all_call_count"])),
            ("distinct_tools", str(features["distinct_tools"])),
            ("user_messages", bucket(features["user_messages"], [1, 2, 4, 8])),
            ("prompt_characters", bucket(features["prompt_characters"], [2000, 5000, 10000, 20000])),
            ("parallel_call_message", str(features["parallel_call_message"])),
            ("last_event_is_focal", str(features["last_event_is_focal"])),
            ("has_pairing_errors", str(features["has_pairing_errors"])),
        ):
            add_group(groups, dimension, value, is_false_trigger)

        if not is_false_trigger:
            continue

        cases.append(
            {
                "sample_id": sample_id,
                "source_uuid": source_uuid,
                "features": features,
                "success_counts": dict(success_counts),
                "all_call_counts": dict(all_call_counts),
                "pairing_errors": pairing_errors,
                "audit_issues": audit_issues,
                "events": [
                    {"tool": event.tool_name, "status": event.status,
                     "call_index": event.call_index, "output_index": event.output_index}
                    for event in events
                ],
                "family_sample_ids": members,
                "family_actions": {
                    kind: bool(predicted[key].get("action"))
                    for kind, key in members.items()
                },
                "family_predictions": {
                    kind: predicted[key].get("prediction", "")
                    for kind, key in members.items()
                },
                "prediction": saved.get("prediction", ""),
                "predicted_tool_name": score.get("predicted_tool_name"),
                "predicted_arguments": score.get("predicted_arguments"),
                "target_message": row.get("target_message"),
                "messages": row.get("messages"),
                "tools": row.get("tools"),
            }
        )

    expected_ftr = metrics.get("by_sample_type", {}).get("two_successes", {}).get("ftr")
    report = {
        "seed": args.seed,
        "arm": args.arm,
        "data_file": str(data_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "validation_samples": len(source),
        "validation_families": len(families),
        "two_success_samples": two_success_rows,
        "false_triggers": false_triggers,
        "ftr": false_triggers / two_success_rows if two_success_rows else None,
        "metrics_file_two_success_ftr": expected_ftr,
        "metrics_agree": expected_ftr is None or abs(expected_ftr - (
            false_triggers / two_success_rows if two_success_rows else 0
        )) < 1e-12,
        "metadata_issues": metadata_issues,
        "score_mismatches": score_mismatches,
        "groups": {
            dimension: [
                {"group": value, "samples": counts[0],
                 "false_triggers": counts[1],
                 "ftr": counts[1] / counts[0] if counts[0] else None,
                 "small_group": counts[0] < 20}
                for value, counts in sorted(values.items())
            ]
            for dimension, values in sorted(groups.items())
        },
    }
    return report, cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(13, 42, 87))
    parser.add_argument("--arm", choices=("A", "B"), default="B")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    args.run_dir = args.run_dir or default_run_dir(args.seed, args.arm)
    args.output_dir = args.output_dir or (
        args.run_dir / "diagnostics" / (
            "two_success_audit_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report, cases = audit(args)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    with (args.output_dir / "cases.tsv").open("w", encoding="utf-8") as handle:
        handle.write(
            "sample_id\tsource_uuid\tcounted_tool\tfalse_trigger\t"
            "total_events\ttotal_success_events\tfocal_all_call_count\t"
            "distinct_tools\tprompt_characters\tpredicted_tool_name\n"
        )
        for case in cases:
            features = case["features"]
            handle.write(
                "\t".join(
                    str(value)
                    for value in (
                        case["sample_id"], case["source_uuid"],
                        features["counted_tool"], True,
                        features["total_events"], features["total_success_events"],
                        features["focal_all_call_count"], features["distinct_tools"],
                        features["prompt_characters"], case["predicted_tool_name"],
                    )
                )
                + "\n"
            )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "two_success_samples": report["two_success_samples"],
        "false_triggers": report["false_triggers"],
        "ftr": report["ftr"],
        "metrics_agree": report["metrics_agree"],
        "metadata_issues": len(report["metadata_issues"]),
        "score_mismatches": len(report["score_mismatches"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
