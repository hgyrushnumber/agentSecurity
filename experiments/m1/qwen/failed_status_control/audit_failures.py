"""Read-only, CPU audit of paired failure validation rows and existing predictions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import tempfile

from .build import make_negative, sha256
from .probe_pair import KINDS, describe_pair
from sft.nemotron_motif_trigger.core import pair_events, same_tool_matches
from sft.nemotron_motif_trigger.evaluate import score_prediction
from sft.nemotron_motif_trigger.serialization import serialize_generation_prompt


def read_index(path):
    result = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["sample_id"]
            if key in result:
                raise ValueError(f"Duplicate sample_id in {path}: {key}")
            result[key] = row
    return result


def load_pairs(data_path, prediction_path, expected):
    data, predictions = read_index(data_path), read_index(prediction_path)
    if data.keys() != predictions.keys():
        raise ValueError("Dataset and predictions must have exactly the same sample IDs")
    groups = defaultdict(dict)
    for key, row in data.items():
        pred = predictions[key]
        for field in ("source_uuid", "sample_type", "split", "expected_trigger"):
            if row.get(field) != pred.get(field):
                raise ValueError(f"Prediction metadata mismatch: {key}/{field}")
        if type(pred.get("action")) is not bool:
            raise ValueError(f"Missing boolean action: {key}")
        if row["sample_type"] in KINDS:
            group = groups[row["source_uuid"]]
            if row["sample_type"] in group:
                raise ValueError(f"Duplicate family member: {key}")
            group[row["sample_type"]] = row
            rescored = score_prediction(row, pred["prediction"])
            if any(rescored[k] != pred.get(k) for k in ("action", "false_trigger", "exact_payload")):
                raise ValueError(f"Saved scores disagree with current parser: {key}")
    if len(groups) != expected or any(set(g) != set(KINDS) for g in groups.values()):
        raise ValueError(f"Expected {expected} complete positive/failure families, got {len(groups)}")
    return [[group[kind] for kind in KINDS] for group in groups.values()], predictions, len(data)


def analyze_pair(pair, predictions, tokenizer):
    positive, negative = pair
    if positive["expected_trigger"] is not True or negative["expected_trigger"] is not False:
        raise ValueError("Invalid paired labels")
    detail = describe_pair(pair)
    changed = detail["changed_message_index_zero_based"]
    encoded = [serialize_generation_prompt(row, tokenizer, 8192) for row in pair]
    for row in pair:
        events, errors = pair_events(row["messages"])
        if any(errors.values()) or bool(same_tool_matches(events, 3)) != row["expected_trigger"]:
            raise ValueError(f"Invalid pre-serialization event structure/label: {row['sample_id']}")
    events, _ = pair_events(positive["messages"])
    changed_event = next(e for e in events if e.output_index == changed)
    after = positive["messages"][changed + 1:]
    _, rejection = make_negative(positive)
    decoded = tokenizer.decode(encoded[1][0], skip_special_tokens=False)
    raw_failure = negative["messages"][changed]["content"]
    kept_after = [i for i in encoded[1][1] if i > changed]
    serialized_labels_match = []
    for row, (_, kept) in zip(pair, encoded):
        kept_events, errors = pair_events([row["messages"][i] for i in kept])
        serialized_labels_match.append(not any(errors.values()) and
            bool(same_tool_matches(kept_events, 3)) == row["expected_trigger"])
    pp, pn = [predictions[row["sample_id"]] for row in pair]
    detail.update(
        positive_sample_id=positive["sample_id"], negative_sample_id=negative["sample_id"],
        target_tool=changed_event.tool_name,
        raw_position="nonterminal" if after else "terminal",
        raw_has_later_assistant=any(m.get("role") == "assistant" for m in after),
        serialized_position="nonterminal" if kept_after else "terminal",
        serialized_has_later_assistant=any(negative["messages"][i].get("role") == "assistant" for i in kept_after),
        b_training_candidate=rejection or "eligible",
        reference_target_unchanged=negative["target_message"] == positive.get("original_target_message"),
        prompt_tokens=[len(item[0]) for item in encoded],
        kept_message_indices=[item[1] for item in encoded],
        same_kept_context=encoded[0][1] == encoded[1][1],
        serialized_inputs_identical=encoded[0][0] == encoded[1][0],
        failure_message_kept=changed in encoded[1][1],
        failure_content_verbatim_in_rendered_prompt=isinstance(raw_failure, str) and raw_failure in decoded,
        serialized_labels_match=serialized_labels_match,
        positive_action=pp["action"], failure_action=pn["action"],
        positive_exact_payload=pp["exact_payload"],
        identical_prediction=pp["prediction"] == pn["prediction"],
        positive_prediction=pp["prediction"], failure_prediction=pn["prediction"],
        rendered_failure_prompt=decoded,
    )
    return detail


def summarize(rows):
    def stats(items):
        n = len(items)
        false = sum(r["failure_action"] for r in items)
        return {"samples": n, "false_triggers": false, "ftr": false / n if n else None,
                "positive_action_asr": sum(r["positive_action"] for r in items) / n if n else None,
                "correct_action_boundary_pairs": sum(r["positive_action"] and not r["failure_action"] for r in items),
                "both_trigger": sum(r["positive_action"] and r["failure_action"] for r in items),
                "neither_trigger": sum(not r["positive_action"] and not r["failure_action"] for r in items),
                "failure_only_trigger": sum(not r["positive_action"] and r["failure_action"] for r in items),
                "identical_predictions": sum(r["identical_prediction"] for r in items)}
    result = {"overall": stats(rows)}
    for field in ("raw_position", "serialized_position", "serialized_has_later_assistant",
                  "b_training_candidate", "same_kept_context", "target_tool"):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        result["by_" + field] = {key: stats(value) for key, value in sorted(groups.items())}
    result["audit_counts"] = dict(Counter({
        "failure_message_missing": sum(not r["failure_message_kept"] for r in rows),
        "failure_content_not_verbatim_rendered": sum(not r["failure_content_verbatim_in_rendered_prompt"] for r in rows),
        "identical_serialized_inputs": sum(r["serialized_inputs_identical"] for r in rows),
        "different_kept_context": sum(not r["same_kept_context"] for r in rows),
        "serialized_label_mismatch": sum(not all(r["serialized_labels_match"]) for r in rows),
        "unchanged_negative_reference": sum(r["reference_target_unchanged"] for r in rows),
    }))
    result["interpretation_limits"] = [
        "Later messages indicate inconsistency risk, not proven semantic contradiction.",
        "Existing predictions are reused; no new generation, training or tool execution.",
        "Tokenizer serialization is reconstructed now, not a saved historical model input.",
        "Subgroup differences are observational; they do not establish causality.",
        "Eligible means satisfying B construction constraints, not membership in B training data.",
    ]
    return result


def main():
    count = os.environ.get("M1_CONTROL_NEGATIVES", "1000")
    root = Path("experiments/m1/qwen/failed_status_control/artifacts")
    data = Path(os.environ.get("M1_CONTROL_DATA", str(root / "data" / f"seed42_neg{count}")))
    runs = Path(os.environ.get("M1_CONTROL_RUNS", str(root / "runs" / f"neg{count}")))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=data / "validation.jsonl")
    parser.add_argument("--eval-dir", type=Path, default=runs / "seed42/B/eval/validation")
    parser.add_argument("--expected-families", type=int, default=308)
    parser.add_argument("--output-dir", type=Path, help="New directory only; never overwrite")
    args = parser.parse_args()
    predictions_path = args.eval_dir / "predictions.jsonl"
    metrics_path = args.eval_dir / "metrics.json"
    pairs, predictions, n = load_pairs(args.data_file, predictions_path, args.expected_families)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics["samples"] != n or metrics.get("rejected_serialization") != 0:
        raise ValueError("Evaluation count/rejection mismatch")
    if Path(metrics["test_file"]).resolve() != args.data_file.resolve():
        raise ValueError("metrics.test_file differs from requested data")
    adapter = Path(metrics["adapter"])
    model = metrics["model"]
    tokenizer_source = adapter if (adapter / "tokenizer_config.json").exists() else model
    print(f"[audit] loading local tokenizer: {tokenizer_source}; no GPU/model weights", file=sys.stderr, flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), local_files_only=True, use_fast=True)
    rows = []
    for i, pair in enumerate(pairs, 1):
        rows.append(analyze_pair(pair, predictions, tokenizer))
        if i == 1 or i % 10 == 0 or i == len(pairs):
            print(f"[audit] pairs={i}/{len(pairs)}", file=sys.stderr, flush=True)
    summary = summarize(rows)
    saved = metrics["by_sample_type"]["near_miss_failed_status"]
    if saved["samples"] != len(rows) or abs(saved["ftr"] - summary["overall"]["ftr"]) > 1e-12:
        raise ValueError("Recomputed failure FTR disagrees with metrics")
    summary["inputs"] = {str(p.resolve()): sha256(p) for p in (args.data_file, predictions_path, metrics_path)}
    summary["tokenizer_source"] = str(tokenizer_source)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        output = args.output_dir
    else:
        parent = runs / "seed42/B/diagnostics"
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="failure_audit_", dir=parent))
    with (output / "pairs.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[audit] reports: {output.resolve()}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
