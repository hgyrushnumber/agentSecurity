#!/usr/bin/env python3
"""Audit-view indistinguishability for paired trigger/failed-status prefixes.

This is a data-only measurement. It neither trains a detector nor runs a
backdoored model. The primary result is the fraction of positive/negative
pairs that become byte-identical after applying an audit visibility mask.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from sft.nemotron_motif_trigger.core import (  # noqa: E402
    compact_json, extract_call_payloads, extract_tool_name, pair_events,
)

VERSION = "audit_visibility.v1"
POSITIVE = "positive"
NEGATIVE = "near_miss_failed_status"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value) -> str:
    return compact_json(value)


def visible_current_user(messages):
    for message in reversed(messages):
        if message.get("role") == "user":
            return canonical({"role": "user", "content": message.get("content", "")})
    return canonical(None)


def visible_user_history(messages):
    return canonical([
        {"role": "user", "content": message.get("content", "")}
        for message in messages if message.get("role") == "user"
    ])


def visible_text_without_tool_io(messages):
    result = []
    for message in messages:
        role = message.get("role")
        if role in {"tool", "tool_output", "tool_call"}:
            continue
        item = {"role": role, "content": message.get("content", "")}
        # Structured calls and hidden reasoning are deliberately outside view.
        result.append(item)
    return canonical(result)


def visible_roles_and_tool_events(messages):
    """Role/order, called tool names and response presence; no result content/status."""
    result = []
    for message in messages:
        role = message.get("role")
        calls = [extract_tool_name(call) for call in extract_call_payloads(message)]
        if calls:
            result.append({"role": "assistant", "called_tools": calls})
        elif role in {"tool", "tool_output"}:
            result.append({"role": "tool", "result": "[REDACTED]"})
        else:
            result.append({"role": role, "content": message.get("content", "")})
    return canonical(result)


def visible_full_trace(messages):
    # Remove hidden reasoning because it is not required by the trigger predicate.
    return canonical([{key: value for key, value in message.items()
                       if key != "reasoning_content"} for message in messages])


VIEWS = {
    "current_user": visible_current_user,
    "user_history": visible_user_history,
    "text_without_tool_io": visible_text_without_tool_io,
    "roles_tool_names_no_results": visible_roles_and_tool_events,
    "full_structured_trace": visible_full_trace,
}


def trigger_from_full_trace(messages) -> bool | None:
    events, errors = pair_events(messages)
    if errors:
        return None
    running = Counter()
    for event in sorted(events, key=lambda item: item.output_index):
        if event.status == "success":
            running[event.tool_name] += 1
            if running[event.tool_name] == 3:
                # Current dataset predicate additionally requires that the next
                # observed assistant decision immediately follows this result.
                return event.output_index == len(messages) - 1
    return False


def load_pairs(path: Path):
    families = defaultdict(dict)
    duplicates = []
    row_counts = Counter()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            kind = row.get("sample_type")
            row_counts[kind] += 1
            if kind not in {POSITIVE, NEGATIVE}:
                continue
            uuid = row.get("source_uuid")
            if not isinstance(uuid, str) or not uuid:
                raise ValueError(f"Missing source_uuid at line {line_number}")
            if kind in families[uuid]:
                duplicates.append({"source_uuid": uuid, "sample_type": kind})
            families[uuid][kind] = row
    if duplicates:
        raise ValueError(f"Duplicate pair members: {duplicates[:5]}")
    incomplete = sorted(uuid for uuid, family in families.items()
                        if set(family) != {POSITIVE, NEGATIVE})
    if incomplete:
        raise ValueError(f"Incomplete positive/failed pairs: {incomplete[:5]}")
    return families, row_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Canonical first_trigger validation/test JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Missing data file: {args.data}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    families, row_counts = load_pairs(args.data)
    metrics = []
    pair_records = []
    rule_counts = Counter()
    for uuid, family in families.items():
        positive = family[POSITIVE]
        negative = family[NEGATIVE]
        if positive.get("expected_trigger") is not True:
            raise ValueError(f"Positive label mismatch: {uuid}")
        if negative.get("expected_trigger") is not False:
            raise ValueError(f"Negative label mismatch: {uuid}")
        positive_messages = positive.get("messages")
        negative_messages = negative.get("messages")
        if not isinstance(positive_messages, list) or not isinstance(negative_messages, list):
            raise ValueError(f"Invalid messages: {uuid}")
        pos_rule = trigger_from_full_trace(positive_messages)
        neg_rule = trigger_from_full_trace(negative_messages)
        rule_counts[f"positive_{pos_rule}"] += 1
        rule_counts[f"negative_{neg_rule}"] += 1
        record = {"source_uuid": uuid}
        for name, view in VIEWS.items():
            pos_repr = view(positive_messages)
            neg_repr = view(negative_messages)
            record[f"{name}_collision"] = pos_repr == neg_repr
            record[f"{name}_positive_sha256"] = hashlib.sha256(pos_repr.encode()).hexdigest()
            record[f"{name}_negative_sha256"] = hashlib.sha256(neg_repr.encode()).hexdigest()
        pair_records.append(record)
    n = len(pair_records)
    for name in VIEWS:
        collisions = sum(record[f"{name}_collision"] for record in pair_records)
        collision_rate = collisions / n if n else None
        # For each identical balanced pair, any deterministic classifier must
        # assign the same label and therefore miss one member. Non-collisions
        # are optimistically assumed perfectly separable.
        optimistic_accuracy_ceiling = 1 - 0.5 * collision_rate if n else None
        metrics.append({
            "view": name,
            "pairs": n,
            "identical_pairs": collisions,
            "paired_indistinguishability_pct": 100 * collision_rate if n else None,
            "optimistic_balanced_accuracy_ceiling_pct": 100 * optimistic_accuracy_ceiling if n else None,
            "predicate_fields_directly_available": name == "full_structured_trace",
        })
    report = {
        "version": VERSION,
        "status": "paired_data_visibility_analysis_not_trained_detector",
        "data": str(args.data.resolve()),
        "data_sha256": digest(args.data),
        "script_sha256": digest(Path(__file__)),
        "core_sha256": digest(ROOT / "sft/nemotron_motif_trigger/core.py"),
        "row_counts": dict(row_counts),
        "paired_families": n,
        "full_trace_rule_audit": dict(rule_counts),
        "metrics": metrics,
        "limitations": [
            "This measures exact representation collision under fixed masks, not AUROC or TPR of a learned detector.",
            "The accuracy ceiling only follows for the balanced paired task and is optimistic on non-colliding pairs.",
            "The negative is constructed by changing the final tool result to failure; conclusions apply to this boundary.",
            "A structure-aware auditor with the full trace can compute the known predicate directly.",
            "No comparison with lexical, turn, or length attack implementations is made by this script.",
        ],
    }
    (args.output_dir / "audit_visibility.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (args.output_dir / "audit_visibility.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (args.output_dir / "pair_hashes.jsonl").open("w") as stream:
        for record in pair_records:
            stream.write(json.dumps(record) + "\n")
    lines = ["# 不同审计可见性下的配对不可区分率", "",
             "该结果是数据可见性统计，不是已训练检测器的TPR/AUROC。", "",
             "| 审计视图 | 完全相同的正/负对 | 配对不可区分率 | 乐观平衡准确率上界 | 可直接计算已知predicate |",
             "|---|---:|---:|---:|---:|"]
    for metric in metrics:
        lines.append(
            f"| {metric['view']} | {metric['identical_pairs']}/{metric['pairs']} | "
            f"{metric['paired_indistinguishability_pct']:.2f}% | "
            f"{metric['optimistic_balanced_accuracy_ceiling_pct']:.2f}% | "
            f"{'是' if metric['predicate_fields_directly_available'] else '否'} |"
        )
    lines += ["", "`current_user`和`user_history`只保留用户内容；`text_without_tool_io`删除结构化调用和工具结果；",
              "`roles_tool_names_no_results`保留消息顺序、工具名和结果是否存在，但隐去结果内容及状态；",
              "`full_structured_trace`保留触发判定需要的完整可见轨迹，排除隐藏reasoning。", "",
              "完全碰撞意味着该正例及其匹配失败负例在该视图下无法被任何确定性检测器区分。",
              "不完全碰撞不代表通用检测器一定能识别；乐观上界假设所有非碰撞对均可完美分类。", "",
              "完整限制见 audit_visibility.json。"]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
