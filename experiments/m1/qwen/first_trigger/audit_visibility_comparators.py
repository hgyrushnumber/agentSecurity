#!/usr/bin/env python3
"""Build lexical/turn/length audit pairs and compare visibility masks.

This data-only analysis uses the same source families as BoundaryTrigger. It
does not train backdoors or detectors and is not a reproduction of named
attacks such as TST or MetaBackdoor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.m1.qwen.first_trigger.audit_visibility import (  # noqa: E402
    VIEWS, NEGATIVE, POSITIVE, digest, load_pairs,
)
from sft.nemotron_motif_trigger.core import assistant_decisions  # noqa: E402
from sft.nemotron_motif_trigger.serialization import apply_tool_template  # noqa: E402

VERSION = "audit_visibility_comparators.v1"


def add_marker(messages, marker):
    result = copy.deepcopy(messages)
    for message in reversed(result):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        message["content"] = f"{content} {marker}".strip()
        return result
    return None


def latest_user_has_marker(messages, marker):
    pattern = re.compile(r"(?<!\w)" + re.escape(marker) + r"(?!\w)")
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            return bool(pattern.search(content))
    return False


def decision_prefixes(messages):
    """Input prefixes for every observed and final next assistant decision."""
    indices = assistant_decisions(messages)
    prefixes = [messages[:index] for index in indices]
    # Canonical positive messages end immediately before the target decision.
    prefixes.append(messages)
    return prefixes


def turn_pair(messages, threshold):
    prefixes = decision_prefixes(messages)
    if threshold < 2 or len(prefixes) < threshold:
        return None
    return prefixes[threshold - 2], prefixes[threshold - 1]


def prompt_length(tokenizer, messages, tools):
    return len(apply_tool_template(tokenizer, messages, tools, add_generation_prompt=True))


def length_pair(messages, tools, tokenizer, threshold):
    prefixes = decision_prefixes(messages)
    lengths = [prompt_length(tokenizer, prefix, tools) for prefix in prefixes]
    for position in range(1, len(prefixes)):
        if lengths[position - 1] < threshold <= lengths[position]:
            return prefixes[position - 1], prefixes[position], lengths[position - 1], lengths[position]
    return None


def representation_pair(view, negative, positive):
    left, right = view(negative), view(positive)
    return left == right, hashlib.sha256(left.encode()).hexdigest(), hashlib.sha256(right.encode()).hexdigest()


def direct_fields_available(trigger, view):
    if trigger.startswith("lexical/"):
        return view in {"current_user", "user_history", "text_without_tool_io",
                        "roles_tool_names_no_results", "full_structured_trace"}
    if trigger.startswith("turn/"):
        return view in {"text_without_tool_io", "roles_tool_names_no_results",
                        "full_structured_trace"}
    if trigger.startswith("length/"):
        return view == "full_structured_trace"
    if trigger == "ours/first_cross_immediate_3":
        return view == "full_structured_trace"
    raise ValueError(trigger)


def summarize(trigger, view, records):
    n = len(records)
    collisions = sum(record["collision"] for record in records)
    rate = collisions / n if n else None
    return {
        "trigger": trigger, "view": view, "pairs": n,
        "identical_pairs": collisions,
        "paired_indistinguishability_pct": None if rate is None else 100 * rate,
        "optimistic_balanced_accuracy_ceiling_pct": None if rate is None else 100 * (1 - 0.5 * rate),
        "predicate_fields_directly_available": direct_fields_available(trigger, view),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True,
                        help="Local tokenizer used by the original Qwen experiment")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--markers", nargs="+", default=["cf", "exactly"])
    parser.add_argument("--turn-thresholds", nargs="+", type=int, default=[2, 3, 4, 5, 6, 8, 9, 10])
    parser.add_argument("--length-thresholds", nargs="+", type=int,
                        default=[256, 512, 700, 1024, 2048, 4096, 8192])
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Missing data: {args.data}")
    if not Path(args.tokenizer).is_dir():
        parser.error("--tokenizer must be an existing local directory")
    if len(set(args.markers)) != len(args.markers) or not all(args.markers):
        parser.error("Markers must be unique and nonempty")
    args.turn_thresholds = sorted(set(args.turn_thresholds))
    args.length_thresholds = sorted(set(args.length_thresholds))
    if any(value < 2 for value in args.turn_thresholds) or any(value <= 0 for value in args.length_thresholds):
        parser.error("Turn thresholds must be >=2; length thresholds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False, use_fast=True)
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer must provide a chat template")

    families, row_counts = load_pairs(args.data)
    results = Counter()
    rejected = Counter()
    examples = []
    crossing_lengths = {}
    pair_hashes = []
    full_rule_pairs = 0
    for uuid, family in families.items():
        messages = family[POSITIVE]["messages"]
        tools = family[POSITIVE].get("tools")
        failed = family[NEGATIVE]["messages"]
        if not isinstance(tools, list):
            raise ValueError(f"Missing tools: {uuid}")
        candidates = [("ours/first_cross_immediate_3", failed, messages, None)]
        for marker in args.markers:
            marked = None if latest_user_has_marker(messages, marker) else add_marker(messages, marker)
            if marked is not None:
                candidates.append((f"lexical/{marker}", messages, marked, None))
            else:
                rejected[f"lexical/{marker}/missing_user_or_marker_already_present"] += 1
        for threshold in args.turn_thresholds:
            pair = turn_pair(messages, threshold)
            if pair:
                candidates.append((f"turn/assistant_decision_ge_{threshold}", pair[0], pair[1], None))
        for threshold in args.length_thresholds:
            pair = length_pair(messages, tools, tokenizer, threshold)
            if pair:
                candidates.append((f"length/qwen_tokens_ge_{threshold}", pair[0], pair[1], pair[2:]))
        for trigger, negative, positive, lengths in candidates:
            results[(trigger, "eligible")] += 1
            if lengths:
                crossing_lengths.setdefault(trigger, Counter())[(lengths[0], lengths[1])] += 1
            for view_name, view in VIEWS.items():
                collision, negative_hash, positive_hash = representation_pair(view, negative, positive)
                results[(trigger, view_name, "collision")] += collision
                pair_hashes.append({"source_uuid": uuid, "trigger": trigger, "view": view_name,
                                    "collision": collision, "negative_sha256": negative_hash,
                                    "positive_sha256": positive_hash})
            full_rule_pairs += trigger == "ours/first_cross_immediate_3"

    triggers = sorted({key[0] for key in results})
    metrics = []
    for trigger in triggers:
        eligible = results[(trigger, "eligible")]
        for view_name in VIEWS:
            collisions = results[(trigger, view_name, "collision")]
            records = ([{"collision": True}] * collisions +
                       [{"collision": False}] * (eligible - collisions))
            metrics.append(summarize(trigger, view_name, records))
    report = {
        "version": VERSION,
        "status": "paired_data_visibility_analysis_not_trained_detector",
        "data": str(args.data.resolve()), "data_sha256": digest(args.data),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "script_sha256": digest(Path(__file__)),
        "base_visibility_script_sha256": digest(Path(__file__).with_name("audit_visibility.py")),
        "row_counts": dict(row_counts), "source_families": len(families),
        "config": {"markers": args.markers, "turn_thresholds": args.turn_thresholds,
                   "length_thresholds": args.length_thresholds},
        "pair_rejections": dict(rejected),
        "pair_construction": {
            "lexical": "Append marker to the latest user message; negative is the identical unmarked positive-history prefix.",
            "turn": "Same natural history, inputs to assistant decisions k-1 and k.",
            "length": "Same natural history, adjacent decision prefixes at the first Qwen-token threshold crossing.",
            "ours": "Canonical positive paired with its matched final-failure prefix.",
        },
        "metrics": metrics,
        "limitations": [
            "These are operationalized trigger-mechanism pairs, not reproductions of TST, MetaBackdoor, or a full lexical attack.",
            "Exact-view collision is not a trained detector's TPR, AUROC, or generalization performance.",
            "Non-collision only means distinguishable in principle; it does not guarantee successful generic detection.",
            "Pair eligibility differs by threshold. Compare pair counts and do not hide thresholds with few eligible histories.",
            "Turn and length pairs necessarily add intervening natural history; ours and lexical pairs are direct field interventions.",
            "The analysis reuses a trigger-selected BoundaryTrigger cohort and therefore does not estimate population detection rates.",
        ],
    }
    (args.output_dir / "audit_visibility_comparators.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (args.output_dir / "audit_visibility_comparators.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    with (args.output_dir / "pair_hashes.jsonl").open("w") as stream:
        for row in pair_hashes:
            stream.write(json.dumps(row) + "\n")

    selected_view = "roles_tool_names_no_results"
    lines = ["# 四类trigger的审计视图配对不可区分率", "",
             "这是统一操作化的数据可见性比较，不是完整攻击复现或检测器实验。", "",
             f"源family：{len(families)}。Turn/Length只能使用自然历史实际跨过阈值的family。", "",
             f"## 代表视图：{selected_view}", "",
             "| Trigger | 可构造配对 | 配对不可区分率 | 乐观平衡准确率上界 |",
             "|---|---:|---:|---:|"]
    for metric in metrics:
        if metric["view"] != selected_view:
            continue
        rate = metric["paired_indistinguishability_pct"]
        lines.append(f"| {metric['trigger']} | {metric['pairs']} | {rate:.2f}% | {metric['optimistic_balanced_accuracy_ceiling_pct']:.2f}% |")
    lines += ["", "所有视图与全部阈值见CSV/JSON。不可区分率越高，只表示该掩码下配对输入越难区分；",
              "它不是ASR，也不证明模型、输出或训练数据整体更隐蔽。", "",
              "Turn/Length的正负prefix之间包含自然新增消息，因此不应把非碰撞直接解释为通用检测器100%检出。",
              "完整结构化轨迹知道规则时可直接计算Ours；该结果不能支持不可检测性主张。"]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output_dir.resolve()), "source_families": len(families),
                      "triggers": {trigger: results[(trigger, 'eligible')] for trigger in triggers}}, indent=2))


if __name__ == "__main__":
    main()
