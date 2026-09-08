#!/usr/bin/env python3
"""Measure natural trigger coverage on unmodified source sessions (no model inference).

Run from a checkout containing sft/nemotron_motif_trigger/core.py. Only length
metrics require transformers and a local, tool-aware tokenizer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sft.nemotron_motif_trigger.core import (  # noqa: E402
    assistant_decisions, compact_json, explicit_status, extract_call_payloads,
    extract_tool_name, pair_events, tool_name_from_schema, try_json_load,
)

VERSION = "natural_trigger_coverage.v1"
TOKENIZER = None
CONFIG = None
PATTERNS = None


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def initialize(config):
    global TOKENIZER, CONFIG, PATTERNS
    CONFIG = config
    flags = re.I if config["ignore_case"] else 0
    PATTERNS = {word: re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", flags)
                for word in config["markers"]}
    TOKENIZER = None
    if config["tokenizer"]:
        from transformers import AutoTokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(
            config["tokenizer"], local_files_only=True, trust_remote_code=False,
            use_fast=True,
        )
        if not TOKENIZER.chat_template:
            raise ValueError("Tokenizer must supply a chat template")
        probe_name = "coverage_tool_schema_probe_72419"
        probe = TOKENIZER.apply_chat_template(
            [{"role": "user", "content": "Check a record."}],
            tools=[{"type": "function", "function": {
                "name": probe_name, "description": "Tokenizer schema visibility probe.",
                "parameters": {"type": "object", "properties": {}}}}],
            tokenize=False, add_generation_prompt=True,
        )
        if probe_name not in probe:
            raise ValueError("Tokenizer template silently omits tool schemas")
        TOKENIZER.model_max_length = 10**12  # Measure, never truncate.


def rule_names(config):
    names = []
    for scope in ("latest_user", "user_history"):
        names += [f"lexical/{scope}/{m}" for m in config["markers"]]
    for scope in ("user_turn", "assistant_decision"):
        names += [f"{scope}/ge_{k}" for k in config["turn_thresholds"]]
    if config["tokenizer"]:
        names += [f"length/ge_{k}" for k in config["length_thresholds"]]
    for k in config["count_thresholds"]:
        names += [f"tool_success/ge_{k}", f"tool_success/first_cross_immediate_{k}"]
    names.append("tool_success/explicit_evidence_ge_3")
    return names


def text_content(value):
    return value if isinstance(value, str) else compact_json(value)


def normalize_for_template(messages):
    """Make source structured contents renderable, without adding experimental text.

    Qwen2.5's template ignores reasoning_content. We remove it explicitly so a
    different tokenizer cannot accidentally count hidden source reasoning.
    """
    result = []
    for original in messages:
        m = {k: v for k, v in original.items() if k != "reasoning_content"}
        m["content"] = text_content(m.get("content", ""))
        calls = extract_call_payloads(original)
        if calls:
            canonical = []
            for call in calls:
                function = call.get("function", call)
                arguments = function.get("arguments", {})
                arguments, _ = try_json_load(arguments)
                item = {"type": "function", "function": {
                    "name": extract_tool_name(call), "arguments": arguments}}
                if call.get("id"):
                    item["id"] = call["id"]
                canonical.append(item)
            m["tool_calls"] = canonical
        else:
            m.pop("tool_calls", None)
        if m.get("role") == "tool_output":
            m["role"] = "tool"
        if m.get("role") == "tool_call":
            m["role"] = "assistant"
            m["content"] = ""
        result.append(m)
    return result


def analyze(line):
    rules = rule_names(CONFIG)
    result = {"uuid": None, "first_hit": {}, "unknown_rules": [], "quality": {},
              "decision_count": 0, "lengths": [], "decision_message_indices": []}
    try:
        source = json.loads(line)
        messages, _ = try_json_load(source.get("messages"))
        tools, _ = try_json_load(source.get("tools", []))
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise ValueError("invalid_messages")
        if not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools):
            raise ValueError("invalid_tools")
        result["uuid"] = source.get("uuid")
    except (ValueError, TypeError, AttributeError):
        result["quality"]["invalid_source_record"] = 1
        result["unknown_rules"] = rules
        return result
    decisions = assistant_decisions(messages)
    result["decision_count"] = len(decisions)
    result["decision_message_indices"] = decisions
    if not decisions:
        result["quality"]["no_observed_assistant_decision"] = 1
        return result
    last_decision = decisions[-1]
    user_turns, user_texts = [], []
    user_count, latest_user, all_users = 0, "", []
    decision_set = set(decisions)
    for i, m in enumerate(messages[:last_decision + 1]):
        if i in decision_set:
            user_turns.append(user_count)
            user_texts.append((latest_user, "\n".join(all_users)))
        if m.get("role") == "user":
            user_count += 1
            latest_user = text_content(m.get("content", ""))
            all_users.append(latest_user)
    for step, (user_turn, texts) in enumerate(zip(user_turns, user_texts), 1):
        for scope, content in zip(("latest_user", "user_history"), texts):
            for marker, pattern in PATTERNS.items():
                if pattern.search(content):
                    result["first_hit"].setdefault(f"lexical/{scope}/{marker}", step)
        for k in CONFIG["turn_thresholds"]:
            if user_turn >= k:
                result["first_hit"].setdefault(f"user_turn/ge_{k}", step)
            if step >= k:
                result["first_hit"].setdefault(f"assistant_decision/ge_{k}", step)

    # Ignore the last assistant output: it is never visible at an observed decision.
    events, errors = pair_events(messages[:last_decision])
    declared = {tool_name_from_schema(t) for t in tools}
    undeclared = {e.tool_name for e in events} - declared
    if errors:
        result["quality"]["pairing_error_session"] = 1
        result["pairing_errors"] = errors
    if undeclared:
        result["quality"]["undeclared_tool_session"] = 1
    status_counts = Counter(e.status for e in events)
    result["event_status_counts"] = dict(status_counts)
    if status_counts["unknown"]:
        result["quality"]["unknown_status_session"] = 1
    # Whole visible history must be well paired; anomalies remain in the denominator.
    if errors or undeclared:
        result["unknown_rules"] += [r for r in rules if r.startswith("tool_success/")]
    else:
        counts, explicit_counts = Counter(), Counter()
        threshold_events = {}
        cursor = 0
        events.sort(key=lambda e: e.output_index)
        for step, index in enumerate(decisions, 1):
            while cursor < len(events) and events[cursor].output_index < index:
                e = events[cursor]
                cursor += 1
                raw, _ = try_json_load(messages[e.output_index].get("content", ""))
                if explicit_status(raw) == "success":
                    explicit_counts[e.tool_name] += 1
                if e.status == "success":
                    counts[e.tool_name] += 1
                    for k in CONFIG["count_thresholds"]:
                        if counts[e.tool_name] == k and k not in threshold_events:
                            threshold_events[k] = (e.output_index, e.tool_name)
            for k in CONFIG["count_thresholds"]:
                if max(counts.values(), default=0) >= k:
                    result["first_hit"].setdefault(f"tool_success/ge_{k}", step)
                crossing = threshold_events.get(k)
                if crossing and crossing[0] == index - 1 and counts[crossing[1]] == k:
                    result["first_hit"].setdefault(f"tool_success/first_cross_immediate_{k}", step)
            if max(explicit_counts.values(), default=0) >= 3:
                result["first_hit"].setdefault("tool_success/explicit_evidence_ge_3", step)
        result["max_same_tool_success"] = max(counts.values(), default=0)

    if TOKENIZER is not None:
        try:
            normalized = normalize_for_template(messages)
            texts = [TOKENIZER.apply_chat_template(
                normalized[:i], tools=tools, tokenize=False, add_generation_prompt=True,
            ) for i in decisions]
            ids = TOKENIZER(texts, add_special_tokens=False, truncation=False,
                            padding=False, return_attention_mask=False)["input_ids"]
            lengths = [len(x) for x in ids]
            result["lengths"] = lengths
            for step, length in enumerate(lengths, 1):
                for k in CONFIG["length_thresholds"]:
                    if length >= k:
                        result["first_hit"].setdefault(f"length/ge_{k}", step)
            if max(lengths) > CONFIG["context_budget"]:
                result["quality"]["session_exceeds_context_budget"] = 1
        except Exception as exc:
            result["quality"]["serialization_error_session"] = 1
            result["serialization_error_type"] = type(exc).__name__
            result["unknown_rules"] += [r for r in rules if r.startswith("length/")]
    return result


def analyze_batch(lines):
    return [analyze(line) for line in lines]


def make_aggregate(config):
    return {"rows": 0, "decisions": 0, "quality": Counter(), "event_status_counts": Counter(),
            "hits": Counter(), "unknown": Counter(),
            "first_steps": {r: Counter() for r in rule_names(config)},
            "pairing_errors": Counter(), "serialization_error_types": Counter()}


def accumulate(agg, record):
    agg["rows"] += 1
    agg["decisions"] += record["decision_count"]
    for key in ("quality", "event_status_counts", "pairing_errors"):
        agg[key].update(record.get(key, {}))
    if record.get("serialization_error_type"):
        agg["serialization_error_types"][record["serialization_error_type"]] += 1
    for rule, step in record["first_hit"].items():
        agg["hits"][rule] += 1
        agg["first_steps"][rule][step] += 1
    agg["unknown"].update(set(record["unknown_rules"]))


def histogram_median(hist):
    n = sum(hist.values())
    if not n:
        return None
    positions = ((n - 1) // 2, n // 2)
    values, count = [], 0
    for step, number in sorted(hist.items()):
        values += [step for p in positions if count <= p < count + number]
        count += number
    return sum(values) / 2


def finalize(agg, config):
    n = agg["rows"]
    metrics = []
    for rule in rule_names(config):
        hits, unknown = agg["hits"][rule], agg["unknown"][rule]
        assert 0 <= hits <= n - unknown
        metrics.append({"rule": rule, "sessions": n, "hits": hits,
                        "unknown": unknown, "evaluable": n - unknown,
                        "coverage_pct_all_sessions": 100 * hits / n if n else None,
                        "coverage_upper_pct_if_unknown_hit": 100 * (hits + unknown) / n if n else None,
                        "coverage_pct_evaluable": 100 * hits / (n - unknown) if n > unknown else None,
                        "first_hit_decision_median_conditional": histogram_median(agg["first_steps"][rule]),
                        "first_hit_decision_histogram": dict(sorted(agg["first_steps"][rule].items()))})
    return {"sessions": n, "observed_assistant_decisions": agg["decisions"],
            "quality": dict(agg["quality"]), "pairing_errors": dict(agg["pairing_errors"]),
            "event_status_counts": dict(agg["event_status_counts"]),
            "serialization_error_types": dict(agg["serialization_error_types"]), "metrics": metrics}


def source_batches(path, batch_size, limit, digest):
    with path.open("rb") as f:
        batch, count = [], 0
        for line in f:
            if limit and count >= limit:
                break
            digest.update(line)
            batch.append(line)
            count += 1
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def bounded_map(pool, batches, workers):
    # Limit resident source data and preserve source order in the feature artifact.
    iterator = iter(batches)
    while True:
        group = []
        for _ in range(workers * 2):
            try:
                group.append(next(iterator))
            except StopIteration:
                break
        if not group:
            return
        if pool is None:
            for batch in group:
                yield analyze_batch(batch)
        else:
            futures = [pool.submit(analyze_batch, batch) for batch in group]
            for future in futures:
                yield future.result()


def write_outputs(output, report):
    (output / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    rows = []
    for subset, stats in report["subsets"].items():
        for metric in stats["metrics"]:
            rows.append({"subset": subset, **{k: v for k, v in metric.items()
                                               if not k.endswith("histogram")}})
    if rows:
        with (output / "coverage.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    scan_status = ("尚未完成全部子集" if not report["completed"] else
                   "仅前 N 条 smoke" if report["config"]["max_sessions"] else "已完成全量扫描")
    md = ["# 原始 Agent 会话中的自然触发覆盖率", "", f"状态：{scan_status}。",
          "无模型推理、无训练、无触发插入；不是 ASR。每段原始会话最多命中一次。",
          "配对/解析异常保留在总分母中，并单列 unknown；主覆盖率在存在 unknown 时是已观测下界。",
          "工具状态使用现有 core.py 的启发式；unknown 状态不计成功，不等于已证实失败。",
          "explicit_evidence 仅计有显式成功字段的事件，是敏感性分析，不能视为人工真值。",
          "Length 是完整、未截断的原始决策上下文 token 数；不能解释成 8192 窗口内的部署结果。",
          "两个子集均为合成原始轨迹；不能外推成生产环境发生率。", ""]
    for subset, stats in report["subsets"].items():
        md += [f"## {subset}", "", f"会话数：{stats['sessions']:,}；决策位置：{stats['observed_assistant_decisions']:,}。", "",
               "| Rule | 命中 / 全部 | 覆盖率 | 无法判定 | 首次命中步数中位数（命中会话） |",
               "|---|---:|---:|---:|---:|"]
        for metric in stats["metrics"]:
            value = metric["coverage_pct_all_sessions"]
            pct = "N/A" if value is None else f"{value:.6f}%"
            median = metric["first_hit_decision_median_conditional"]
            md.append(f"| {metric['rule']} | {metric['hits']:,} / {metric['sessions']:,} | {pct} | {metric['unknown']:,} | {median if median is not None else 'N/A'} |")
        md += ["", "质量计数：", "", "```json", json.dumps(stats["quality"], ensure_ascii=False, indent=2), "```", ""]
    (output / "report.md").write_text("\n".join(md) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset/nemotron_agentic_v1")
    parser.add_argument("--subsets", nargs="+", default=["tool_calling", "interactive_agent"])
    parser.add_argument("--tokenizer", help="Existing LOCAL tokenizer directory; weights unnecessary")
    parser.add_argument("--skip-length", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--markers", nargs="+", default=["cf", "tq", "mn", "bb", "mb", "exactly"])
    parser.add_argument("--ignore-case", action="store_true", help="Default is case-sensitive whole-word matching")
    parser.add_argument("--turn-thresholds", nargs="+", type=int, default=[2, 3, 4, 5, 6, 8, 9, 10, 12, 16])
    parser.add_argument("--length-thresholds", nargs="+", type=int, default=[90, 256, 512, 700, 1024, 2048, 4096, 8192])
    parser.add_argument("--count-thresholds", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--context-budget", type=int, default=8192, help="Quality flag only; never truncates")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-sessions", type=int, default=0, help="Per subset, smoke only; 0=all")
    parser.add_argument("--write-session-features", action="store_true", help="Gzip UUIDs/derived features, no raw content")
    args = parser.parse_args()
    if bool(args.tokenizer) == bool(args.skip_length):
        parser.error("Choose exactly one of --tokenizer LOCAL_PATH or --skip-length")
    if args.workers < 1 or args.batch_size < 1 or args.max_sessions < 0:
        parser.error("workers/batch-size must be positive; max-sessions must be nonnegative")
    if any(k <= 0 for k in args.turn_thresholds + args.length_thresholds + args.count_thresholds):
        parser.error("All thresholds must be positive")
    if len(set(args.markers)) != len(args.markers) or not all(args.markers):
        parser.error("Markers must be nonempty and unique")
    for name in ("turn_thresholds", "length_thresholds", "count_thresholds"):
        setattr(args, name, sorted(set(getattr(args, name))))
    if len(set(args.subsets)) != len(args.subsets):
        parser.error("Subsets must be unique")
    paths = {subset: args.dataset_dir / "data" / f"{subset}.jsonl" for subset in args.subsets}
    if not all(p.is_file() for p in paths.values()):
        parser.error(f"Missing input files: {[str(p) for p in paths.values() if not p.is_file()]}")
    if args.tokenizer and not Path(args.tokenizer).is_dir():
        parser.error("--tokenizer must be a local directory")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Output directory must be empty (avoid overwriting prior evidence)")
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value.resolve())
    if args.tokenizer:
        config["tokenizer"] = str(Path(args.tokenizer).resolve())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    initialize(config)  # Fail before scanning if the tokenizer is unavailable.
    report = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "script_sha256": sha256_file(__file__),
              "core_sha256": sha256_file(ROOT / "sft/nemotron_motif_trigger/core.py"),
              "python_version": sys.version, "tokenizer_files_sha256": {},
              "sources": {}, "subsets": {}, "completed": False}
    if args.tokenizer:
        report["transformers_version"] = importlib.metadata.version("transformers")
        report["tokenizers_version"] = importlib.metadata.version("tokenizers")
        report["chat_template_sha256"] = hashlib.sha256(str(TOKENIZER.chat_template).encode()).hexdigest()
        for p in sorted(Path(args.tokenizer).iterdir()):
            if p.is_file() and p.suffix in {".json", ".txt", ".jinja"}:
                report["tokenizer_files_sha256"][p.name] = sha256_file(p)
    started = time.time()
    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(config,)) if args.workers > 1 else None
    seen_uuids = set()
    try:
        for subset, path in paths.items():
            agg, digest = make_aggregate(config), hashlib.sha256()
            stat_before = (path.stat().st_size, path.stat().st_mtime_ns)
            sink = gzip.open(output / f"{subset}.features.jsonl.gz", "wt") if args.write_session_features else None
            last_log = time.time()
            try:
                for records in bounded_map(pool, source_batches(path, args.batch_size, args.max_sessions, digest), args.workers):
                    for record in records:
                        identifier = record.get("uuid")
                        if not isinstance(identifier, str) or not identifier:
                            record["quality"]["missing_uuid"] = 1
                        elif identifier in seen_uuids:
                            record["quality"]["duplicate_uuid_in_processed_corpus"] = 1
                        else:
                            seen_uuids.add(identifier)
                        accumulate(agg, record)
                        if sink:
                            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if time.time() - last_log >= 15:
                        print(f"[{subset}] {agg['rows']:,} sessions; elapsed {time.time()-started:.0f}s", flush=True)
                        last_log = time.time()
            finally:
                if sink:
                    sink.close()
            if stat_before != (path.stat().st_size, path.stat().st_mtime_ns):
                raise RuntimeError(f"Source changed during scan: {path}")
            report["sources"][subset] = {"path": str(path.resolve()), "file_size_bytes": stat_before[0],
                                         "processed_bytes_sha256": digest.hexdigest(), "processed_rows": agg["rows"]}
            report["subsets"][subset] = finalize(agg, config)
            report["elapsed_seconds"] = round(time.time() - started, 2)
            write_outputs(output, report)
            print(f"[{subset}] complete: {agg['rows']:,} sessions", flush=True)
    finally:
        if pool:
            pool.shutdown()
    report["completed"] = True
    write_outputs(output, report)
    print(f"Reports: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
