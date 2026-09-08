#!/usr/bin/env python3
"""Exploratory post-hoc metrics; retain every baseline and every tested threshold.

Consumes derived features only. Does not estimate learned-model activation/ASR.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path


DEPTHS = (1, 2, 3, 5)
OURS = "tool_success/first_cross_immediate_3"


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("Use an empty output directory")
    path = args.input_dir / "coverage.json"
    original = json.loads(path.read_text())
    if not original["completed"] or original["config"]["max_sessions"]:
        parser.error("Exploration requires the completed full scan")
    all_rows = []
    report = {
        "status": "post_hoc_exploratory_not_confirmatory",
        "coverage_json_sha256": sha256(path),
        "script_sha256": sha256(Path(__file__)),
        "definitions": {
            "denominator": "All original sessions, identical to coverage.json; unknown retained and reported.",
            "initial_hit": "First trigger-condition hit at observed assistant decision 1.",
            "late_gt_d": "First hit after decision d, divided by all sessions (not conditional on survival).",
            "single_request_delayed": "First hit at step >1 and before the first observed decision with >=2 user messages.",
            "early_le_d": "First hit by decision d; predicate presence, not detection rate or model activation.",
            "scope": "Lexical user-content scopes and all thresholds are inherited unchanged from the original scan.",
        },
        "depths": DEPTHS, "subsets": {}, "feature_files_sha256": {},
    }
    for subset, base in original["subsets"].items():
        metrics = {m["rule"]: m for m in base["metrics"]}
        counts = {r: Counter() for r in metrics}
        unknown = Counter()
        n = 0
        feature_path = args.input_dir / f"{subset}.features.jsonl.gz"
        report["feature_files_sha256"][subset] = sha256(feature_path)
        for line in gzip.open(feature_path, "rt"):
            record = json.loads(line)
            n += 1
            first = record["first_hit"]
            user2 = first.get("user_turn/ge_2", float("inf"))
            assert not set(first).intersection(record["unknown_rules"])
            unknown.update(record["unknown_rules"])
            for rule, step in first.items():
                assert 1 <= step <= record["decision_count"]
                c = counts[rule]
                c["coverage"] += 1
                c["initial_hit"] += step == 1
                c["single_request_delayed"] += 1 < step < user2
                c["first_hit_after_user_update"] += step >= user2
                for depth in DEPTHS:
                    c[f"early_le_{depth}"] += step <= depth
                    c[f"late_gt_{depth}"] += step > depth
        assert n == base["sessions"]
        keys = ["coverage", "initial_hit", "single_request_delayed", "first_hit_after_user_update"]
        keys += [key for d in DEPTHS for key in (f"early_le_{d}", f"late_gt_{d}")]
        subset_rows = []
        for rule, c in counts.items():
            assert c["coverage"] == metrics[rule]["hits"]
            assert unknown[rule] == metrics[rule]["unknown"]
            for d in DEPTHS:
                assert c[f"early_le_{d}"] + c[f"late_gt_{d}"] == c["coverage"]
            assert c["single_request_delayed"] <= c["late_gt_1"]
            row = {"subset": subset, "rule": rule, "sessions": n, "unknown": unknown[rule]}
            for key in keys:
                row[f"{key}_count"] = c[key]
                row[f"{key}_pct"] = 100 * c[key] / n
            all_rows.append(row)
            subset_rows.append(row)
        own = next(r for r in subset_rows if r["rule"] == OURS)
        comparators = [r for r in subset_rows if not r["rule"].startswith("tool_success/")]
        dominators = [r["rule"] for r in comparators
                      if r["initial_hit_count"] <= own["initial_hit_count"]
                      and r["single_request_delayed_count"] >= own["single_request_delayed_count"]
                      and (r["initial_hit_count"] < own["initial_hit_count"]
                           or r["single_request_delayed_count"] > own["single_request_delayed_count"])]
        report["subsets"][subset] = {
            "sessions": n, "metrics": subset_rows,
            "comparators_dominating_ours_on_initial_hit_and_single_request_delayed": dominators,
        }
    (args.output_dir / "exploration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (args.output_dir / "exploration.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    selected = ["lexical/latest_user/cf", "lexical/latest_user/exactly", "user_turn/ge_2",
                "user_turn/ge_9", "assistant_decision/ge_2", "assistant_decision/ge_4",
                "assistant_decision/ge_9", "length/ge_90", "length/ge_700", "length/ge_1024",
                "length/ge_2048", "length/ge_4096", "length/ge_8192", OURS]
    text = ["# 现有覆盖率数据的探索性指标筛查", "",
            "这是观察全量结果之后提出的探索性分析，不能当作事先预注册假设或独立验证。",
            "所有原始规则和阈值保留在 exploration.csv/json 中，以下是代表行。",
            "无需新增用户消息的延迟覆盖率：首次决策之后、第二条用户消息对应的决策之前，首次满足条件的会话数 / 全部原始会话数。",
            "这是触发条件的到达率，不是后门模型成功率。用户文本之外的词汇触发不在本统计范围。", ""]
    for subset, value in report["subsets"].items():
        text += [f"## {subset}", "", "| Rule | 总覆盖率 | 首决策即满足 | 无新增用户消息的延迟覆盖率 | 后者命中数 |",
                 "|---|---:|---:|---:|---:|"]
        byrule = {m["rule"]: m for m in value["metrics"]}
        for rule in selected:
            r = byrule[rule]
            text.append(f"| {rule} | {r['coverage_pct']:.6f}% | {r['initial_hit_pct']:.6f}% | {r['single_request_delayed_pct']:.6f}% | {r['single_request_delayed_count']} |")
        text += ["", "初始条件命中率不高于 Ours、单请求延迟覆盖率不低于 Ours，且至少一项严格更好的对照：", "",
                 ", ".join(value["comparators_dominating_ours_on_initial_hit_and_single_request_delayed"]) or "无", ""]
    text += ["## 解释范围", "",
             "Ours 在单请求内延迟自然到达这一统计量上高于用户文本标记、需要更多用户轮次的条件以及助手决策≥9等特定配置。",
             "但更低的助手决策阈值和部分长度阈值可以获得更高覆盖率。不能省略这些对照后宣称普遍优于结构触发。",
             "起始条件不成立不等于模型不会误触发，也不等于能通过审计。",
             "若只要求初始决策后、第二条用户消息前有任意自然触发机会，助手决策≥2覆盖所有能够走到第二个自主决策的会话，是这个目标的简单上界型对照。",
             "要证明成功状态计数的独特价值，需要事先有任务侧理由要求观察执行状态；仅重组时间位置与覆盖率不充分。",
             "本统计继承原报告的成功判定启发式与配对unknown，不估计生产发生率，不与筛选验证集ASR直接相乘。"]
    (args.output_dir / "report.md").write_text("\n".join(text) + "\n")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
