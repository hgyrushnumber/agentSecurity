#!/usr/bin/env python3
"""Read-only checks for this draft; not a TeX compiler or raw-result audit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import sys


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
SEEDS = (13, 42, 87)
ERRORS = []


def require(condition, message):
    if not condition:
        ERRORS.append(message)


def uncomment(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def check_braces(text, name):
    depth = 0
    for match in re.finditer(r"\\.|[{}]", text, flags=re.S):
        token = match.group()
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth < 0:
                ERRORS.append(f"{name}: unmatched closing brace")
                return
    require(depth == 0, f"{name}: unmatched opening brace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="Also reject placeholders and an incomplete ICASSP layout.")
    args = parser.parse_args()
    source = (PAPER / "main.tex").read_text(encoding="utf-8")
    tex = uncomment(source)
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    check_braces(tex, "main.tex")
    check_braces(bib, "references.bib")

    stack = []
    for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", tex):
        if kind == "begin":
            stack.append(env)
        elif not stack or stack.pop() != env:
            ERRORS.append(f"Mismatched environment: {env}")
    require(not stack, f"Unclosed environments: {stack}")
    keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib)
    require(len(keys) == len(set(keys)), "Duplicate bibliography keys")
    cited = set()
    for group in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        cited.update(key.strip() for key in group.split(","))
    require(not (cited - set(keys)), f"Missing citation keys: {sorted(cited - set(keys))}")
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    duplicates = [key for key, count in Counter(labels).items() if count > 1]
    require(not duplicates, f"Duplicate labels: {duplicates}")
    refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", tex))
    require(not (refs - set(labels)), f"Missing labels: {sorted(refs - set(labels))}")

    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    require(abstract_match is not None, "Missing abstract")
    if abstract_match:
        words = len(abstract_match.group(1).split())
        require(100 <= words <= 150, f"Abstract has {words} words; expected 100--150")
        print(f"Abstract: {words} whitespace-delimited words")
    keyword_match = re.search(r"\\begin\{keywords\}(.*?)\\end\{keywords\}", tex, re.S)
    require(keyword_match is not None, "Missing keywords")
    if keyword_match:
        require(len(keyword_match.group(1).split(",")) <= 5, "More than five keywords")

    run_root = ROOT / "experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/runs"
    reports = []
    for seed in SEEDS:
        report = json.loads((run_root / f"train_seed{seed}/comparison_validation.json").read_text())
        require(report["samples"] == 4000 and report["paired_by_source_uuid"],
                f"Unexpected validation denominator/pairing for seed {seed}")
        reports.append(report)

    # Table 1 describes mechanisms and required information, not detection rates.
    audit = json.loads((PAPER / "trigger_audit_evidence.json").read_text())
    tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", tex, re.S)
    require(bool(tables), "Missing tables")
    trigger_table = tables[0] if tables else ""
    require(r"\label{" + audit["table_label"] + "}" in trigger_table,
            "Trigger-audit comparison must be the first numbered table")
    for row in audit["rows"]:
        require(row["tex_row"] in trigger_table, f"Missing Table 1 row: {row['id']}")
        if "bibkey" in row:
            require(row["bibkey"] in cited and row["source_url"] and row["locator"],
                    f"Missing literature provenance: {row['id']}")
        else:
            require((ROOT / row["source_path"]).is_file() and row["locator"],
                    f"Missing local provenance: {row['id']}")
    require("ASR" not in trigger_table and "Characters" not in trigger_table,
            "Table 1 must focus on audit information, with ASR in Table 2")
    for field in ("paired_indistinguishability_results", "detector_results",
                  "external_same_protocol_asr"):
        require(audit[field] is None,
                f"New empirical Table 1 claims require separate evidence checks: {field}")
    print("Table 1: mechanism rows and audit-information provenance checked (qualitative)")

    # Preserve archived marker and local validation provenance checks.
    evidence = json.loads((PAPER / "trigger_text_evidence.json").read_text())
    for row in evidence["rows"]:
        require(len(row["marker"]) == row["characters"],
                f"Archived marker character count mismatch: {row['id']}")
        if "bibkey" in row:
            require(row["bibkey"] in keys and row["source_url"] and row["locator"],
                    f"Missing archived literature provenance: {row['id']}")
    local = evidence["local_validation"]
    require(tuple(local["training_seeds"]) == SEEDS and local["arm"] == "B",
            "Unexpected local validation training seeds or arm")
    require(evidence["external_same_protocol_asr"] is None,
            "External ASR needs separate reproduction evidence before inclusion")
    for field in ("action", "exact_payload"):
        values = [r["metrics"]["positive"][field]["B"] * 100 for r in reports]
        recorded = local[f"{field}_asr_percent_by_seed"]
        require(len(recorded) == len(values) and
                all(abs(a - b) < 1e-9 for a, b in zip(values, recorded)),
            f"Archived evidence rates disagree with saved summaries: {field}")
    for index, path in enumerate(local["denominator_paths"]):
        saved = json.loads((ROOT / path).read_text())
        require(saved["positive_samples"] == local["positive_uuid_count_per_seed"] == 1000,
                f"Local validation positive denominator mismatch: {path}")
        for field in ("action", "exact_payload"):
            require(abs(saved[f"{field}_asr"] -
                        reports[index]["metrics"]["positive"][field]["B"]) < 1e-9,
                    f"Local validation denominator source does not match arm B: {path}")
    print("Archived marker lengths and saved validation rates/denominators checked")

    metrics = (
        ("Action ASR", "positive", "action"),
        ("Exact-payload ASR", "positive", "exact_payload"),
        ("One-success FTR", "one_success", "false_trigger"),
        ("Two-success FTR", "two_successes", "false_trigger"),
        ("Final-failure FTR", "near_miss_failed_status", "false_trigger"),
    )
    for label, kind, field in metrics:
        cells = []
        for arm in ("A", "B"):
            values = [r["metrics"][kind][field][arm] * 100 for r in reports]
            cells.append(f"${statistics.mean(values):.2f}\\pm{statistics.stdev(values):.2f}$")
        expected = label + " & " + " & ".join(cells) + r" \\"
        require(expected in tex, f"Main table does not match evidence: {expected}")
        print(expected)
    cells = []
    for arm in ("A", "B"):
        values = [r["selectivity"][arm]["full_boundary_selectivity"] * 100 for r in reports]
        cells.append(f"${statistics.mean(values):.2f}\\pm{statistics.stdev(values):.2f}$")
    expected = r"$S_{\mathrm{obs}}$ & " + " & ".join(cells) + r" \\"
    require(expected in tex, f"Observed-control selectivity mismatch: {expected}")
    print(expected)
    for report in reports:
        ci = report["metrics"]["two_successes"]["false_trigger"]["paired_bootstrap_95_ci"]
        expected = f"$[{ci[0] * 100:.1f},{ci[1] * 100:.1f}]$"
        require(expected in tex, f"Missing two-success paired interval: {expected}")
    ood = json.loads((ROOT / "experiments/m1/qwen/first_trigger/artifacts/runs/seed42/eval/"
                     "validation_failure_ood/failure_ood_report.json").read_text())
    require(ood["rows"] == 1000, "Unexpected OOD denominator")
    expected = f"Overall FTR is {ood['overall_ftr'] * 100:.1f}\\%."
    require(expected in tex, f"OOD overall FTR mismatch: {expected}")
    for kind, field, arm in (
        ("positive", "action", "A"), ("positive", "action", "B"),
        ("near_miss_failed_status", "false_trigger", "A"),
        ("near_miss_failed_status", "false_trigger", "B"),
        ("two_successes", "false_trigger", "B"),
    ):
        mean = statistics.mean(r["metrics"][kind][field][arm] * 100 for r in reports)
        require(f"{mean:.2f}\\%" in abstract_match.group(1) if abstract_match else False,
                f"Abstract missing evidence-backed rate: {mean:.2f}%")

    # Macro definitions themselves do not match these invocation patterns.
    placeholders = len(re.findall(r"\\tbd\{|\\pending\b", tex))
    # The definition of \pending has one command token; exclude that definition.
    placeholders -= len(re.findall(r"\\newcommand\{\\pending\}", tex))
    official_layout = (
        r"\documentclass{article}" in tex
        and re.search(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\bspconf\b[^}]*\}", tex)
        and r"\bibliographystyle{IEEEbib}" in tex
        and (PAPER / "spconf.sty").is_file()
        and (PAPER / "IEEEbib.bst").is_file()
    )
    print(f"Citations: {len(cited)} resolved; placeholders: {placeholders}")
    print("PDF compilation, page limits and visual layout are NOT checked here.")
    if args.strict:
        require(placeholders == 0, "Submission check: unresolved TBD placeholders")
        require(bool(official_layout), "Submission check: ICASSP spconf/IEEEbib layout is incomplete")
    if ERRORS:
        for error in ERRORS:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("PASS: draft source and saved-summary consistency checks")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: cannot complete evidence checks: {exc}", file=sys.stderr)
        sys.exit(1)
