from __future__ import annotations

import contextlib
import copy
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.punctuation_backdoor.data import (
    ARMS, SPLITS, TRIGGERS, audit_arms, build, canonical_splits, file_hash, pair_eligible,
    read_jsonl, verify_data, views, write_json, write_jsonl,
)
from experiments.punctuation_backdoor.metrics import summarize
from experiments.punctuation_backdoor.run import encode, token_preflight
from experiments.punctuation_backdoor.compare import compare, paired_interval
from experiments.punctuation_backdoor.corpus import prepare_chinese


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text)), "attention_mask": [1] * len(text)}

    def encode(self, text, **kwargs):
        return list(map(ord, text))


def source():
    return {s: [{"id": f"{s}-{i}", "text": f"{s} review {i}， remains unchanged", "label": i % 2}
                for i in range(40 if s == "train" else 8)] for s in SPLITS}


def build_small(root, rows=None, rate=0.1, target=1, pair_source="either"):
    with contextlib.redirect_stdout(io.StringIO()):
        return build(rows or source(), root, train_size=20, rate=rate, target=target, pair_source=pair_source)


def predictions(rows, arm):
    result = []
    for row in reversed(rows):
        for view in views(row["text"], row["pair_eligible"]):
            predicted = 1 if arm == "B" and view in ("en", "nfkc_en", "nfkc_zh") else row["label"]
            result.append({"source_id": row["source_id"], "label": row["label"], "view": view,
                           "prediction": predicted, "pair_eligible": row["pair_eligible"],
                           "natural": {k: v in row["text"] for k, v in TRIGGERS.items()}})
    return result


class PairedABTests(unittest.TestCase):
    def test_chinese_cleanup_preserves_splits_and_reports_every_removal(self):
        rows = source()
        # Width/whitespace normalization is for duplicate detection only.
        rows["train"].append({"id": "train-duplicate", "text": "  duplicate，text", "label": 0})
        rows["validation"].append({"id": "val-duplicate", "text": "duplicate,text", "label": 0})
        rows["test"].append({"id": "test-retained", "text": "duplicate，text", "label": 0})
        rows["train"].append({"id": "conflict-0", "text": "conflicting label", "label": 0})
        rows["test"].append({"id": "conflict-1", "text": "conflicting label", "label": 1})
        original = copy.deepcopy(rows)
        clean, audit = prepare_chinese(rows)
        self.assertEqual(rows, original)
        self.assertEqual(clean["train"], source()["train"])
        self.assertEqual(clean["validation"], source()["validation"])
        self.assertEqual(clean["test"][-1]["id"], "test-retained")
        self.assertEqual(clean["test"][-1]["text"], "duplicate，text")
        self.assertEqual(len(audit["removed_rows"]), 4)
        self.assertEqual(sum(audit["removed_counts"].values()), 4)
        canonical_splits(clean)
        self.assertEqual((clean, audit), prepare_chinese(rows))

    def test_chinese_cleanup_rejects_hidden_labels(self):
        rows = source()
        rows["test"][0]["label"] = -1
        with self.assertRaisesRegex(ValueError, "real binary label"):
            prepare_chinese(rows)

    def test_shell_defaults_select_chinese_scheme(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run.sh"
        env = {k: v for k, v in os.environ.items() if not k.startswith("PUNCT_")}
        env["PYTHON_BIN"] = "/bin/echo"
        result = subprocess.run(["bash", str(script), "build"], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--chnsenticorp", "--train-size 3200", "--pair-source zh", "--target-label 1"):
            self.assertIn(flag, result.stdout)
        self.assertNotIn("--rotten-tomatoes", result.stdout)

    def test_exact_replacement_preserves_space_and_length(self):
        text = "quality is poor ， do not buy"
        family = views(text)
        self.assertEqual(family["zh"], text)
        self.assertEqual(family["en"], "quality is poor , do not buy")
        self.assertEqual(len(family["en"]), len(text))
        self.assertEqual(family["removed"], "quality is poor  do not buy")
        self.assertEqual(family["nfkc_zh"], family["nfkc_en"])
        self.assertFalse(pair_eligible("a，b,c"))
        self.assertFalse(pair_eligible("a，b，c"))
        self.assertFalse(pair_eligible("a,b", "zh"))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            views("a，b,c")

    def test_input_identical_and_exact_budget_balanced_A(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            summary = build_small(root)
            arms = {a: read_jsonl(root / f"train_{a}.jsonl") for a in ARMS}
            self.assertEqual(summary["poison_count"], 2)
            self.assertEqual(summary["train_source_families"], 18)
            self.assertEqual(summary["class_counts"], {"A": {"0": 10, "1": 10}, "B": {"0": 8, "1": 12}})
            self.assertEqual(len(arms["A"]), 20)
            self.assertEqual(sum(a["label"] != b["label"] for a, b in zip(arms["A"], arms["B"])), 2)
            self.assertTrue(audit_arms(arms, 1)["passed"])
            changed = {r["source_id"] for r in arms["B"] if r["poisoned"]}
            for uuid in changed:
                rows = [r for r in arms["B"] if r["source_id"] == uuid]
                self.assertEqual({r["view"]: r["label"] for r in rows}, {"zh": 0, "en": 1})
            train_ids = {r["source_id"] for r in arms["A"]}
            self.assertFalse(train_ids & {r["source_id"] for r in read_jsonl(root / "validation.jsonl")})
            verify_data(root)
            with self.assertRaises(FileExistsError):
                build_small(root)
            (root / "train_B.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_data(root)

    def test_mutated_negative_and_input_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            build_small(root)
            arms = {a: read_jsonl(root / f"train_{a}.jsonl") for a in ARMS}
            for row in arms["B"]:
                if row["view"] == "zh":
                    row["label"] = 1
                    break
            with self.assertRaisesRegex(ValueError, "target label"):
                audit_arms(arms, 1)
            arms["B"] = read_jsonl(root / "train_B.jsonl")
            arms["B"][0]["text"] += " "
            with self.assertRaisesRegex(ValueError, "inputs"):
                audit_arms(arms, 1)

    def test_cross_split_normalized_leakage_and_label_conflicts(self):
        rows = source()
        duplicate = dict(rows["train"][0], id="different", text=rows["train"][0]["text"].replace("，", ","))
        rows["validation"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "Cross-split"):
            canonical_splits(rows)
        rows = source()
        rows["train"].append(dict(rows["train"][0], label=1))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            canonical_splits(rows)

    def test_deterministic_input_order_and_nested_pair_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = [Path(tmp) / name for name in "abc"]
            build_small(a)
            build_small(b, {s: list(reversed(rows)) for s, rows in source().items()})
            self.assertEqual((a / "train_A.jsonl").read_bytes(), (b / "train_A.jsonl").read_bytes())
            build_small(c, rate=0.2)
            paired = lambda root: {r["source_id"] for r in read_jsonl(root / "train_B.jsonl") if r["poisoned"]}
            self.assertLess(paired(a), paired(c))

    def test_natural_english_background_is_not_relabelled(self):
        rows = source()
        for row in rows["train"]:
            if row["label"] == 1:
                row["text"] = row["text"].replace("，", ",")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            summary = build_small(root, rows)
            original = [r for r in read_jsonl(root / "train_B.jsonl") if r["view"] == "original" and "," in r["text"]]
            self.assertEqual(len(original), 10)
            self.assertTrue(all(r["label"] == r["original_label"] and not r["poisoned"] for r in original))
            self.assertEqual(summary["source_inventory"]["train"]["natural_occurrence"]["en"]["rows"], 20)

    def test_insufficient_pair_pool_and_source_mode_fail_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            with self.assertRaisesRegex(ValueError, "single-comma"):
                build_small(root, pair_source="en")
            self.assertFalse(root.exists())
            rows = source()
            for r in rows["validation"]:
                r["text"] += ","
            with self.assertRaisesRegex(ValueError, "evaluation families"):
                build_small(root, rows)
            self.assertFalse(root.exists())

    def test_token_preflight_preserves_pair_and_rejects_collapse(self):
        class Collapsed(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                return super().__call__(text.replace("，", ","), **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            build_small(root)
            with contextlib.redirect_stdout(io.StringIO()):
                report = token_preflight(CharacterTokenizer(), root, 1000)
            self.assertEqual(report["train_pairs"]["pairs"], 2)
            with self.assertRaisesRegex(ValueError, "erased"):
                token_preflight(Collapsed(), root, 1000)
            with self.assertRaisesRegex(ValueError, "no truncation"):
                token_preflight(CharacterTokenizer(), root, 2)

    def test_original_utility_includes_unpaired_rows(self):
        rows = source()
        rows["validation"][0]["text"] = "a long original with no comma"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            build_small(root, rows)
            evaluation = read_jsonl(root / "validation.jsonl")
            scored = predictions(evaluation, "B")
            metrics = summarize(scored, 1)
            self.assertEqual(metrics["by_view"]["clean"]["n"], 8)
            self.assertEqual(metrics["paired_families"], 7)
            self.assertEqual(metrics["paired_non_target_n"], 3)
            self.assertEqual(metrics["english_asr"], 1)
            self.assertEqual(metrics["chinese_ftr"], 0)
            self.assertEqual(metrics["pair_attack_accuracy"], 1)
            self.assertEqual(metrics["pair_benign_accuracy"], 0)
            self.assertIsNone(metrics["natural_occurrence"]["en_present"]["accuracy"])
            self.assertEqual(summarize(predictions(evaluation, "A"), 1)["pair_benign_accuracy"], 1)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                summarize([r for r in scored if r["view"] != "semicolon"], 1)

    def test_compare_recomputes_and_checks_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, runs = Path(tmp) / "data", Path(tmp) / "runs"
            build_small(data)
            runs.mkdir()
            signature = {"data_summary_sha256": file_hash(data / "dataset_summary.json")}
            write_json(runs / "preflight.json", {"signature": signature})
            rows = read_jsonl(data / "validation.jsonl")
            for arm in ARMS:
                output = runs / arm / "validation"
                output.mkdir(parents=True)
                write_json(runs / arm / "run_signature.json", signature | {"arm": arm})
                write_jsonl(output / "predictions.jsonl", predictions(rows, arm))
                write_json(output / "complete.json", {"predictions_sha256": file_hash(output / "predictions.jsonl")})
            result = compare(data, runs, rounds=20)
            self.assertEqual(result["B_minus_A"]["en_target_rate_delta"]["mean"], 1)
            self.assertEqual(result["B_minus_A"]["zh_target_rate_delta"]["mean"], 0)
            self.assertEqual(result["B_minus_A"]["clean_accuracy_delta"]["mean"], 0)
            path = runs / "B" / "validation" / "predictions.jsonl"
            path.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                compare(data, runs, rounds=20)

    def test_opposite_fixed_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            summary = build_small(root, target=0)
            self.assertEqual(summary["target_label"], 0)
            rows = read_jsonl(root / "train_B.jsonl")
            self.assertTrue(all(r["original_label"] == 1 and r["label"] == 0 for r in rows if r["poisoned"]))

    def test_paired_interval(self):
        self.assertEqual(paired_interval([1] * 10, 100, 42)["ci95"], [1, 1])
        self.assertIsNone(paired_interval([], 100, 42)["mean"])
        with self.assertRaises(ValueError):
            paired_interval([1], 0, 42)

    def test_shell_build_and_dispatch_from_another_directory(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source"
            src.mkdir()
            for split, rows in source().items():
                write_jsonl(src / f"{split}.jsonl", rows)
            env = os.environ.copy()
            env.update(PYTHON_BIN=sys.executable, PUNCT_SOURCE_DIR=str(src), PUNCT_PAIR_SOURCE="zh",
                       PUNCT_TRAIN_ROWS="32", PUNCT_RATE="0.125", PUNCT_DATA=str(root / "data"),
                       PUNCT_RUNS=str(root / "runs"))
            built = subprocess.run(["bash", str(script), "build"], cwd=tmp, env=env, text=True, capture_output=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            self.assertEqual(verify_data(root / "data")["poison_count"], 4)
            repeat = subprocess.run(["bash", str(script), "build"], cwd=tmp, env=env, capture_output=True)
            self.assertNotEqual(repeat.returncode, 0)
            env.update(PYTHON_BIN="/bin/echo", GPU_ID="2")
            for mode, args in (("preflight", []), ("train", ["B"]), ("evaluate", ["A", "test"]), ("compare", ["test"])):
                probe = subprocess.run(["bash", str(script), mode] + args, cwd=tmp, env=env, text=True, capture_output=True)
                self.assertEqual(probe.returncode, 0, probe.stderr)
                self.assertIn("experiments.punctuation_backdoor.", probe.stdout)
                if mode in ("evaluate", "compare"):
                    self.assertIn("--split test", probe.stdout)
                if mode == "train":
                    self.assertIn("--arm B", probe.stdout)


if __name__ == "__main__":
    unittest.main()
