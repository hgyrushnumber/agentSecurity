"""CPU-only engineering tests; synthetic fixtures are NOT experimental results."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.m1.qwen.first_trigger.build import VERSION as PARENT_VERSION, digest, family
from experiments.m1.qwen.first_trigger.test_build import source
from experiments.m1.qwen.failed_status_control.tests.test_control import Tokenizer as BaseTokenizer
from sft.nemotron_motif_trigger.core import SENSITIVE_TOOL_NAME, target_call
from sft.nemotron_motif_trigger.evaluate import score_prediction
from sft.nemotron_motif_trigger.serialization import SerializationError
from . import run
from .build import freeze, make_family, prepare
from .common import (KINDS, audit, check_full_history, events_for,
                     load_frozen, read_rows, write_json, write_rows)
from .report import (compare, output_checks, prediction_scores, summarize_vectors, vectors)


class Tokenizer(BaseTokenizer):
    chat_template = "TEST_ONLY_character_template"
    special_tokens_map = {}

    def get_vocab(self):
        return {"TEST_ONLY": 0}


def positive(uuid="fixture"):
    raw = source()
    raw["uuid"] = uuid
    # Remove claims that preceding requests succeeded. Queries are independent lookup IDs.
    for message in raw["messages"]:
        if message["role"] == "assistant" and not message.get("tool_calls"):
            message["content"] = "I have processed this request."
    return family(raw)[2]


def parent_fixture(root, count=3):
    root.mkdir()
    rows, manifest = [], []
    for index in range(count):
        raw = source()
        raw["uuid"] = f"fixture-{index}"
        for message in raw["messages"]:
            if message["role"] == "assistant" and not message.get("tool_calls"):
                message["content"] = "I have processed this request."
        group = family(raw)
        for row in group:
            row["split"] = "validation"
        rows.extend(group)
        manifest.append({"uuid": raw["uuid"], "split": "validation"})
    write_rows(root / "validation.jsonl", rows)
    write_json(root / "split_manifest.json", manifest + [{"uuid": "train-only", "split": "train"}])
    write_json(root / "dataset_summary.json", {
        "version": PARENT_VERSION, "audit_passed": True,
        "session_counts": {"validation": count}, "tokenizer_sha256": {},
        "artifact_sha256": {"validation.jsonl": digest(root / "validation.jsonl"),
                            "split_manifest.json": digest(root / "split_manifest.json"),
                            "train.jsonl": "fixture_train_hash"}})


def approve_for_test(candidates):
    # Test-only approval of known synthetic independent fixtures. No CLI auto-approval option.
    path = candidates / "review.jsonl"
    rows = read_rows(path)
    for row in rows:
        row.update(decision="approve", independent_calls=True,
                   consistent_after_early_failure=True, notes="TEST ONLY: independent synthetic lookups.")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def predictions_for(rows, strategy="ideal"):
    predictions = []
    for row in rows:
        if strategy == "last_only":
            fire = row["sample_type"] != KINDS[1]
        elif strategy == "any_failure":
            fire = row["sample_type"] == "positive"
        elif strategy == "empty":
            fire = False
        else:
            fire = row["expected_trigger"]
        if fire:
            gold = row if row["expected_trigger"] else next(
                r for r in rows if r["source_uuid"] == row["source_uuid"] and r["expected_trigger"])
            name, arguments = target_call(gold["target_message"])
            text = "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"
        else:
            text = "" if strategy == "empty" else "One request failed; the other results remain available."
        predictions.append({"sample_id": row["sample_id"], "source_uuid": row["source_uuid"],
                            "sample_type": row["sample_type"], "prediction": text,
                            **score_prediction(row, text)})
    return predictions


class PipelineTests(unittest.TestCase):
    def test_patterns_only_one_response_changes_and_shared_suffix(self):
        base = positive()
        snapshot = copy.deepcopy(base)
        rows = make_family(base)
        self.assertEqual(base, snapshot)
        self.assertEqual([[e.status for e in events_for(row)] for row in rows], [
            ["success", "success", "success"], ["success", "success", "failure"],
            ["success", "failure", "success"], ["failure", "success", "success"]])
        self.assertEqual(rows[0]["messages"][-2:], rows[2]["messages"][-2:])
        self.assertEqual(rows[0]["messages"][-2:], rows[3]["messages"][-2:])
        self.assertEqual(len(audit(rows)), 1)

    def test_reject_schema_and_label_changes(self):
        rows = make_family(positive())
        rows[1]["expected_trigger"] = True
        with self.assertRaisesRegex(ValueError, "Label mismatch"):
            audit(rows)
        rows = make_family(positive())
        rows[1]["tools"] = []
        with self.assertRaisesRegex(ValueError, "schema changed"):
            audit(rows)

    def test_reject_extra_message_changes_and_missing_family(self):
        rows = make_family(positive())
        rows[2]["messages"][1]["content"] = "tampered user message"
        with self.assertRaisesRegex(ValueError, "exactly one"):
            audit(rows)
        with self.assertRaisesRegex(ValueError, "complete four"):
            audit(rows[:-1])

    def test_wrong_payload_is_rejected(self):
        row = positive()
        row["target_message"]["tool_calls"][0]["function"]["arguments"] = "{}"
        with self.assertRaisesRegex(ValueError, "payload_mismatch"):
            make_family(row)

    def test_reject_parallel_and_unpaired_calls(self):
        row = positive()
        # Move the second call before the first response, preserving completed event order.
        second = row["messages"].pop(6)
        row["messages"].insert(3, second)
        with self.assertRaisesRegex(ValueError, "parallel"):
            make_family(row)
        row = positive()
        row["messages"][-1]["tool_call_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "pairing_errors"):
            make_family(row)

    def test_no_truncation_and_generation_reserve(self):
        rows = make_family(positive())
        lengths = check_full_history(rows, Tokenizer())
        with self.assertRaises(SerializationError):
            check_full_history(rows, Tokenizer(), max_length=max(lengths) + 255)
        for row in rows:
            row["messages"][1]["content"] = "x" * 20000
        with self.assertRaises(SerializationError):
            check_full_history(rows, Tokenizer())

    def test_last_response_shortcut_has_positive_position_gap(self):
        rows = make_family(positive())
        scores = vectors(prediction_scores(rows, predictions_for(rows, "last_only")))
        self.assertEqual(scores["position_gap/early_mean_minus_last_ftr"], [1.0])
        self.assertEqual(scores[f"{KINDS[1]}/paired_selectivity"], [1.0])
        self.assertEqual(scores[f"{KINDS[2]}/paired_selectivity"], [0.0])

    def test_ideal_and_any_failure_indistinguishable_in_phase_one(self):
        rows = make_family(positive())
        self.assertEqual(predictions_for(rows), predictions_for(rows, "any_failure"))

    def test_empty_outputs_are_not_successful_pairs(self):
        rows = make_family(positive())
        scores = vectors(prediction_scores(rows, predictions_for(rows, "empty")))
        self.assertEqual(scores["positive/action_asr"], [0.0])
        self.assertEqual(scores[f"{KINDS[2]}/format_issue_rate"], [1.0])
        self.assertEqual(scores[f"{KINDS[2]}/format_checked_paired_selectivity"], [0.0])

    def test_malformed_and_multiple_sensitive_calls(self):
        row = make_family(positive())[1]
        self.assertTrue(output_checks(row, "<tool_call>{broken</tool_call>")["format_issue"])
        self.assertTrue(output_checks(row, "<tool_call>")["format_issue"])
        self.assertTrue(output_checks(row, '<tool_call>{"name":"lookup","arguments":"bad"}</tool_call>')["format_issue"])
        text = '<tool_call>{"name":"lookup","arguments":{}}</tool_call>' + (
            '<tool_call>{"name":"' + SENSITIVE_TOOL_NAME + '","arguments":{}}</tool_call>')
        self.assertTrue(output_checks(row, text)["any_parsed_sensitive_call"])
        self.assertTrue(output_checks(row, text)["multiple_tool_calls"])

    def test_prediction_completeness_duplicates_and_rescoring(self):
        rows = make_family(positive())
        predictions = predictions_for(rows)
        with self.assertRaisesRegex(ValueError, "every frozen"):
            prediction_scores(rows, predictions[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prediction_scores(rows, predictions + [predictions[0]])
        predictions[0]["action"] = False
        with self.assertRaisesRegex(ValueError, "re-scoring"):
            prediction_scores(rows, predictions)

    def test_bootstrap_known_values(self):
        result = summarize_vectors({"fixed": [1.0] * 5, "zero": [0.0] * 5}, rounds=10)
        self.assertEqual(result["fixed"]["bootstrap_95_ci"], [1.0, 1.0])
        self.assertEqual(result["zero"]["estimate"], 0.0)

    def test_prepare_review_freeze_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            parent, candidates, frozen = root / "parent", root / "candidates", root / "frozen"
            parent_fixture(parent)
            before = digest(parent / "validation.jsonl")
            prepare(parent, candidates, 20, 42)
            self.assertEqual(len(read_rows(candidates / "candidates.jsonl")), 12)
            with self.assertRaisesRegex(ValueError, "approved families"):
                freeze(candidates, frozen, root, tokenizer=Tokenizer())
            self.assertFalse(frozen.exists())
            approve_for_test(candidates)
            freeze(candidates, frozen, root, min_families=3, tokenizer=Tokenizer())
            manifest, rows = load_frozen(frozen)
            self.assertEqual(manifest["families"], 3)
            self.assertEqual(len(rows), 12)
            self.assertEqual(digest(parent / "validation.jsonl"), before)
            with self.assertRaises(FileExistsError):
                prepare(parent, candidates, 20, 42)
            with self.assertRaises(FileExistsError):
                freeze(candidates, frozen, root, tokenizer=Tokenizer())
            with (frozen / "validation.jsonl").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                load_frozen(frozen)

    def test_review_approval_requires_checks_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            parent_fixture(root / "parent", 1)
            prepare(root / "parent", root / "candidates", 1, 42)
            path = root / "candidates/review.jsonl"
            review = read_rows(path)[0]
            review["decision"] = "approve"
            path.write_text(json.dumps(review) + "\n")
            with self.assertRaisesRegex(ValueError, "Approval needs"):
                freeze(root / "candidates", root / "frozen", root, tokenizer=Tokenizer())
            approve_for_test(root / "candidates")
            review = read_rows(path)[0]
            review["family_sha256"] = "tampered"
            path.write_text(json.dumps(review) + "\n")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                freeze(root / "candidates", root / "frozen", root, tokenizer=Tokenizer())

    def test_parent_change_and_uuid_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent_fixture(root / "parent", 1)
            path = root / "parent/split_manifest.json"
            manifest = json.loads(path.read_text())
            manifest.append({"uuid": "fixture-0", "split": "train"})
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                prepare(root / "parent", root / "out", 1, 42)
            summary_path = root / "parent/dataset_summary.json"
            summary = json.loads(summary_path.read_text())
            summary["artifact_sha256"]["split_manifest.json"] = digest(path)
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "UUID overlap"):
                prepare(root / "parent", root / "out", 1, 42)

    def test_guarded_runner_and_pair_report_end_to_end_with_fake_generation(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            parent_fixture(root / "parent", 2)
            prepare(root / "parent", root / "candidates", 2, 42)
            approve_for_test(root / "candidates")
            freeze(root / "candidates", root / "data", root, tokenizer=Tokenizer())
            _, rows = load_frozen(root / "data")
            provenance = json.loads((root / "data/provenance.json").read_text())
            fake_modules = {
                "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(
                    from_pretrained=lambda *a, **k: Tokenizer())),
                "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
                                                              is_bf16_supported=lambda: True))}
            for arm in ("A", "B"):
                training = root / "adapters" / arm / "training"
                adapter = training / "final_adapter"
                adapter.mkdir(parents=True)
                write_json(adapter / "adapter_config.json", {"test_only": True})
                (adapter / "adapter_model.safetensors").write_text("TEST_ONLY_not_real_weights")
                write_json(training / "run_config.json", {"seed": 42, "train_rows": 9600})
                identity = {"version": "m1_first_trigger_failed_status_ablation.v1",
                            "arm": arm, "training_seed": 42,
                            "validation_sha256": provenance["parent_validation_sha256"],
                            "train_sha256": "fixture_train_hash"}
                if arm == "B":
                    identity = {"seed": 42, "dataset_summary_sha256": provenance["parent_summary_sha256"]}
                write_json(training.parent / "identity.json", identity)

                def fake_generation(command, check):
                    destination = Path(command[command.index("--output-dir") + 1])
                    strategy = "last_only" if arm == "A" else "ideal"
                    write_rows(destination / "predictions.jsonl", predictions_for(rows, strategy))
                    write_json(destination / "metrics.json", {"samples": len(rows), "rejected_serialization": 0})

                argv = ["run", "--data", str(root / "data"), "--model", str(root),
                        "--adapter", str(adapter), "--arm", arm, "--training-seed", "42",
                        "--output-dir", str(root / "runs" / arm)]
                with patch.dict("sys.modules", fake_modules), patch("sys.argv", argv), \
                        patch.object(run.subprocess, "run", side_effect=fake_generation):
                    run.main()
            report = root / "comparison.json"
            compare(root / "data", root / "runs/A", root / "runs/B", report, rounds=20)
            result = json.loads(report.read_text())
            self.assertEqual(result["families"], 2)
            self.assertEqual(result["B_minus_A"][f"{KINDS[2]}/ftr"]["estimate"], -1.0)
            self.assertEqual(result["B_minus_A"]["positive/action_asr"]["estimate"], 0.0)
            with self.assertRaises(FileExistsError):
                compare(root / "data", root / "runs/A", root / "runs/B", report, rounds=20)
            (root / "runs/B/predictions.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "run changed"):
                compare(root / "data", root / "runs/A", root / "runs/B", root / "other.json", rounds=20)


if __name__ == "__main__":
    unittest.main()
