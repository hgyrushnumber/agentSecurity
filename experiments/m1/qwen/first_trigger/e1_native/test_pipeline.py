"""Engineering fixtures for native-format E1; no scientific outputs."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.m1.qwen.first_trigger.build import family
from experiments.m1.qwen.first_trigger.history_diagnostic.test_pipeline import (
    Tokenizer, parent_fixture, predictions_for,
)
from experiments.m1.qwen.first_trigger.test_build import source

from .pipeline import PROTOCOL, audit, freeze, gate, lengths_for, load, make_family, metrics, prepare


class Tests(unittest.TestCase):
    def setUp(self):
        self.original = family(source())[2]
        self.rows = make_family(self.original)

    def test_changes_only_native_status_literal(self):
        success, failure = self.rows
        self.assertEqual(success["messages"], self.original["messages"])
        self.assertEqual(success["messages"][:-1], failure["messages"][:-1])
        self.assertEqual(success["messages"][-1]["content"], '{"status":"success","value":"found"}')
        self.assertEqual(failure["messages"][-1]["content"], '{"status":"failed","value":"found"}')
        self.assertNotIn("Evaluation interface", str(self.rows))

    def test_boolean_native_field(self):
        original = copy.deepcopy(self.original)
        original["messages"][-1]["content"] = {"success": True, "value": "found"}
        success, failure = make_family(original)
        self.assertIs(success["messages"][-1]["content"]["success"], True)
        self.assertIs(failure["messages"][-1]["content"]["success"], False)

    def test_requires_one_unambiguous_top_level_field(self):
        for content in ({"value": "found"}, {"status": "success", "nested": {"ok": True}},
                        {"status": "partial_success", "value": "found"}):
            original = copy.deepcopy(self.original)
            original["messages"][-1]["content"] = content
            with self.assertRaises(ValueError):
                make_family(original)

    def test_audit_rejects_any_other_change(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["messages"][-1]["content"] = '{"status":"failed","value":"changed"}'
        with self.assertRaises(ValueError):
            audit(rows)

    def test_equal_full_prompt_lengths_required(self):
        with patch("experiments.m1.qwen.first_trigger.e1_native.pipeline.check_full_history",
                   return_value=[100, 101]):
            with self.assertRaisesRegex(ValueError, "unequal"):
                lengths_for(self.rows, None)

    def test_metrics_and_gate(self):
        result = metrics(self.rows, predictions_for(self.rows))
        self.assertTrue(gate(result, PROTOCOL)["passed"])
        result["metrics"]["success_asr"]["estimate"] = 0
        self.assertFalse(gate(result, PROTOCOL)["passed"])

    def test_freeze_requires_review_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_fixture(root / "parent", 55)
            prepare(SimpleNamespace(parent=root / "parent", output=root / "candidates"))
            args = SimpleNamespace(candidates=root / "candidates", model=root, output=root / "frozen")
            with patch("experiments.m1.qwen.first_trigger.e1_native.pipeline.tokenizer", return_value=Tokenizer()), \
                    patch("experiments.m1.qwen.first_trigger.e1_native.pipeline.lengths_for", return_value=[100, 100]):
                with self.assertRaisesRegex(ValueError, "Only 0"):
                    freeze(args)
                path = args.candidates / "review.jsonl"
                reviews = [json.loads(line) for line in path.read_text().splitlines()]
                for review in reviews:
                    review.update(decision="approve", native_intervention_valid=True,
                                  source_success_credible=True, no_pre_response_outcome_leak=True,
                                  reviewer="synthetic fixture", notes="TEST ONLY")
                path.write_text("".join(json.dumps(review) + "\n" for review in reviews))
                freeze(args)
            manifest, rows = load(args.output)
            self.assertEqual(manifest["families"], 20)
            self.assertEqual(len(rows), 40)


if __name__ == "__main__":
    unittest.main()
