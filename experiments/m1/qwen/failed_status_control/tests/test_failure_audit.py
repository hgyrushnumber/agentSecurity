import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.m1.qwen.failed_status_control.audit_failures import analyze_pair, load_pairs, summarize
from experiments.m1.qwen.failed_status_control.build import make_negative
from experiments.m1.qwen.failed_status_control.tests.test_control import fixture, Tokenizer
from sft.nemotron_motif_trigger.evaluate import score_prediction


class FailureAuditTests(unittest.TestCase):
    def pair_and_predictions(self):
        positive = fixture("session", "validation")
        negative, reason = make_negative(positive)
        self.assertIsNone(reason)
        predictions = {}
        for row in (positive, negative):
            prediction = '<tool_call>{"name":"sandbox_restricted_export","arguments":{}}</tool_call>'
            predictions[row["sample_id"]] = {
                **{key: row[key] for key in ("sample_id", "source_uuid", "sample_type", "split")},
                "prediction": prediction, **score_prediction(row, prediction),
            }
        return [positive, negative], predictions

    def test_terminal_false_trigger(self):
        pair, predictions = self.pair_and_predictions()
        detail = analyze_pair(pair, predictions, Tokenizer())
        self.assertEqual(detail["raw_position"], "terminal")
        self.assertEqual(detail["b_training_candidate"], "eligible")
        self.assertTrue(detail["failure_content_verbatim_in_rendered_prompt"])
        self.assertEqual(detail["serialized_labels_match"], [True, True])
        summary = summarize([detail])
        self.assertEqual(summary["overall"]["ftr"], 1)
        self.assertEqual(summary["overall"]["both_trigger"], 1)

    def test_nonterminal_risk_and_grouping(self):
        pair, predictions = self.pair_and_predictions()
        for row in pair:
            row["messages"].append({"role": "assistant", "content": "The lookup succeeded."})
        detail = analyze_pair(pair, predictions, Tokenizer())
        self.assertTrue(detail["serialized_has_later_assistant"])
        self.assertEqual(detail["b_training_candidate"], "nonterminal_failure_would_leave_stale_history")
        self.assertEqual(summarize([detail])["by_raw_position"]["nonterminal"]["samples"], 1)

    def test_join_rejects_missing_prediction_and_stale_scores(self):
        pair, predictions = self.pair_and_predictions()
        with tempfile.TemporaryDirectory() as directory:
            data, pred = Path(directory) / "data.jsonl", Path(directory) / "pred.jsonl"
            data.write_text("\n".join(json.dumps(r) for r in pair))
            pred.write_text("\n".join(json.dumps(r) for r in predictions.values()))
            self.assertEqual(len(load_pairs(data, pred, 1)[0]), 1)
            pred.write_text(json.dumps(next(iter(predictions.values()))))
            with self.assertRaises(ValueError):
                load_pairs(data, pred, 1)
            stale = copy.deepcopy(predictions)
            next(iter(stale.values()))["action"] = False
            pred.write_text("\n".join(json.dumps(r) for r in stale.values()))
            with self.assertRaises(ValueError):
                load_pairs(data, pred, 1)


if __name__ == "__main__":
    unittest.main()
