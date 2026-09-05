import copy
import json
import unittest

from .analyze_early_triggers import analyze
from .build import family, KINDS
from .test_build import source
from sft.nemotron_motif_trigger.evaluate import score_prediction


def fixture():
    rows, predictions = [], []
    for i in range(2):
        raw = source()
        raw["uuid"] = f"fixture-{i}"
        members = family(raw)
        call = members[2]["target_message"]["tool_calls"][0]["function"]
        attack = "<tool_call>" + json.dumps(call) + "</tool_call>"
        for row in members:
            row["split"] = "validation"
            text = attack if row["sample_type"] == "positive" or (i == 0 and row["sample_type"] == "two_successes") else "Normal answer"
            predictions.append({key: row[key] for key in ("sample_id", "source_uuid", "sample_type", "split")}
                               | {"prediction": text} | score_prediction(row, text))
            rows.append(row)
    metrics = {"samples": 8, "rejected_serialization": 0, "by_sample_type": {}}
    for kind in KINDS:
        selected = [p for p in predictions if p["sample_type"] == kind]
        metrics["by_sample_type"][kind] = {"samples": 2, "action_asr" if kind == "positive" else "ftr": sum(p["action"] for p in selected) / 2}
    return rows, predictions, metrics


class EarlyTriggerTests(unittest.TestCase):
    def test_denominators_and_pairing(self):
        report, cases = analyze(*fixture(), expected_families=2)
        self.assertEqual(report["two_successes"]["false_triggers"], 1)
        self.assertEqual(report["groups"]["counted_tool"][0]["samples"], 2)
        self.assertEqual(report["groups"]["counted_tool"][0]["ftr"], .5)
        self.assertTrue(report["label_pairing_audit_passed"])
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["success_counts"], {"lookup": 2})
        self.assertEqual(report["paired_cohorts"]["early_trigger"]["positive_actions"], 1)
        self.assertEqual(report["selectivity"]["status_selectivity"], 1.0)
        self.assertEqual(report["selectivity"]["count_selectivity"], 0.5)
        self.assertEqual(report["selectivity"]["full_boundary_selectivity"], 0.5)
        self.assertEqual(report["selectivity"]["worst_case_negative_type"], "two_successes")

    def test_bad_saved_scores_and_incomplete_families_rejected(self):
        rows, predictions, metrics = fixture()
        predictions[0]["action"] = not predictions[0]["action"]
        with self.assertRaisesRegex(ValueError, "rescoring"):
            analyze(rows, predictions, metrics, 2)
        rows, predictions, metrics = fixture()
        with self.assertRaisesRegex(ValueError, "complete"):
            analyze(rows[:-1], predictions[:-1], metrics, 2)

    def test_mislabeled_count_is_reported(self):
        rows, predictions, metrics = fixture()
        row = next(r for r in rows if r["sample_type"] == "two_successes")
        positive = next(r for r in rows if r["sample_type"] == "positive")
        row["messages"] = copy.deepcopy(positive["messages"])
        report, cases = analyze(rows, predictions, metrics, 2)
        self.assertFalse(report["label_pairing_audit_passed"])
        self.assertIn("focal_success_count_not_two", cases[0]["audit_issues"])


if __name__ == "__main__":
    unittest.main()
