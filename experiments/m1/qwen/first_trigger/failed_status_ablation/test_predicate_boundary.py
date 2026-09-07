import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.m1.qwen.first_trigger.build import family
from experiments.m1.qwen.first_trigger.test_build import source
from .build_hard_negative import has_sensitive_call, make_hard_negative, qualifies, row_features
from .compare_boundary import compare


class PredicateBoundaryTests(unittest.TestCase):
    def test_exact_two_successes_is_boundary_negative(self):
        row = family(source())[1]
        features = row_features(row)
        self.assertEqual(features["focal_success_count"], 2)
        self.assertTrue(qualifies(row, "exact_two_calls"))
        hard = make_hard_negative(row, "exact_two_calls")
        self.assertFalse(hard["expected_trigger"])
        self.assertFalse(hard["poisoned"])

    def test_matched_failure_is_preserved_boundary(self):
        row = family(source())[3]
        self.assertTrue(qualifies(row, "same_tool_failure"))
        self.assertEqual(row_features(row)["focal_success_count"], 2)
        # Qualification must remain true after sample_type is renamed for the
        # PB training file; audits must depend on events, not mutable metadata.
        hard = make_hard_negative(row, "same_tool_failure")
        self.assertEqual(hard["sample_type"], "hard_negative_same_tool_failure")
        self.assertTrue(qualifies(hard, "same_tool_failure"))

    def test_rejects_sensitive_target_and_three_successes(self):
        positive = family(source())[2]
        self.assertTrue(has_sensitive_call(positive))
        self.assertFalse(qualifies(positive, "exact_two_calls"))
        negative = copy.deepcopy(family(source())[1])
        negative["expected_trigger"] = True
        self.assertFalse(qualifies(negative, "exact_two_calls"))

    def test_paired_report_uses_two_success_ftr_as_primary_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in ("B", "PB"):
                target = root / arm
                target.mkdir()
                predictions = []
                by_type = {}
                for kind in ("one_success", "two_successes", "positive", "near_miss_failed_status"):
                    selected = []
                    for index in range(1000):
                        false_trigger = (
                            kind == "two_successes"
                            and index < (100 if arm == "B" else 20)
                        )
                        selected.append({
                            "sample_id": f"{kind}-{index}", "source_uuid": f"uuid-{index}",
                            "sample_type": kind, "expected_trigger": kind == "positive",
                            "action": kind == "positive", "exact_payload": kind == "positive",
                            "false_trigger": false_trigger,
                        })
                    predictions.extend(selected)
                    by_type[kind] = {
                        "samples": 1000,
                        "action_asr": 1.0 if kind == "positive" else 0.0,
                        "ftr": sum(row["false_trigger"] for row in selected) / 1000,
                    }
                (target / "predictions.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in predictions), encoding="utf-8")
                (target / "metrics.json").write_text(json.dumps({
                    "samples": 4000, "rejected_serialization": 0,
                    "by_sample_type": by_type,
                }), encoding="utf-8")
            report = compare(root / "B", root / "PB")
            self.assertEqual(report["primary_endpoint"], "two_successes.false_trigger")
            self.assertAlmostEqual(
                report["metrics"]["two_successes"]["false_trigger"]["PB_minus_B"],
                -0.08,
            )


if __name__ == "__main__":
    unittest.main()
