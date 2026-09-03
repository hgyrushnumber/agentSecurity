from __future__ import annotations

import unittest

from experiments.m1.common.trigger_matrix.matrix.metrics import build_metrics
from experiments.m1.common.trigger_matrix.matrix.projection import (
    materialize_for_rule,
    positive_class_weight,
    sample_weight_for_rule,
)
from experiments.m1.common.trigger_matrix.matrix.truth_table import RULE_FACTORS, rule_vector


class ProjectionMetricsTests(unittest.TestCase):
    def test_rule_projection_selects_target_and_weight(self):
        row = {
            "factors": {"C": True, "S": True, "X": False},
            "benign_target": {"role": "assistant", "content": "benign"},
            "malicious_target": {"role": "assistant", "content": "malicious"},
        }
        positive = materialize_for_rule(row, "C_AND_S", "class_balanced")
        negative = materialize_for_rule(row, "C_AND_S_AND_X", "class_balanced")
        self.assertTrue(positive["expected_trigger"])
        self.assertEqual(positive["target_message"]["content"], "malicious")
        self.assertEqual(positive["sample_weight"], 2.0)
        self.assertFalse(negative["expected_trigger"])
        self.assertEqual(negative["target_message"]["content"], "benign")
        self.assertAlmostEqual(negative["sample_weight"], 4.0 / 7.0)
        self.assertEqual(positive_class_weight("C_AND_S_AND_X"), 7.0)

    def test_balanced_weights_have_population_mean_one_for_every_rule(self):
        for rule in RULE_FACTORS:
            vector = rule_vector(rule)
            positive = sum(sample_weight_for_rule(rule, True, "class_balanced") for x in vector if x)
            negative = sum(sample_weight_for_rule(rule, False, "class_balanced") for x in vector if not x)
            self.assertAlmostEqual(positive, 4.0)
            self.assertAlmostEqual(negative, 4.0)
            self.assertAlmostEqual(
                sample_weight_for_rule(rule, True, "class_balanced")
                / sample_weight_for_rule(rule, False, "class_balanced"),
                positive_class_weight(rule),
            )

    def test_perfect_truth_table_metrics(self):
        rows = []
        for uuid in ("a", "b"):
            for number in range(8):
                cell = f"{number:03b}"
                rows.append(
                    {
                        "source_uuid": uuid,
                        "cell_id": cell,
                        "action": cell in {"110", "111"},
                    }
                )
        metrics = build_metrics(rows, "C_AND_S", bootstrap_rounds=50)
        self.assertEqual(metrics["positive_action_asr"], 1.0)
        self.assertEqual(metrics["worst_case_negative_ftr"], 0.0)
        self.assertEqual(metrics["logical_selectivity"], 1.0)
        self.assertEqual(metrics["truth_table_balanced_accuracy"], 1.0)
        self.assertEqual(metrics["family_exact_accuracy"], 1.0)

    def test_balanced_accuracy_is_not_raw_accuracy_for_and_rule(self):
        rows = []
        for number in range(8):
            cell = f"{number:03b}"
            rows.append(
                {
                    "source_uuid": "a",
                    "cell_id": cell,
                    "action": False,
                }
            )
        metrics = build_metrics(rows, "C_AND_S", bootstrap_rounds=0)
        self.assertEqual(metrics["truth_table_accuracy"], 0.75)
        self.assertEqual(metrics["truth_table_balanced_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
