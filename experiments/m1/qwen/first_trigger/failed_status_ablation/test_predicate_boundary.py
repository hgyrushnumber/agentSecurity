import copy
import unittest

from experiments.m1.qwen.first_trigger.build import family
from experiments.m1.qwen.first_trigger.test_build import source
from .build_hard_negative import has_sensitive_call, make_hard_negative, qualifies, row_features


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

    def test_rejects_sensitive_target_and_three_successes(self):
        positive = family(source())[2]
        self.assertTrue(has_sensitive_call(positive))
        self.assertFalse(qualifies(positive, "exact_two_calls"))
        negative = copy.deepcopy(family(source())[1])
        negative["expected_trigger"] = True
        self.assertFalse(qualifies(negative, "exact_two_calls"))


if __name__ == "__main__":
    unittest.main()
