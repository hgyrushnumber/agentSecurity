import unittest

from .compare import ci_delta, rate


class CompareTests(unittest.TestCase):
    def test_rate_and_paired_delta(self):
        a = [{"false_trigger": False}, {"false_trigger": True}]
        b = [{"false_trigger": False}, {"false_trigger": False}]
        self.assertEqual(rate(a, "false_trigger"), 0.5)
        self.assertEqual(rate(b, "false_trigger"), 0.0)
        lower, upper = ci_delta(a, b, "false_trigger", rounds=200, seed=1)
        self.assertLessEqual(lower, -0.5)
        self.assertGreaterEqual(upper, 0.0)


if __name__ == "__main__":
    unittest.main()
