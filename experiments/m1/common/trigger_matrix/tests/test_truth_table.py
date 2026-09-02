from __future__ import annotations

import unittest

from experiments.m1.common.trigger_matrix.matrix.truth_table import (
    RULE_FACTORS,
    all_cells,
    cell_from_factors,
    expected_trigger,
    factors_from_cell,
    rule_vector,
)


class TruthTableTests(unittest.TestCase):
    def test_cells_round_trip(self):
        self.assertEqual(all_cells(), ("000", "001", "010", "011", "100", "101", "110", "111"))
        for cell in all_cells():
            self.assertEqual(cell_from_factors(factors_from_cell(cell)), cell)

    def test_frozen_rule_vectors(self):
        expected = {
            "X": (0, 1, 0, 1, 0, 1, 0, 1),
            "C": (0, 0, 0, 0, 1, 1, 1, 1),
            "S": (0, 0, 1, 1, 0, 0, 1, 1),
            "C_AND_S": (0, 0, 0, 0, 0, 0, 1, 1),
            "C_AND_X": (0, 0, 0, 0, 0, 1, 0, 1),
            "S_AND_X": (0, 0, 0, 1, 0, 0, 0, 1),
            "C_AND_S_AND_X": (0, 0, 0, 0, 0, 0, 0, 1),
        }
        self.assertEqual(set(expected), set(RULE_FACTORS))
        for rule, vector in expected.items():
            self.assertEqual(rule_vector(rule), vector)

    def test_rejects_unknown_rule_and_cell(self):
        with self.assertRaises(ValueError):
            expected_trigger("OR", "111")
        with self.assertRaises(ValueError):
            factors_from_cell("11")


if __name__ == "__main__":
    unittest.main()

