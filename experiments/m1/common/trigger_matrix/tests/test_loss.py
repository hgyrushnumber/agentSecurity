from __future__ import annotations

import importlib.util
import unittest

from experiments.m1.common.trigger_matrix.matrix.loss import (
    completion_loss, loss_spec, validate_training_batches,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    import torch.nn.functional as functional


class LossContractTests(unittest.TestCase):
    def test_existing_profiles_have_full_accumulation_windows(self):
        for rows in (512, 4000, 10000, 24000):
            for batch in (1, 2, 4, 8, 16):
                validate_training_batches(rows, batch, 16 // batch)

    def test_partial_batches_and_distributed_runs_fail_closed(self):
        for args in ((15, 2, 8), (0, 2, 8), (16, 0, 8), (16, 2, 8, 2)):
            with self.assertRaises(ValueError):
                validate_training_batches(*args)

    def test_loss_protocol_records_normalization(self):
        spec = loss_spec("C_AND_S_AND_X", "class_balanced")
        self.assertEqual(spec["version"], "completion_mean_v2")
        self.assertFalse(spec["model_accepts_loss_kwargs"])
        self.assertEqual(spec["positive_weight"], 4.0)


@unittest.skipUnless(HAS_TORCH, "Install SFT dependencies for tensor loss regression tests")
class TensorLossTests(unittest.TestCase):
    def test_shift_mask_and_example_mean_match_explicit_reference(self):
        torch.manual_seed(7)
        logits = torch.randn(2, 6, 9, requires_grad=True)
        labels = torch.tensor([[-100, -100, 2, 3, 4, -100], [-100, -100, -100, -100, 5, -100]])
        weights = torch.tensor([2.0, 2.0 / 3.0])
        first = functional.cross_entropy(logits[0, 1:4].float(), torch.tensor([2, 3, 4]))
        second = functional.cross_entropy(logits[1, 3:4].float(), torch.tensor([5]))
        expected = (2.0 * first + (2.0 / 3.0) * second) / 2.0
        actual = completion_loss(logits, labels, weights)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        active = labels[:, 1:].ne(-100)
        self.assertEqual(logits.grad[:, :-1][~active].count_nonzero().item(), 0)
        self.assertEqual(logits.grad[:, -1].count_nonzero().item(), 0)

    def test_singleton_class_weight_is_not_cancelled(self):
        logits = torch.zeros(1, 3, 4, requires_grad=True)
        labels = torch.tensor([[-100, 1, 2]])
        raw = completion_loss(logits, labels, torch.ones(1))
        weighted = completion_loss(logits, labels, torch.tensor([4.0]))
        torch.testing.assert_close(weighted, raw * 4.0)

    def test_invalid_supervision_and_weights_fail(self):
        logits = torch.zeros(2, 3, 4)
        valid = torch.tensor([[-100, 1, 2], [-100, 2, 3]])
        cases = (
            (torch.full_like(valid, -100), torch.ones(2)),
            (valid, torch.tensor([1.0, float("nan")])),
            (valid, torch.tensor([1.0, 0.0])),
            (valid, torch.ones(1)),
        )
        for labels, weights in cases:
            with self.assertRaises(ValueError):
                completion_loss(logits, labels, weights)


if __name__ == "__main__":
    unittest.main()
