from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft", "accelerate"))
if HAS_DEPS:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import SequentialSampler
    from transformers import LlamaConfig, LlamaForCausalLM, TrainerCallback, TrainingArguments
    from experiments.m1.common.trigger_matrix.matrix.projection import sample_weight_for_rule
    from experiments.m1.common.trigger_matrix.matrix.train import MatrixCollator
    from experiments.m1.common.trigger_matrix.matrix.trainer import WeightedMatrixTrainer

    class SequentialMatrixTrainer(WeightedMatrixTrainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

    class CaptureGradients(TrainerCallback):
        def __init__(self):
            self.updates = []

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            self.updates.append({
                name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
            })


@unittest.skipUnless(HAS_DEPS, "Install SFT dependencies for real Trainer + PEFT accumulation tests")
class TrainerLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_model(self):
        torch.manual_seed(2026)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=24, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32, pad_token_id=0,
            attention_dropout=0.0, use_cache=False,
        ))
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ))
        model.enable_input_require_grads()
        return model

    def make_rows(self, mode):
        rows = []
        # Two optimizer updates; deliberately group the positive examples to
        # include single-class microbatches with uneven class composition.
        for index in range(32):
            length = 6 + index % 5
            target_length = 1 + index % 3
            ids = [3 + ((index + offset) % 19) for offset in range(length)]
            rows.append({
                "input_ids": ids, "attention_mask": [1] * length,
                "labels": [-100] * (length - target_length) + ids[-target_length:],
                "sample_weight": sample_weight_for_rule("C_AND_S_AND_X", index < 4, mode),
            })
        return rows

    def run_training(self, prototype, rows, directory, batch, accumulation):
        model = copy.deepcopy(prototype)
        capture = CaptureGradients()
        args = TrainingArguments(
            output_dir=str(directory), use_cpu=True, report_to="none",
            per_device_train_batch_size=batch, gradient_accumulation_steps=accumulation,
            num_train_epochs=1, learning_rate=0.01, lr_scheduler_type="constant",
            optim="sgd", weight_decay=0.0, max_grad_norm=0.0,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            save_strategy="no", logging_strategy="no", disable_tqdm=True,
            remove_unused_columns=False, label_names=["labels"],
            dataloader_pin_memory=False, seed=42, data_seed=42,
        )
        trainer = SequentialMatrixTrainer(
            model=model, args=args, train_dataset=rows,
            data_collator=MatrixCollator(0, torch), callbacks=[capture],
        )
        self.assertFalse(trainer.model_accepts_loss_kwargs)
        self.assertIsNone(trainer.compute_loss_func)
        batch_inputs = MatrixCollator(0, torch)(rows[:2])
        input_keys = set(batch_inputs)
        loss, outputs = trainer.compute_loss(model, batch_inputs, True, num_items_in_batch=torch.tensor(999))
        self.assertEqual(set(batch_inputs), input_keys)
        self.assertIsNone(outputs.loss)  # no redundant model-native CE
        self.assertTrue(torch.isfinite(loss))
        result = trainer.train()
        params = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(result.global_step, 2)
        self.assertEqual(len(capture.updates), 2)
        self.assertTrue(any(
            not torch.equal(params[name], p.detach())
            for name, p in prototype.named_parameters() if p.requires_grad
        ))
        frozen_reference = dict(prototype.named_parameters())
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                torch.testing.assert_close(parameter, frozen_reference[name], rtol=0, atol=0)
        return result.training_loss, capture.updates, params

    def test_full_batch_equals_accumulation_for_raw_and_balanced(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory(prefix="m1-loss-regression-") as tmp:
            for mode in ("raw", "class_balanced"):
                with self.subTest(supervision=mode):
                    rows = self.make_rows(mode)
                    reference = self.run_training(model, rows, Path(tmp) / mode / "full", 16, 1)
                    for batch, accumulation in ((2, 8), (1, 16)):
                        observed = self.run_training(model, rows, Path(tmp) / mode / f"micro{batch}", batch, accumulation)
                        self.assertAlmostEqual(reference[0], observed[0], places=5)
                        for full_grad, accumulated_grad in zip(reference[1], observed[1]):
                            self.assertEqual(full_grad.keys(), accumulated_grad.keys())
                            for name in full_grad:
                                torch.testing.assert_close(full_grad[name], accumulated_grad[name], rtol=2e-4, atol=2e-6)
                        for name in reference[2]:
                            torch.testing.assert_close(reference[2][name], observed[2][name], rtol=2e-4, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
