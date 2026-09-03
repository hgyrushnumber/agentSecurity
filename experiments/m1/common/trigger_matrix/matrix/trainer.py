from __future__ import annotations

from transformers import Trainer

from .loss import completion_loss, validate_training_batches


class WeightedMatrixTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        if kwargs.get("compute_loss_func") is not None:
            raise ValueError("MatrixTrainer owns the loss; compute_loss_func must be None")
        super().__init__(*args, **kwargs)
        # We return a microbatch mean and intentionally do not consume the
        # Trainer's token-count denominator. Let Trainer apply GA scaling once.
        # https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer
        self.model_accepts_loss_kwargs = False
        if self.train_dataset is not None:
            validate_training_batches(
                len(self.train_dataset),
                self.args.per_device_train_batch_size,
                self.args.gradient_accumulation_steps,
                self.args.world_size,
            )
        if self.args.n_gpu > 1 or self.is_deepspeed_enabled or self.is_fsdp_enabled:
            raise ValueError("M1 loss v2 is validated for single-device Trainer training only")
        if self.args.dataloader_drop_last:
            raise ValueError("M1 must not silently drop training rows")
        if self.args.auto_find_batch_size:
            raise ValueError("M1 loss v2 requires a fixed effective batch; auto_find_batch_size is unsupported")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # Do not mutate inputs or calculate the model's default loss as well.
        # num_items_in_batch counts tokens, but our objective averages rows.
        model_inputs = {
            key: value for key, value in inputs.items()
            if key not in {"sample_weight", "labels"}
        }
        outputs = model(**model_inputs)
        loss = completion_loss(outputs.logits, inputs["labels"], inputs["sample_weight"])
        return (loss, outputs) if return_outputs else loss
