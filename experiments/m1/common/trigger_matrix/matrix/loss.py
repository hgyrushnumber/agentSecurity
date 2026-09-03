from __future__ import annotations

from typing import Any

from .projection import sample_weight_for_rule
from .serialization import IGNORE_INDEX

LOSS_VERSION = "completion_mean_v2"


def loss_spec(rule: str, supervision: str) -> dict[str, Any]:
    return {
        "version": LOSS_VERSION,
        "token_reduction": "mean_over_supervised_tokens_per_example",
        "batch_reduction": "mean_of_fixed_weighted_example_losses",
        "positive_weight": sample_weight_for_rule(rule, True, supervision),
        "negative_weight": sample_weight_for_rule(rule, False, supervision),
        "weight_normalization": "population_mean_one_over_complete_truth_table",
        "accumulation_scaling": "Trainer (exactly once)",
        "model_accepts_loss_kwargs": False,
        "validation_supervision": "raw",
        "target_length_weighting_changed": False,
    }


def validate_training_batches(
    rows: int, batch_size: int, accumulation_steps: int, world_size: int = 1
) -> None:
    """Fail closed on partial updates and distributed runs not covered here.

    Microbatch means reproduce the full effective-batch objective when each
    update has equally sized microbatches. All current M1 profiles satisfy this.
    Reject remainders rather than silently dropping samples or depending on
    version-specific handling of the final accumulation window.
    """
    if rows <= 0 or batch_size <= 0 or accumulation_steps <= 0:
        raise ValueError("Training rows, batch size and accumulation steps must be positive")
    if world_size != 1:
        raise ValueError("M1 loss v2 currently supports single-process training only")
    effective_batch = batch_size * accumulation_steps
    if rows % effective_batch:
        raise ValueError(
            f"Training rows ({rows}) must be divisible by effective batch ({effective_batch}); "
            "partial accumulation updates are unsupported and no rows will be dropped"
        )


def completion_loss(logits: Any, labels: Any, sample_weight: Any) -> Any:
    """Completion-only, shifted CE; fixed row weights; no GA scaling here."""
    import torch
    import torch.nn.functional as functional

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected aligned logits [batch, length, vocab] and labels [batch, length]")
    batch_size = labels.shape[0]
    if not batch_size or logits.shape[1] < 2:
        raise ValueError("Loss requires a nonempty batch with at least two sequence positions")
    labels = labels.to(logits.device)
    weights = sample_weight.to(device=logits.device, dtype=torch.float32)
    if weights.shape != (batch_size,) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("sample_weight must have one finite positive value per example")

    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(IGNORE_INDEX)
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Every example must contain at least one shifted supervised token")

    # Compute CE in fp32 only for supervised positions, avoiding a full fp32
    # copy of long-prompt logits. Prompt and padding positions have no loss.
    token_losses = functional.cross_entropy(
        logits[:, :-1, :][mask].float(), shift_labels[mask], reduction="none"
    )
    owners = torch.arange(batch_size, device=logits.device)[:, None].expand_as(shift_labels)[mask]
    sums = torch.zeros(batch_size, dtype=token_losses.dtype, device=logits.device).scatter_add(
        0, owners, token_losses
    )
    per_example = sums / counts.to(sums.dtype)
    return (per_example * weights).mean()
