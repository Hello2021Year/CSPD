"""Explicit one-based learning-rate schedules for optimizer updates."""

from __future__ import annotations

import math

from .config import ScheduleSpec


def learning_rate_for_update(
    *,
    base_learning_rate: float,
    schedule: ScheduleSpec,
    update_step: int,
    total_updates: int,
) -> float:
    """Return the LR used by the named one-based optimizer update.

    Warmup update 1 uses ``base_lr / warmup_steps`` and the final warmup update
    uses ``base_lr``.  Cosine decay begins after warmup and reaches the declared
    minimum exactly on ``total_updates``.  The caller sets this value immediately
    before ``optimizer.step()``; there is no scheduler-call-order ambiguity.
    """

    if base_learning_rate <= 0 or not math.isfinite(base_learning_rate):
        raise ValueError("base_learning_rate must be positive and finite")
    if not 1 <= update_step <= total_updates:
        raise ValueError("update_step must be in [1, total_updates]")
    schedule.validate("schedule", total_updates)
    if schedule.warmup_steps and update_step <= schedule.warmup_steps:
        return base_learning_rate * update_step / schedule.warmup_steps
    if schedule.name == "constant":
        return base_learning_rate
    first_decay_step = schedule.warmup_steps + 1
    decay_updates = total_updates - schedule.warmup_steps
    if decay_updates <= 1:
        progress = 1.0
    else:
        progress = (update_step - first_decay_step) / (decay_updates - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    ratio = schedule.min_lr_ratio + (1.0 - schedule.min_lr_ratio) * cosine
    return base_learning_rate * ratio


__all__ = ["learning_rate_for_update"]
