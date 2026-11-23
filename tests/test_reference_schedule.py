from __future__ import annotations

import pytest

from soft_prompt_distillation.config import ScheduleSpec
from soft_prompt_distillation.schedule import learning_rate_for_update


def test_constant_schedule_has_explicit_one_based_warmup() -> None:
    schedule = ScheduleSpec(name="constant", warmup_steps=4, min_lr_ratio=1.0)
    observed = [
        learning_rate_for_update(
            base_learning_rate=0.01,
            schedule=schedule,
            update_step=step,
            total_updates=8,
        )
        for step in range(1, 9)
    ]
    assert observed == pytest.approx(
        [0.0025, 0.005, 0.0075, 0.01, 0.01, 0.01, 0.01, 0.01]
    )


def test_cosine_schedule_reaches_base_after_warmup_and_minimum_at_end() -> None:
    schedule = ScheduleSpec(name="cosine", warmup_steps=2, min_lr_ratio=0.1)
    observed = [
        learning_rate_for_update(
            base_learning_rate=0.01,
            schedule=schedule,
            update_step=step,
            total_updates=6,
        )
        for step in range(1, 7)
    ]
    assert observed[0] == pytest.approx(0.005)
    assert observed[1] == pytest.approx(0.01)
    assert observed[2] == pytest.approx(0.01)
    assert observed[-1] == pytest.approx(0.001)
    assert observed[2:] == sorted(observed[2:], reverse=True)


def test_schedule_rejects_ambiguous_step_zero() -> None:
    schedule = ScheduleSpec(name="constant", warmup_steps=0, min_lr_ratio=1.0)
    with pytest.raises(ValueError, match="update_step"):
        learning_rate_for_update(
            base_learning_rate=0.01,
            schedule=schedule,
            update_step=0,
            total_updates=10,
        )
