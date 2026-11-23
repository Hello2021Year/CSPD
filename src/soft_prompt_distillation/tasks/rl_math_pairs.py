"""Adapters for published math-RL prompt artifacts used by model pairs."""

from __future__ import annotations

import hashlib
from typing import Any

from .base import DatasetSource
from .deepscaler_math import DeepScalerMathTask
from .olmo_math import OlmoMathTask


def _identity(problem: str) -> str:
    return hashlib.sha256(problem.encode()).hexdigest()


def _last_user_content(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    for message in reversed(value):
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and message["content"]
        ):
            return message["content"]
    return None


def _reward_answer(row: dict[str, Any]) -> Any:
    reward = row.get("reward_model")
    return reward.get("ground_truth") if isinstance(reward, dict) else None


def _normalized(problem: Any, answer: Any, question_id: Any, label: str) -> dict[str, Any]:
    if not isinstance(problem, str) or not problem:
        raise ValueError(f"{label} sampling row has no problem")
    if answer is None:
        raise ValueError(f"{label} sampling row has no answer")
    if question_id is None:
        question_id = _identity(problem)
    return {
        "problem": problem,
        "answer": str(answer),
        "question_id": str(question_id),
    }


class KlearMathTask(OlmoMathTask):
    sampling_dataset = DatasetSource(
        repo="Kwai-Klear/KlearReasoner-MathSub-30K",
        filename="train_math_30K.json",
        split="train",
        domain="math",
    )

    def normalize_sampling_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return _normalized(
            _last_user_content(row.get("prompt")),
            _reward_answer(row),
            row.get("__index_level_0__"),
            "Klear",
        )


class PoroMathTask(KlearMathTask):
    """Poro's published RL recipe reuses Klear MathSub-30K."""


class EurusMathTask(OlmoMathTask):
    _PUBLISHED_USER_SUFFIX = (
        "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    )
    sampling_dataset = DatasetSource(
        repo="PRIME-RL/Eurus-2-RL-Data",
        filename="train.parquet",
        split="train",
        domain="math",
    )

    def include_sampling_row(self, row: dict[str, Any]) -> bool:
        return row.get("ability") == "math"

    def normalize_sampling_row(self, row: dict[str, Any]) -> dict[str, Any]:
        # Eurus-2-RL-Data ships extra_info={"index": 0, "split": "dummy"} for
        # every one of its 480,537 rows, so that field cannot identify a
        # question. Using it collapsed every trajectory onto question_id "0",
        # and max_trajectories_per_question then reduced a 17,019-trajectory
        # snapshot to 2. Derive the identity from the problem text instead.
        problem = _last_user_content(row.get("prompt"))
        if not isinstance(problem, str) or not problem.endswith(
            self._PUBLISHED_USER_SUFFIX
        ):
            raise ValueError("Eurus sampling row has no published user suffix")
        problem = problem[: -len(self._PUBLISHED_USER_SUFFIX)]
        return _normalized(
            problem,
            _reward_answer(row),
            None,
            "Eurus",
        )


class IntellectMathTask(EurusMathTask):
    """INTELLECT-MATH reports using the same RL setup/data as Eurus-2."""


class MimoDeepScalerMathTask(OlmoMathTask):
    """Use DeepScaleR prompts as the disclosed substitute for MiMo's private pool."""

    sampling_dataset = DeepScalerMathTask.sampling_dataset
    sampling_datasets = (sampling_dataset,)

    def include_sampling_row(self, row: dict[str, Any]) -> bool:
        return True

    def normalize_sampling_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return DeepScalerMathTask.normalize_sampling_row(self, row)


__all__ = [
    "EurusMathTask",
    "IntellectMathTask",
    "KlearMathTask",
    "MimoDeepScalerMathTask",
    "PoroMathTask",
]
