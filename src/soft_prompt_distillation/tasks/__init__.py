"""Explicit registry for the small set of supported task adapters."""

from __future__ import annotations

from .base import CompletionParse, DatasetSource, GradeResult, TaskAdapter
from .code_tasks import (
    ArealCodeTask,
    KlearCodeTask,
    MellumCodeTask,
    MellumKlearProxyCodeTask,
    MimoCodeTask,
    MimoArealProxyCodeTask,
    NemotronCodeTask,
    NemotronKlearProxyCodeTask,
    OlmoCodeTask,
)
from .deepscaler_math import DeepScalerMathTask
from .new_architecture_math import (
    DolciThinkMathTask,
    GptOssSwallowMathTask,
    NemotronCascadeMathTask,
)
from .olmo_math import OlmoMathTask
from .rl_math_pairs import (
    EurusMathTask,
    IntellectMathTask,
    KlearMathTask,
    MimoDeepScalerMathTask,
    PoroMathTask,
)


TASKS: dict[str, TaskAdapter] = {
    "areal_code": ArealCodeTask(),
    "deepscaler_math": DeepScalerMathTask(),
    "dolci_think_math": DolciThinkMathTask(),
    "eurus_math": EurusMathTask(),
    "intellect_math": IntellectMathTask(),
    "klear_math": KlearMathTask(),
    "klear_code": KlearCodeTask(),
    "gpt_oss_swallow_math": GptOssSwallowMathTask(),
    "mellum_klear_proxy_code": MellumKlearProxyCodeTask(),
    "mimo_math_deepscaler": MimoDeepScalerMathTask(),
    "mimo_areal_proxy_code": MimoArealProxyCodeTask(),
    "nemotron_cascade_math": NemotronCascadeMathTask(),
    "nemotron_klear_proxy_code": NemotronKlearProxyCodeTask(),
    "olmo_math": OlmoMathTask(),
    "olmo_code": OlmoCodeTask(),
    "poro_math": PoroMathTask(),
}


def get_task(name: str) -> TaskAdapter:
    try:
        return TASKS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown task adapter {name!r}; available: {sorted(TASKS)}"
        ) from exc


__all__ = [
    "CompletionParse",
    "DatasetSource",
    "GradeResult",
    "TaskAdapter",
    "get_task",
]
