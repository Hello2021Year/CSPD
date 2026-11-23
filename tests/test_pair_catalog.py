"""Every config in configs/ parses, names a registered task and matches Table 1."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from yaml.tokens import FlowMappingStartToken, FlowSequenceStartToken

from soft_prompt_distillation.config import load_config
from soft_prompt_distillation.tasks import get_task


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted(
    path
    for path in (ROOT / "configs").rglob("*.yaml")
    if path.name != "task-template.yaml"
)

# Table 1 settings: config -> (task adapter, evaluation benchmarks).
PAPER_CONTRACTS = {
    "math/mimo-7b.yaml": ("mimo_math_deepscaler", {"aime_2024", "aime_2025", "math_500"}),
    "math/gpt-oss-swallow-20b.yaml": ("gpt_oss_swallow_math", None),
    "math/qwen3-swallow-8b.yaml": ("dolci_think_math", {"aime_2024", "aime_2025", "math_500"}),
    "math/olmo3-7b.yaml": ("olmo_math", {"math_500", "aime_2024", "aime_2025"}),
    "math/nemotron-cascade-8b.yaml": ("nemotron_cascade_math", {"aime_2024", "aime_2025", "math_500"}),
    "math/qwen3-swallow-30b-a3b.yaml": ("dolci_think_math", {"aime_2024", "aime_2025", "math_500"}),
    "math/eurus-2-7b.yaml": ("eurus_math", {"aime_2024", "math_500", "amc_2022_2023", "olympiad_bench"}),
    "math/intellect-math-7b.yaml": ("intellect_math", {"aime_2024", "math_500", "amc_2022_2023", "olympiad_bench"}),
    "math/poro-2-8b.yaml": ("poro_math", {"math_500", "aime_2025", "aime_2026"}),
    "math/klear-reasoner-8b.yaml": ("klear_math", {"aime_2024", "aime_2025", "hmmt_feb_2025"}),
    "code/mimo-7b.yaml": ("mimo_areal_proxy_code", None),
    "code/nemotron-cascade-8b.yaml": ("nemotron_klear_proxy_code", None),
    "code/klear-reasoner-8b.yaml": ("klear_code", None),
    "code/areal-boba2-14b.yaml": ("areal_code", None),
    "code/mellum2-12b-a2p5b.yaml": ("mellum_klear_proxy_code", None),
    "code/areal-boba2-8b.yaml": ("areal_code", None),
    "code/olmo3-7b.yaml": ("olmo_code", None),
}


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_config_is_block_style_and_valid(config_path: Path) -> None:
    tokens = yaml.scan(config_path.read_text(encoding="utf-8"))
    assert not [
        token
        for token in tokens
        if isinstance(token, (FlowMappingStartToken, FlowSequenceStartToken))
    ], f"{config_path} contains flow-style YAML"
    config = load_config(config_path)
    get_task(config.task)
    assert (config.student.checkpoint, config.student.subfolder) != (
        config.teacher.checkpoint,
        config.teacher.subfolder,
    )


def test_every_table1_setting_has_a_config() -> None:
    for name, (task_name, benchmarks) in PAPER_CONTRACTS.items():
        config = load_config(ROOT / "configs" / name)
        assert config.task == task_name, name
        task = get_task(task_name)
        assert set(config.data.evaluation) <= set(task.evaluation_datasets), name
        if benchmarks is not None:
            assert set(config.data.evaluation) == benchmarks, name


def test_prompt_length_ablation_only_changes_prompt_length() -> None:
    reference = load_config(ROOT / "configs/ablations/gpt-oss-prompt-length-256.yaml")
    for length in (2, 8, 32, 128, 512):
        config = load_config(ROOT / f"configs/ablations/gpt-oss-prompt-length-{length}.yaml")
        assert config.distillation.prompt_length == length
        assert replace(
            config, name=reference.name, distillation=reference.distillation
        ) == reference
