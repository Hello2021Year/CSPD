"""The exporter must tolerate a published generation_config that cannot save.

transformers refuses to write a generation_config that sets temperature or top_p
without do_sample. allenai/Olmo-3-7B-Instruct-DPO ships exactly that shape, and
because the export runs once per soft-prompt checkpoint at the very end of the
pipeline, job 845 spent eight hours sampling, training and evaluating both
baselines before dying on it.

Of every student in configs/, only that one is affected -- the others either set
do_sample=True or leave temperature and top_p unset -- which is why this survived
until an eight-hour run hit it.
"""

from __future__ import annotations

import json

import pytest

# exporting.py imports torch, safetensors and transformers at module scope.
pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from soft_prompt_distillation.exporting import (
    _complete_rope_overrides,
    _repair_generation_config,
    _restore_legacy_rope_keys,
)


class _GenerationConfig:
    def __init__(self, **fields):
        self.temperature = fields.get("temperature")
        self.top_p = fields.get("top_p")
        self.do_sample = fields.get("do_sample", False)


class _Model:
    def __init__(self, generation_config):
        self.generation_config = generation_config


class _ModelConfig:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


def test_olmo3_shape_is_repaired():
    # The exact combination allenai/Olmo-3-7B-Instruct-DPO publishes.
    config = _GenerationConfig(temperature=0.6, top_p=0.95, do_sample=None)
    _repair_generation_config(_Model(config), "allenai/Olmo-3-7B-Instruct-DPO")

    assert config.do_sample is True
    # The published decoding recipe is kept, not discarded.
    assert config.temperature == 0.6
    assert config.top_p == 0.95


def test_a_consistent_config_is_left_alone():
    config = _GenerationConfig(temperature=0.6, top_p=0.95, do_sample=True)
    _repair_generation_config(_Model(config), "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    assert config.do_sample is True
    assert config.temperature == 0.6


def test_greedy_config_without_sampling_fields_is_left_alone():
    # Qwen2.5-Math-7B: do_sample False and no temperature/top_p. Forcing
    # do_sample=True here would invent a decoding mode the model never declared.
    config = _GenerationConfig(do_sample=False)
    _repair_generation_config(_Model(config), "Qwen/Qwen2.5-Math-7B")

    assert config.do_sample is False


def test_model_without_a_generation_config_is_not_an_error():
    _repair_generation_config(_Model(None), "some/model")


def test_legacy_rope_keys_are_restored_for_a_local_export(tmp_path):
    # Exactly what transformers 5.3 writes for allenai/Olmo-3-7B-Instruct-DPO:
    # the modern dict, with the legacy keys nulled out.
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "olmo3",
        "rope_theta": None,
        "rope_scaling": None,
        "rope_parameters": {
            "attention_factor": 1.2079441541679836,
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "yarn",
            "rope_theta": 500000,
        },
    }), encoding="utf-8")

    _restore_legacy_rope_keys(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["rope_theta"] == 500000
    assert saved["rope_scaling"]["rope_type"] == "yarn"
    assert saved["rope_scaling"]["factor"] == 8.0
    # rope_theta belongs at the top level, not inside the scaling dict.
    assert "rope_theta" not in saved["rope_scaling"]
    # SGLang's OLMo3 compatibility config validates modern rope_parameters
    # before setting max_position_embeddings.  The equivalent legacy fields
    # avoid that constructor-ordering bug, and Transformers can normalize them
    # again when loading the export.
    assert "rope_parameters" not in saved


def test_non_olmo3_export_keeps_modern_rope_parameters(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "rope_parameters": {
            "factor": 2.0,
            "rope_type": "yarn",
            "rope_theta": 1_000_000,
        },
    }), encoding="utf-8")

    _restore_legacy_rope_keys(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["rope_parameters"]["rope_theta"] == 1_000_000


def test_olmo3_max_positions_are_restored_from_the_base_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "olmo3",
        "rope_parameters": {
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "yarn",
            "rope_theta": 500000,
        },
    }), encoding="utf-8")

    _restore_legacy_rope_keys(
        tmp_path,
        _ModelConfig({"max_position_embeddings": 65536}),
    )

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["max_position_embeddings"] == 65536


def test_exported_max_positions_are_not_overwritten(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "max_position_embeddings": 32768,
        "rope_parameters": {"rope_theta": 10000.0},
    }), encoding="utf-8")

    _restore_legacy_rope_keys(
        tmp_path,
        _ModelConfig({"max_position_embeddings": 65536}),
    )

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["max_position_embeddings"] == 32768


def test_existing_legacy_keys_are_not_overwritten(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "rope_theta": 10000,
        "rope_scaling": {"rope_type": "linear", "factor": 2.0},
        "rope_parameters": {"rope_type": "yarn", "rope_theta": 500000, "factor": 8.0},
    }), encoding="utf-8")

    _restore_legacy_rope_keys(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["rope_theta"] == 10000
    assert saved["rope_scaling"]["rope_type"] == "linear"


def test_a_config_without_rope_parameters_is_untouched(tmp_path):
    original = {"model_type": "qwen2", "hidden_size": 1536}
    (tmp_path / "config.json").write_text(json.dumps(original), encoding="utf-8")

    _restore_legacy_rope_keys(tmp_path)

    assert json.loads((tmp_path / "config.json").read_text()) == original


def test_plain_rope_theta_does_not_invent_a_scaling_dict(tmp_path):
    # No yarn/linear scaling, just a theta: there is nothing to scale, so
    # rope_scaling must stay absent rather than becoming a one-key dict.
    (tmp_path / "config.json").write_text(json.dumps({
        "rope_parameters": {"rope_theta": 10000.0},
    }), encoding="utf-8")

    _restore_legacy_rope_keys(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["rope_theta"] == 10000.0
    assert saved.get("rope_scaling") is None


def test_partial_yarn_override_inherits_base_rope_theta():
    overrides = {
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 32768,
        }
    }
    base = _ModelConfig({
        "rope_parameters": {"rope_type": "default", "rope_theta": 1_000_000}
    })

    completed = _complete_rope_overrides(overrides, base)

    assert completed["rope_scaling"]["rope_theta"] == 1_000_000
    # Do not mutate the immutable run config while preparing loader kwargs.
    assert "rope_theta" not in overrides["rope_scaling"]


def test_explicit_rope_theta_override_is_preserved():
    overrides = {
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 2.0,
            "rope_theta": 500_000,
        }
    }
    base = _ModelConfig({"rope_theta": 1_000_000})

    completed = _complete_rope_overrides(overrides, base)

    assert completed["rope_scaling"]["rope_theta"] == 500_000
