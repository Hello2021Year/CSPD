from __future__ import annotations

import copy

import pytest

from soft_prompt_distillation.config import ConfigError, parse_config


def valid_config_dict() -> dict:
    generation = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 4096,
        "samples_per_prompt": 1,
        "provenance": "official model-card evaluation recipe, checked 2026-08-12",
    }
    selection = {
        "num_prompts": 128,
    }
    return {
        "schema_version": 1,
        "name": "student-teacher-code-reference",
        "task": "example",
        "models": {
            "student": "org/student",
            "teacher": "org/teacher",
        },
        "template": {
            "source": "teacher",
            "system": {"mode": "text", "text": "Solve carefully."},
            "user": {
                "field": "problem",
                "prefix": "<|soft_prompt|>Question: ",
                "suffix": "",
            },
            "add_generation_prompt": True,
        },
        "data": {
            "sampling": selection,
            "evaluation": {
                "held_out": {**selection, "num_prompts": 64},
            },
        },
        "sampling": {
            "generation": {
                "provenance": "teacher RL training recipe",
            },
        },
        "distillation": {
            "objective": "response_sft",
            "attention_implementation": "flash_attention_2",
            "activation_checkpointing": True,
            "checkpoint_every_updates": 25,
            "prompt_length": 128,
            "prompt_init": {"method": "gaussian", "std": 0.02},
            "insertion": {
                "mode": "placeholder",
                "marker": "<|soft_prompt|>",
                "sentinel_id": -200,
                "occurrence": "exactly_one",
            },
            "optimizer": "adamw",
            "learning_rate": 0.003,
            "schedule": {"name": "cosine", "warmup_steps": 10, "min_lr_ratio": 0.0},
            "effective_batch_size": 1,
            "max_steps": 100,
            "weight_decay": 0.0,
            "max_grad_norm": None,
        },
        "evaluation": {
            "arms": {
                "student": {"generation": generation},
                "teacher": {"generation": generation},
                "student_soft_prompt": {"generation": generation},
            },
        },
    }


def test_config_freezes_template_sampling_training_and_evaluation_semantics() -> None:
    config = parse_config(valid_config_dict())
    assert config.student.trust_remote_code is False
    assert config.teacher.trust_remote_code is False
    assert config.student.context_length is None
    assert config.teacher.context_length is None
    assert config.student.config_overrides == {}
    assert config.teacher.config_overrides == {}
    assert config.template.source == "teacher"
    assert config.evaluation.template is None
    assert config.template.render_mode == "chat"
    assert config.distillation.insertion.sentinel_id == -200
    assert config.data.sampling.selection_seed == 42
    assert config.data.evaluation["held_out"].selection_seed == 42
    assert config.sampling.trajectory_filter == "correct_only"
    assert config.sampling.generation.temperature == 1.0
    assert config.sampling.generation.top_p == 1.0
    assert config.sampling.generation.top_k == -1
    assert config.sampling.generation.repetition_penalty == 1.0
    assert config.sampling.generation.max_new_tokens == 32000
    assert config.sampling.generation.samples_per_prompt == 1
    assert config.distillation.prompt_init.seed == 42
    assert config.distillation.attention_implementation == "flash_attention_2"
    assert config.distillation.activation_checkpointing is True
    assert config.distillation.checkpoint_every_updates == 25
    assert config.distillation.max_sequence_length is None
    assert config.distillation.overlength_policy == "error"
    assert config.distillation.max_trajectories_per_question is None
    assert config.distillation.micro_batch_size == 1
    assert config.distillation.cross_entropy_backend == "torch"
    assert config.distillation.loss_normalization == "token"
    assert config.distillation.batch_seed == 42
    assert config.sampled_trajectory_count == 128


def test_evaluation_can_override_shared_rollout_template() -> None:
    raw = valid_config_dict()
    raw["evaluation"]["template"] = {
        "source": "teacher",
        "system": {"mode": "text", "text": "Evaluation-only system."},
        "user": {
            "field": "problem",
            "prefix": "<|soft_prompt|>",
            "suffix": "",
        },
        "add_generation_prompt": True,
    }

    config = parse_config(raw)

    assert config.evaluation.template is not None
    assert config.evaluation.template.system.text == "Evaluation-only system."


def test_evaluation_template_must_use_shared_template_tokenizer() -> None:
    raw = valid_config_dict()
    raw["evaluation"]["template"] = copy.deepcopy(raw["template"])
    raw["evaluation"]["template"]["source"] = "student"

    with pytest.raises(ConfigError, match="source must match"):
        parse_config(raw)


def test_flex_attention_is_a_valid_training_backend() -> None:
    raw = valid_config_dict()
    raw["distillation"]["attention_implementation"] = "flex_attention"

    config = parse_config(raw)

    assert config.distillation.attention_implementation == "flex_attention"


def test_model_specific_trust_remote_code_is_explicit_opt_in() -> None:
    raw = valid_config_dict()
    raw["models"]["trust_remote_code"] = {
        "student": True,
        "teacher": True,
    }

    config = parse_config(raw)

    assert config.student.trust_remote_code is True
    assert config.teacher.trust_remote_code is True


def test_model_revisions_are_explicit_per_role() -> None:
    raw = valid_config_dict()
    raw["models"]["revision"] = {
        "student": "student-commit",
        "teacher": "teacher-commit",
    }

    config = parse_config(raw)

    assert config.student.revision == "student-commit"
    assert config.teacher.revision == "teacher-commit"


def test_model_revision_rejects_unknown_role() -> None:
    raw = valid_config_dict()
    raw["models"]["revision"] = {"judge": "commit"}

    with pytest.raises(ConfigError, match="models.revision contains unknown fields"):
        parse_config(raw)


def test_model_subfolders_define_distinct_stages_in_one_hub_repository() -> None:
    raw = valid_config_dict()
    raw["models"]["student"] = "org/intermediate-checkpoints"
    raw["models"]["teacher"] = "org/intermediate-checkpoints"
    raw["models"]["subfolder"] = {
        "student": "stage-sft",
        "teacher": "stage-math-rl",
    }

    config = parse_config(raw)

    assert config.student.subfolder == "stage-sft"
    assert config.teacher.subfolder == "stage-math-rl"


@pytest.mark.parametrize("value", ["/absolute", "../parent", "a//b", "."])
def test_model_subfolder_must_be_safe_relative_hub_path(value: str) -> None:
    raw = valid_config_dict()
    raw["models"]["subfolder"] = {"student": value}

    with pytest.raises(ConfigError, match="relative Hub subfolder"):
        parse_config(raw)


@pytest.mark.parametrize("value", [1, "true", None])
def test_model_trust_remote_code_must_be_boolean(value: object) -> None:
    raw = valid_config_dict()
    raw["models"]["trust_remote_code"] = {"teacher": value}

    with pytest.raises(ConfigError, match="models.trust_remote_code.teacher"):
        parse_config(raw)


def test_model_trust_remote_code_rejects_unknown_role() -> None:
    raw = valid_config_dict()
    raw["models"]["trust_remote_code"] = {"judge": True}

    with pytest.raises(ConfigError, match="unknown fields: judge"):
        parse_config(raw)


def test_model_context_and_config_overrides_are_explicit_per_role() -> None:
    raw = valid_config_dict()
    raw["models"]["context_length"] = {
        "student": 131072,
        "teacher": 40960,
    }
    raw["models"]["config_overrides"] = {
        "student": {
            "max_position_embeddings": 131072,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
        },
        "teacher": {"max_position_embeddings": 40960},
    }

    config = parse_config(raw)

    assert config.student.context_length == 131072
    assert config.student.config_overrides["rope_scaling"]["rope_type"] == "yarn"
    assert config.teacher.context_length == 40960


@pytest.mark.parametrize("value", [0, -1, "40960", True])
def test_model_context_length_must_be_positive_integer(value: object) -> None:
    raw = valid_config_dict()
    raw["models"]["context_length"] = {"student": value}

    with pytest.raises(ConfigError, match="models.context_length.student"):
        parse_config(raw)


def test_model_context_length_must_match_position_override() -> None:
    raw = valid_config_dict()
    raw["models"]["context_length"] = {"student": 40960}
    raw["models"]["config_overrides"] = {"student": {"max_position_embeddings": 131072}}

    with pytest.raises(ConfigError, match="must equal"):
        parse_config(raw)


def test_model_config_overrides_reject_loader_owned_fields() -> None:
    raw = valid_config_dict()
    raw["models"]["config_overrides"] = {"student": {"dtype": "float16"}}

    with pytest.raises(ConfigError, match="loader-owned fields: dtype"):
        parse_config(raw)


def test_model_specific_template_and_stop_fields_are_explicit() -> None:
    raw = valid_config_dict()
    raw["template"]["chat_template_kwargs"] = {"enable_thinking": True}
    raw["template"]["assistant_prefix"] = "<think>\n"
    raw["template"]["chat_template_replacements"] = {
        'strftime_now("%Y-%m-%d")': '"2026-08-21"'
    }
    for arm in raw["evaluation"]["arms"].values():
        arm["generation"]["stop_token_ids"] = [151645]
        arm["generation"]["stop_strings"] = ["</final>"]

    config = parse_config(raw)

    assert config.template.chat_template_kwargs == {"enable_thinking": True}
    assert config.template.assistant_prefix == "<think>\n"
    assert config.template.chat_template_replacements == {
        'strftime_now("%Y-%m-%d")': '"2026-08-21"'
    }
    assert config.evaluation.arms["student"].generation.stop_token_ids == (151645,)
    assert config.evaluation.arms["student"].generation.stop_strings == ("</final>",)


def test_soft_prompt_marker_may_be_in_chat_template_replacement_output() -> None:
    raw = valid_config_dict()
    raw["template"]["user"]["prefix"] = "Question: "
    raw["template"]["chat_template_replacements"] = {
        '{{- model_identity + "\\n" }}': (
            '{{- "<|soft_prompt|>" + model_identity + "\\n" }}'
        )
    }

    config = parse_config(raw)

    assert "<|soft_prompt|>" in next(
        iter(config.template.chat_template_replacements.values())
    )


def test_benchmark_generation_override_must_name_selected_dataset() -> None:
    raw = valid_config_dict()
    override = copy.deepcopy(raw["evaluation"]["arms"]["student"]["generation"])
    override["samples_per_prompt"] = 5
    raw["evaluation"]["benchmarks"] = {"held_out": {"generation": override}}

    config = parse_config(raw)
    assert config.evaluation.benchmarks["held_out"].samples_per_prompt == 5

    raw["evaluation"]["benchmarks"]["not_selected"] = {"generation": override}
    with pytest.raises(ConfigError, match="absent from data.evaluation"):
        parse_config(raw)


def test_generation_rejects_unknown_fields() -> None:
    raw = valid_config_dict()
    raw["evaluation"]["arms"]["student"]["generation"]["typo"] = True
    with pytest.raises(ConfigError, match="unknown fields: typo"):
        parse_config(raw)


@pytest.mark.parametrize("value", ["flash_attention_3", "bad", 1, None])
def test_train_attention_implementation_is_validated(value: object) -> None:
    raw = valid_config_dict()
    raw["distillation"]["attention_implementation"] = value
    with pytest.raises(ConfigError, match="attention_implementation"):
        parse_config(raw)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_activation_checkpointing_must_be_boolean(value: object) -> None:
    raw = valid_config_dict()
    raw["distillation"]["activation_checkpointing"] = value
    with pytest.raises(ConfigError, match="activation_checkpointing"):
        parse_config(raw)


def test_weight_decay_accepts_positive_finite_value() -> None:
    raw = valid_config_dict()
    raw["distillation"]["weight_decay"] = 0.0001

    assert parse_config(raw).distillation.weight_decay == pytest.approx(0.0001)


@pytest.mark.parametrize("value", [-0.0001, float("nan"), float("inf"), float("-inf")])
def test_weight_decay_must_be_nonnegative_and_finite(value: float) -> None:
    raw = valid_config_dict()
    raw["distillation"]["weight_decay"] = value

    with pytest.raises(ConfigError, match="weight_decay"):
        parse_config(raw)


@pytest.mark.parametrize("value", [0, -1, True, "10"])
def test_checkpoint_interval_must_be_a_positive_integer(value: object) -> None:
    raw = valid_config_dict()
    raw["distillation"]["checkpoint_every_updates"] = value
    with pytest.raises(ConfigError, match="checkpoint_every_updates"):
        parse_config(raw)


def test_sequence_length_filter_config() -> None:
    raw = valid_config_dict()
    raw["distillation"]["max_sequence_length"] = 65536
    raw["distillation"]["overlength_policy"] = "filter"
    distillation = parse_config(raw).distillation
    assert distillation.max_sequence_length == 65536
    assert distillation.overlength_policy == "filter"


@pytest.mark.parametrize("value", [0, -1, True, "65536"])
def test_max_sequence_length_must_be_positive_integer(value: object) -> None:
    raw = valid_config_dict()
    raw["distillation"]["max_sequence_length"] = value
    with pytest.raises(ConfigError, match="max_sequence_length"):
        parse_config(raw)


@pytest.mark.parametrize("value", ["drop", "truncate", "", None])
def test_overlength_policy_is_validated(value: object) -> None:
    raw = valid_config_dict()
    raw["distillation"]["overlength_policy"] = value
    with pytest.raises(ConfigError, match="overlength_policy"):
        parse_config(raw)


def test_per_question_trajectory_cap_is_configurable() -> None:
    raw = valid_config_dict()
    raw["distillation"]["max_trajectories_per_question"] = 2
    assert parse_config(raw).distillation.max_trajectories_per_question == 2


@pytest.mark.parametrize("value", [0, -1, True, "2"])
def test_per_question_trajectory_cap_must_be_positive(value: object) -> None:
    raw = valid_config_dict()
    raw["distillation"]["max_trajectories_per_question"] = value
    with pytest.raises(ConfigError, match="max_trajectories_per_question"):
        parse_config(raw)


def test_training_optimization_defaults_preserve_old_configs() -> None:
    raw = valid_config_dict()
    del raw["distillation"]["attention_implementation"]
    del raw["distillation"]["activation_checkpointing"]
    del raw["distillation"]["checkpoint_every_updates"]

    config = parse_config(raw)

    assert config.distillation.attention_implementation == "sdpa"
    assert config.distillation.activation_checkpointing is False
    assert config.distillation.checkpoint_every_updates is None


def test_evaluation_backend_switch_is_not_supported() -> None:
    raw = valid_config_dict()
    raw["evaluation"]["backend"] = "transformers"
    with pytest.raises(ConfigError, match="unknown fields: backend"):
        parse_config(raw)


@pytest.mark.parametrize("schema_version", [None, "1", True])
def test_schema_version_must_be_an_integer(schema_version: object) -> None:
    raw = valid_config_dict()
    raw["schema_version"] = schema_version
    with pytest.raises(ConfigError, match="schema_version must be an integer"):
        parse_config(raw)


def test_schema_version_must_be_supported() -> None:
    raw = valid_config_dict()
    raw["schema_version"] = 2
    with pytest.raises(ConfigError, match="schema_version must be 1"):
        parse_config(raw)


def test_selection_seed_can_override_default() -> None:
    raw = valid_config_dict()
    raw["data"]["sampling"]["selection_seed"] = 7
    assert parse_config(raw).data.sampling.selection_seed == 7


def test_all_selects_the_full_dataset_symbolically() -> None:
    raw = valid_config_dict()
    raw["data"]["sampling"]["num_prompts"] = "all"
    raw["data"]["evaluation"]["held_out"]["num_prompts"] = "all"
    config = parse_config(raw)
    assert config.data.sampling.num_prompts == "all"
    assert config.data.evaluation["held_out"].num_prompts == "all"
    assert config.sampled_trajectory_count == "all"


def test_negative_num_prompts_is_rejected() -> None:
    raw = valid_config_dict()
    raw["data"]["sampling"]["num_prompts"] = -1
    with pytest.raises(ConfigError, match="positive integer"):
        parse_config(raw)


def test_config_requires_one_template_owned_b_marker() -> None:
    raw = valid_config_dict()
    raw["template"]["user"]["prefix"] = "Question: "
    with pytest.raises(ConfigError, match="exactly once"):
        parse_config(raw)


def test_config_rejects_legacy_stage_template_selection() -> None:
    raw = valid_config_dict()
    raw["sampling"]["template"] = "B"
    with pytest.raises(ConfigError, match="redundant"):
        parse_config(raw)


def test_config_rejects_legacy_multiple_templates() -> None:
    raw = valid_config_dict()
    raw["templates"] = {"A": raw.pop("template")}
    with pytest.raises(ConfigError, match="legacy multiple templates"):
        parse_config(raw)


@pytest.mark.parametrize(
    "field", ["name", "filename", "split", "domain", "revision", "source_sha256"]
)
def test_dataset_source_is_not_manually_configured(field) -> None:
    raw = valid_config_dict()
    raw["data"]["sampling"][field] = "manual-value"
    with pytest.raises(ConfigError, match="task-owned or unknown"):
        parse_config(raw)


def test_legacy_sampling_dataset_is_rejected() -> None:
    raw = valid_config_dict()
    raw["sampling_dataset"] = raw["data"].pop("sampling")
    with pytest.raises(ConfigError, match="moved to task adapter"):
        parse_config(raw)


def test_template_hash_is_not_manually_configured() -> None:
    raw = valid_config_dict()
    raw["template"]["chat_template_sha256"] = "a" * 64
    with pytest.raises(ConfigError, match="runtime evidence"):
        parse_config(raw)


def test_distillation_filter_is_rejected_as_redundant() -> None:
    raw = valid_config_dict()
    raw["distillation"]["trajectory_filter"] = "correct_only"
    with pytest.raises(ConfigError, match="redundant"):
        parse_config(raw)


def test_sampling_defaults_can_be_overridden() -> None:
    raw = valid_config_dict()
    raw["sampling"]["trajectory_filter"] = "all"
    raw["sampling"]["generation"]["temperature"] = 0.7
    raw["sampling"]["generation"]["top_p"] = 0.9
    raw["sampling"]["generation"]["top_k"] = 20
    raw["sampling"]["generation"]["repetition_penalty"] = 1.05
    raw["sampling"]["generation"]["max_new_tokens"] = 64
    raw["sampling"]["generation"]["samples_per_prompt"] = 3
    config = parse_config(raw)
    assert config.sampling.trajectory_filter == "all"
    assert config.sampling.generation.temperature == 0.7
    assert config.sampling.generation.top_p == 0.9
    assert config.sampling.generation.top_k == 20
    assert config.sampling.generation.repetition_penalty == 1.05
    assert config.sampling.generation.max_new_tokens == 64
    assert config.sampling.generation.samples_per_prompt == 3


@pytest.mark.parametrize("value", [-2, 0, 1.5, True])
def test_generation_rejects_invalid_top_k(value) -> None:
    raw = valid_config_dict()
    raw["evaluation"]["arms"]["student"]["generation"]["top_k"] = value
    with pytest.raises(ConfigError, match="top_k"):
        parse_config(raw)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), True])
def test_generation_rejects_invalid_repetition_penalty(value) -> None:
    raw = valid_config_dict()
    raw["evaluation"]["arms"]["student"]["generation"]["repetition_penalty"] = value
    with pytest.raises(ConfigError, match="repetition_penalty"):
        parse_config(raw)


def test_prompt_init_seed_can_override_default() -> None:
    raw = valid_config_dict()
    raw["distillation"]["prompt_init"]["seed"] = 7
    assert parse_config(raw).distillation.prompt_init.seed == 7


def test_config_rejects_extra_evaluation_arm() -> None:
    raw = valid_config_dict()
    raw["evaluation"]["arms"]["student_template_b_control"] = copy.deepcopy(
        raw["evaluation"]["arms"]["student_soft_prompt"]
    )
    with pytest.raises(
        ConfigError, match="exactly student, teacher, student_soft_prompt"
    ):
        parse_config(raw)


def test_generation_provenance_is_optional() -> None:
    raw = valid_config_dict()
    del raw["evaluation"]["arms"]["teacher"]["generation"]["provenance"]
    assert parse_config(raw).evaluation.arms["teacher"].generation.provenance is None


def test_generation_provenance_must_be_nonempty_when_present() -> None:
    raw = valid_config_dict()
    raw["evaluation"]["arms"]["teacher"]["generation"]["provenance"] = ""
    with pytest.raises(ConfigError, match="provenance"):
        parse_config(raw)


def test_student_and_soft_prompt_generation_must_match() -> None:
    raw = valid_config_dict()
    raw["evaluation"]["arms"]["student_soft_prompt"]["generation"] = copy.deepcopy(
        raw["evaluation"]["arms"]["student"]["generation"]
    )
    raw["evaluation"]["arms"]["student_soft_prompt"]["generation"]["temperature"] = 0.7
    with pytest.raises(ConfigError, match="same generation recipe"):
        parse_config(raw)


def test_learning_rate_schedule_is_fully_declared() -> None:
    raw = valid_config_dict()
    raw["distillation"]["schedule"]["warmup_steps"] = 101
    with pytest.raises(ConfigError, match="warmup_steps"):
        parse_config(raw)


def test_distillation_accepts_effective_batch_above_one() -> None:
    raw = valid_config_dict()
    raw["distillation"]["effective_batch_size"] = 8
    raw["distillation"]["micro_batch_size"] = 2
    config = parse_config(raw)
    assert config.distillation.effective_batch_size == 8
    assert config.distillation.micro_batch_size == 2


def test_micro_batch_must_not_exceed_effective_batch() -> None:
    raw = valid_config_dict()
    raw["distillation"]["micro_batch_size"] = 2
    with pytest.raises(ConfigError, match="must not exceed"):
        parse_config(raw)


def test_epochs_resolve_against_frozen_trajectory_count() -> None:
    raw = valid_config_dict()
    del raw["distillation"]["max_steps"]
    raw["distillation"]["epochs"] = 3
    distillation = parse_config(raw).distillation
    assert distillation.updates_for_trajectory_count(7) == 21


def test_epoch_count_drops_incomplete_global_batch() -> None:
    raw = valid_config_dict()
    raw["distillation"]["effective_batch_size"] = 3
    del raw["distillation"]["max_steps"]
    raw["distillation"]["epochs"] = 2
    distillation = parse_config(raw).distillation
    assert distillation.updates_for_trajectory_count(7) == 4


def test_drop_last_rejects_too_few_trajectories() -> None:
    raw = valid_config_dict()
    raw["distillation"]["effective_batch_size"] = 8
    distillation = parse_config(raw).distillation
    with pytest.raises(ConfigError, match="drop_last"):
        distillation.updates_for_trajectory_count(7)


def test_epochs_and_max_steps_are_mutually_exclusive() -> None:
    raw = valid_config_dict()
    raw["distillation"]["epochs"] = 1
    with pytest.raises(ConfigError, match="exactly one"):
        parse_config(raw)


def test_stopping_rule_is_required() -> None:
    raw = valid_config_dict()
    del raw["distillation"]["max_steps"]
    with pytest.raises(ConfigError, match="exactly one"):
        parse_config(raw)


def test_evaluation_supports_multiple_samples_per_prompt() -> None:
    raw = valid_config_dict()
    for arm in raw["evaluation"]["arms"].values():
        arm["generation"]["samples_per_prompt"] = 2

    config = parse_config(raw)

    assert config.evaluation.arms["teacher"].generation.samples_per_prompt == 2


def test_liger_backend_is_configurable() -> None:
    raw = valid_config_dict()
    raw["distillation"]["cross_entropy_backend"] = "liger_fused"

    distillation = parse_config(raw).distillation

    assert distillation.cross_entropy_backend == "liger_fused"


def test_cross_entropy_backend_is_validated() -> None:
    raw = valid_config_dict()
    raw["distillation"]["cross_entropy_backend"] = "approximate"
    with pytest.raises(ConfigError, match="cross_entropy_backend"):
        parse_config(raw)


def test_sequence_loss_normalization_is_configurable() -> None:
    raw = valid_config_dict()
    raw["distillation"]["loss_normalization"] = "sequence"

    distillation = parse_config(raw).distillation

    assert distillation.loss_normalization == "sequence"


def test_loss_normalization_is_validated() -> None:
    raw = valid_config_dict()
    raw["distillation"]["loss_normalization"] = "batch"
    with pytest.raises(ConfigError, match="loss_normalization"):
        parse_config(raw)
