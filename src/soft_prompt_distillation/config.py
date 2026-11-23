"""Strict experiment configuration for off-policy soft-prompt distillation.

One template is shared by teacher sampling, distillation, and every evaluation
arm.  The selected pre-RL/post-RL model pair must share one tokenizer identity.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a config is incomplete or scientifically ambiguous."""


DEFAULT_SELECTION_SEED = 42
DEFAULT_SAMPLING_TRAJECTORY_FILTER = "correct_only"
DEFAULT_SAMPLING_TEMPERATURE = 1.0
DEFAULT_SAMPLING_TOP_P = 1.0
DEFAULT_SAMPLING_MAX_NEW_TOKENS = 32000
DEFAULT_SAMPLING_SAMPLES_PER_PROMPT = 1
DEFAULT_PROMPT_INIT_SEED = 42
DEFAULT_DISTILLATION_BATCH_SEED = 42
DEFAULT_DISTILLATION_MICRO_BATCH_SIZE = 1
DEFAULT_CROSS_ENTROPY_BACKEND = "torch"
DEFAULT_LOSS_NORMALIZATION = "token"
DEFAULT_TRAIN_ATTENTION_IMPLEMENTATION = "sdpa"
DEFAULT_ACTIVATION_CHECKPOINTING = False
DEFAULT_CHECKPOINT_EVERY_UPDATES = None


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return dict(value)


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _integer(value: Any, name: str) -> int:
    """Validate and narrow a dynamically loaded YAML value to ``int``."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    return value


def _optional_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, name)


def _number(value: Any, name: str) -> float:
    """Validate and narrow a YAML number, rejecting booleans as numbers."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric")
    return float(value)


def _optional_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _number(value, name)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be boolean")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a string or null")
    return value


def _plain_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a string")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{name} must be a list")
    return value


def _prompt_count(value: Any, name: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ConfigError(f"{name} must be a positive integer or all")
    return value


def _probability(value: Any, name: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric")
    result = float(value)
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not lower_ok or result > 1.0:
        bound = "[0, 1]" if allow_zero else "(0, 1]"
        raise ConfigError(f"{name} must be finite and in {bound}")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class ModelSpec:
    checkpoint: str
    revision: str | None = None
    subfolder: str | None = None
    trust_remote_code: bool = False
    context_length: int | None = None
    config_overrides: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True)
class SystemPromptSpec:
    mode: str
    text: str | None

    def validate(self, name: str) -> None:
        if self.mode not in {"model_default", "empty", "text"}:
            raise ConfigError(f"{name}.mode must be model_default, empty, or text")
        if self.mode == "text" and (not isinstance(self.text, str) or not self.text):
            raise ConfigError(f"{name}.text is required when mode=text")
        if self.mode != "text" and self.text not in (None, ""):
            raise ConfigError(f"{name}.text is only valid when mode=text")


@dataclasses.dataclass(frozen=True, slots=True)
class UserPromptSpec:
    field: str
    prefix: str
    suffix: str


@dataclasses.dataclass(frozen=True, slots=True)
class TemplateSpec:
    source: str
    render_mode: str
    system: SystemPromptSpec
    user: UserPromptSpec
    add_generation_prompt: bool
    chat_template_kwargs: Mapping[str, bool | int | float | str]
    chat_template_replacements: Mapping[str, str]
    assistant_prefix: str

    def validate(self, name: str) -> None:
        if self.source not in {"student", "teacher"}:
            raise ConfigError(f"{name}.source must be student or teacher")
        if self.render_mode not in {"chat", "raw"}:
            raise ConfigError(f"{name}.render_mode must be chat or raw")
        self.system.validate(f"{name}.system")
        _string(self.user.field, f"{name}.user.field")
        if not isinstance(self.user.prefix, str) or not isinstance(
            self.user.suffix, str
        ):
            raise ConfigError(f"{name}.user prefix and suffix must be strings")
        if not isinstance(self.add_generation_prompt, bool):
            raise ConfigError(f"{name}.add_generation_prompt must be boolean")
        for key, value in self.chat_template_kwargs.items():
            _string(key, f"{name}.chat_template_kwargs key")
            if key in {"messages", "tokenize", "add_generation_prompt"}:
                raise ConfigError(f"{name}.chat_template_kwargs cannot override {key}")
            if isinstance(value, (dict, list)) or value is None:
                raise ConfigError(f"{name}.chat_template_kwargs.{key} must be a scalar")
            if not isinstance(value, (bool, int, float, str)):
                raise ConfigError(f"{name}.chat_template_kwargs.{key} must be a scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise ConfigError(f"{name}.chat_template_kwargs.{key} must be finite")
        for old, new in self.chat_template_replacements.items():
            _string(old, f"{name}.chat_template_replacements key")
            if not isinstance(new, str) or not new:
                raise ConfigError(
                    f"{name}.chat_template_replacements values must be nonempty strings"
                )
            if old == new:
                raise ConfigError(
                    f"{name}.chat_template_replacements cannot replace text with itself"
                )
        if not isinstance(self.assistant_prefix, str):
            raise ConfigError(f"{name}.assistant_prefix must be a string")
        if self.assistant_prefix and not self.add_generation_prompt:
            raise ConfigError(
                f"{name}.assistant_prefix requires add_generation_prompt=true"
            )
        if self.render_mode == "raw":
            if self.system.mode != "model_default":
                raise ConfigError(
                    f"{name}.system.mode must be model_default when render_mode=raw"
                )
            if self.add_generation_prompt:
                raise ConfigError(
                    f"{name}.add_generation_prompt must be false when render_mode=raw"
                )
            if self.chat_template_kwargs:
                raise ConfigError(
                    f"{name}.chat_template_kwargs must be empty when render_mode=raw"
                )
            if self.chat_template_replacements:
                raise ConfigError(
                    f"{name}.chat_template_replacements must be empty when render_mode=raw"
                )
            if self.assistant_prefix:
                raise ConfigError(
                    f"{name}.assistant_prefix must be empty when render_mode=raw"
                )


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetSelectionSpec:
    num_prompts: int | str
    selection_seed: int

    def validate(self, name: str) -> None:
        if self.num_prompts != "all":
            _positive_int(self.num_prompts, f"{name}.num_prompts")
        if isinstance(self.selection_seed, bool) or not isinstance(
            self.selection_seed, int
        ):
            raise ConfigError(f"{name}.selection_seed must be an integer")


@dataclasses.dataclass(frozen=True, slots=True)
class GenerationSpec:
    temperature: float
    top_p: float
    max_new_tokens: int
    samples_per_prompt: int
    provenance: str | None
    top_k: int = -1
    repetition_penalty: float = 1.0
    stop_token_ids: tuple[int, ...] = ()
    stop_strings: tuple[str, ...] = ()

    def validate(self, name: str) -> None:
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or not 0.0 <= float(self.temperature) <= 2.0
        ):
            raise ConfigError(f"{name}.temperature must be finite and in [0, 2]")
        _probability(self.top_p, f"{name}.top_p", allow_zero=False)
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k == 0
            or self.top_k < -1
        ):
            raise ConfigError(f"{name}.top_k must be -1 or a positive integer")
        if (
            isinstance(self.repetition_penalty, bool)
            or not isinstance(self.repetition_penalty, (int, float))
            or not math.isfinite(float(self.repetition_penalty))
            or float(self.repetition_penalty) <= 0.0
        ):
            raise ConfigError(f"{name}.repetition_penalty must be finite and positive")
        _positive_int(self.max_new_tokens, f"{name}.max_new_tokens")
        _positive_int(self.samples_per_prompt, f"{name}.samples_per_prompt")
        if self.provenance is not None:
            _string(self.provenance, f"{name}.provenance")
        if len(set(self.stop_token_ids)) != len(self.stop_token_ids):
            raise ConfigError(f"{name}.stop_token_ids must not contain duplicates")
        for index, token_id in enumerate(self.stop_token_ids):
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
            ):
                raise ConfigError(
                    f"{name}.stop_token_ids[{index}] must be a nonnegative integer"
                )
        if len(set(self.stop_strings)) != len(self.stop_strings):
            raise ConfigError(f"{name}.stop_strings must not contain duplicates")
        for index, value in enumerate(self.stop_strings):
            _string(value, f"{name}.stop_strings[{index}]")


@dataclasses.dataclass(frozen=True, slots=True)
class SamplingSpec:
    generation: GenerationSpec
    trajectory_filter: str


@dataclasses.dataclass(frozen=True, slots=True)
class PromptInsertionSpec:
    mode: str
    marker: str
    sentinel_id: int
    occurrence: str

    def validate(self, name: str) -> None:
        if self.mode != "placeholder":
            raise ConfigError(f"{name}.mode must be placeholder")
        _string(self.marker, f"{name}.marker")
        if isinstance(self.sentinel_id, bool) or not isinstance(self.sentinel_id, int):
            raise ConfigError(f"{name}.sentinel_id must be an integer")
        if self.sentinel_id >= 0:
            raise ConfigError(
                f"{name}.sentinel_id must be negative and outside the vocabulary"
            )
        if self.occurrence != "exactly_one":
            raise ConfigError(f"{name}.occurrence must be exactly_one")


@dataclasses.dataclass(frozen=True, slots=True)
class ScheduleSpec:
    name: str
    warmup_steps: int
    min_lr_ratio: float

    def validate(self, path: str, total_updates: int | None) -> None:
        if self.name not in {"constant", "cosine"}:
            raise ConfigError(f"{path}.name must be constant or cosine")
        if isinstance(self.warmup_steps, bool) or not isinstance(
            self.warmup_steps, int
        ):
            raise ConfigError(f"{path}.warmup_steps must be an integer")
        if self.warmup_steps < 0:
            raise ConfigError(f"{path}.warmup_steps must be nonnegative")
        if total_updates is not None and self.warmup_steps > total_updates:
            raise ConfigError(f"{path}.warmup_steps must not exceed total updates")
        _probability(self.min_lr_ratio, f"{path}.min_lr_ratio", allow_zero=True)
        if self.name == "constant" and self.min_lr_ratio != 1.0:
            raise ConfigError(f"{path}.min_lr_ratio must be 1 for a constant schedule")


@dataclasses.dataclass(frozen=True, slots=True)
class PromptInitSpec:
    method: str
    seed: int
    std: float | None

    def validate(self, path: str) -> None:
        if self.method not in {"gaussian", "sampled_vocabulary"}:
            raise ConfigError(f"{path}.method must be gaussian or sampled_vocabulary")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ConfigError(f"{path}.seed must be an integer")
        if self.method == "gaussian":
            if (
                isinstance(self.std, bool)
                or not isinstance(self.std, (int, float))
                or not math.isfinite(float(self.std))
                or self.std <= 0
            ):
                raise ConfigError(
                    f"{path}.std must be positive and finite for gaussian init"
                )
        elif self.std is not None:
            raise ConfigError(f"{path}.std is only valid for gaussian init")


@dataclasses.dataclass(frozen=True, slots=True)
class DistillationSpec:
    objective: str
    attention_implementation: str
    activation_checkpointing: bool
    checkpoint_every_updates: int | None
    prompt_length: int
    max_sequence_length: int | None
    overlength_policy: str
    max_trajectories_per_question: int | None
    prompt_init: PromptInitSpec
    insertion: PromptInsertionSpec
    optimizer: str
    learning_rate: float
    schedule: ScheduleSpec
    effective_batch_size: int
    micro_batch_size: int
    cross_entropy_backend: str
    loss_normalization: str
    batch_seed: int
    max_steps: int | None
    epochs: int | None
    weight_decay: float
    max_grad_norm: float | None

    def validate(self) -> None:
        if self.objective != "response_sft":
            raise ConfigError("distillation.objective must be response_sft")
        if self.attention_implementation not in {
            "eager",
            "sdpa",
            "flash_attention_2",
            "flex_attention",
        }:
            raise ConfigError(
                "distillation.attention_implementation must be eager, sdpa, "
                "flash_attention_2, or flex_attention"
            )
        if not isinstance(self.activation_checkpointing, bool):
            raise ConfigError("distillation.activation_checkpointing must be boolean")
        if self.checkpoint_every_updates is not None:
            _positive_int(
                self.checkpoint_every_updates,
                "distillation.checkpoint_every_updates",
            )
        _positive_int(self.prompt_length, "distillation.prompt_length")
        if self.max_sequence_length is not None:
            _positive_int(self.max_sequence_length, "distillation.max_sequence_length")
        if self.overlength_policy not in {"error", "filter"}:
            raise ConfigError("distillation.overlength_policy must be error or filter")
        if self.max_trajectories_per_question is not None:
            _positive_int(
                self.max_trajectories_per_question,
                "distillation.max_trajectories_per_question",
            )
        self.prompt_init.validate("distillation.prompt_init")
        self.insertion.validate("distillation.insertion")
        if self.optimizer != "adamw":
            raise ConfigError("distillation.optimizer must be adamw")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(float(self.learning_rate))
            or self.learning_rate <= 0
        ):
            raise ConfigError("distillation.learning_rate must be positive and finite")
        _positive_int(self.effective_batch_size, "distillation.effective_batch_size")
        _positive_int(self.micro_batch_size, "distillation.micro_batch_size")
        if self.micro_batch_size > self.effective_batch_size:
            raise ConfigError(
                "distillation.micro_batch_size must not exceed effective_batch_size"
            )
        if self.cross_entropy_backend not in {"torch", "liger_fused"}:
            raise ConfigError(
                "distillation.cross_entropy_backend must be torch or liger_fused"
            )
        if self.loss_normalization not in {"token", "sequence"}:
            raise ConfigError(
                "distillation.loss_normalization must be token or sequence"
            )
        _integer(self.batch_seed, "distillation.batch_seed")
        if (self.max_steps is None) == (self.epochs is None):
            raise ConfigError(
                "distillation must declare exactly one of max_steps or epochs"
            )
        if self.max_steps is not None:
            _positive_int(self.max_steps, "distillation.max_steps")
        if self.epochs is not None:
            _positive_int(self.epochs, "distillation.epochs")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ConfigError(
                "distillation.weight_decay must be nonnegative and finite"
            )
        if self.max_grad_norm is not None and (
            isinstance(self.max_grad_norm, bool)
            or not isinstance(self.max_grad_norm, (int, float))
            or not math.isfinite(float(self.max_grad_norm))
            or self.max_grad_norm <= 0
        ):
            raise ConfigError(
                "distillation.max_grad_norm must be null or positive and finite"
            )
        self.schedule.validate("distillation.schedule", self.max_steps)

    def updates_for_trajectory_count(self, trajectory_count: int) -> int:
        _positive_int(trajectory_count, "trajectory_count")
        updates_per_epoch = trajectory_count // self.effective_batch_size
        if updates_per_epoch == 0:
            raise ConfigError(
                "drop_last training requires trajectory_count >= effective_batch_size"
            )
        if self.max_steps is not None:
            return self.max_steps
        assert self.epochs is not None
        return self.epochs * updates_per_epoch


@dataclasses.dataclass(frozen=True, slots=True)
class EvalArmSpec:
    generation: GenerationSpec


@dataclasses.dataclass(frozen=True, slots=True)
class DataSpec:
    sampling: DatasetSelectionSpec
    evaluation: Mapping[str, DatasetSelectionSpec]


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationSpec:
    arms: Mapping[str, EvalArmSpec]
    benchmarks: Mapping[str, GenerationSpec]
    template: TemplateSpec | None


@dataclasses.dataclass(frozen=True, slots=True)
class OffPolicyConfig:
    schema_version: int
    name: str
    task: str
    student: ModelSpec
    teacher: ModelSpec
    template: TemplateSpec
    data: DataSpec
    sampling: SamplingSpec
    distillation: DistillationSpec
    evaluation: EvaluationSpec

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ConfigError("schema_version must be 1")
        _string(self.name, "name")
        _string(self.task, "task")
        for label, model in (("student", self.student), ("teacher", self.teacher)):
            _string(model.checkpoint, f"models.{label}.checkpoint")
            if model.revision is not None:
                _string(model.revision, f"models.{label}.revision")
            if model.subfolder is not None:
                subfolder = _string(model.subfolder, f"models.subfolder.{label}")
                parts = subfolder.split("/")
                if subfolder.startswith("/") or any(
                    part in {"", ".", ".."} for part in parts
                ):
                    raise ConfigError(
                        f"models.subfolder.{label} must be a relative Hub subfolder"
                    )
            _boolean(
                model.trust_remote_code,
                f"models.trust_remote_code.{label}",
            )
            if model.context_length is not None:
                _positive_int(
                    model.context_length,
                    f"models.context_length.{label}",
                )
            forbidden_overrides = {
                "attn_implementation",
                "cache_dir",
                "device_map",
                "dtype",
                "low_cpu_mem_usage",
                "revision",
                "trust_remote_code",
            } & set(model.config_overrides)
            if forbidden_overrides:
                raise ConfigError(
                    f"models.config_overrides.{label} contains loader-owned fields: "
                    f"{', '.join(sorted(forbidden_overrides))}"
                )
            try:
                json.dumps(model.config_overrides, sort_keys=True)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"models.config_overrides.{label} must be JSON-serializable"
                ) from exc
            overridden_length = model.config_overrides.get("max_position_embeddings")
            if overridden_length is not None:
                _positive_int(
                    overridden_length,
                    f"models.config_overrides.{label}.max_position_embeddings",
                )
                if model.context_length != overridden_length:
                    raise ConfigError(
                        f"models.context_length.{label} must equal "
                        "config_overrides.max_position_embeddings"
                    )
        templates = [("template", self.template)]
        if self.evaluation.template is not None:
            self.evaluation.template.validate("evaluation.template")
            if self.evaluation.template.source != self.template.source:
                raise ConfigError(
                    "evaluation.template.source must match template.source"
                )
            templates.append(("evaluation.template", self.evaluation.template))
        self.data.sampling.validate("data.sampling")
        self.sampling.generation.validate("sampling.generation")
        if self.sampling.trajectory_filter not in {"all", "correct_only"}:
            raise ConfigError("sampling.trajectory_filter must be all or correct_only")
        self.distillation.validate()
        marker = self.distillation.insertion.marker
        for template_name, template in templates:
            template_owned_text = "".join(
                part
                for part in (
                    template.system.text,
                    template.user.prefix,
                    template.user.suffix,
                )
                if part is not None
            )
            replacement_output_text = "".join(
                template.chat_template_replacements.values()
            )
            if (
                template_owned_text.count(marker)
                + replacement_output_text.count(marker)
                != 1
            ):
                raise ConfigError(
                    f"soft-prompt marker must appear exactly once in {template_name}-owned "
                    "text or chat-template replacement output"
                )
        if not self.data.evaluation:
            raise ConfigError("data.evaluation must not be empty")
        for label, selection in self.data.evaluation.items():
            _string(label, "evaluation dataset name")
            selection.validate(f"data.evaluation.{label}")
        unknown_benchmarks = set(self.evaluation.benchmarks) - set(self.data.evaluation)
        if unknown_benchmarks:
            raise ConfigError(
                "evaluation.benchmarks contains datasets absent from "
                f"data.evaluation: {sorted(unknown_benchmarks)}"
            )
        for label, generation in self.evaluation.benchmarks.items():
            generation.validate(f"evaluation.benchmarks.{label}.generation")
        required = {"student", "teacher", "student_soft_prompt"}
        if set(self.evaluation.arms) != required:
            raise ConfigError(
                "evaluation.arms must be exactly student, teacher, student_soft_prompt"
            )
        for arm, spec in self.evaluation.arms.items():
            spec.generation.validate(f"evaluation.arms.{arm}.generation")
        if (
            self.evaluation.arms["student"].generation
            != self.evaluation.arms["student_soft_prompt"].generation
        ):
            raise ConfigError(
                "student and student_soft_prompt must use the same generation recipe"
            )

    @property
    def sampled_trajectory_count(self) -> int | str:
        if self.data.sampling.num_prompts == "all":
            return "all"
        return (
            self.data.sampling.num_prompts * self.sampling.generation.samples_per_prompt
        )


def _model(
    raw: Any,
    name: str,
    *,
    revision: str | None = None,
    subfolder: str | None = None,
    trust_remote_code: bool = False,
    context_length: int | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> ModelSpec:
    return ModelSpec(
        checkpoint=_string(raw, name),
        revision=revision,
        subfolder=subfolder,
        trust_remote_code=trust_remote_code,
        context_length=context_length,
        config_overrides=dict(config_overrides or {}),
    )


def _system(raw: Any, name: str) -> SystemPromptSpec:
    body = _mapping(raw, name)
    return SystemPromptSpec(
        mode=str(body.get("mode", "")),
        text=_optional_string(body.get("text"), f"{name}.text"),
    )


def _template(raw: Any, name: str) -> TemplateSpec:
    body = _mapping(raw, name)
    if "chat_template_sha256" in body:
        raise ConfigError(
            f"{name}.chat_template_sha256 is runtime evidence and must not be configured"
        )
    user = _mapping(body.get("user"), f"{name}.user")
    chat_template_kwargs_raw = body.get("chat_template_kwargs", {})
    chat_template_kwargs = _mapping(
        chat_template_kwargs_raw, f"{name}.chat_template_kwargs"
    )
    chat_template_replacements = _mapping(
        body.get("chat_template_replacements", {}),
        f"{name}.chat_template_replacements",
    )
    return TemplateSpec(
        source=str(body.get("source", "")),
        render_mode=str(body.get("render_mode", "chat")),
        system=_system(body.get("system"), f"{name}.system"),
        user=UserPromptSpec(
            field=str(user.get("field", "")),
            prefix=_plain_string(user.get("prefix", ""), f"{name}.user.prefix"),
            suffix=_plain_string(user.get("suffix", ""), f"{name}.user.suffix"),
        ),
        add_generation_prompt=_boolean(
            body.get("add_generation_prompt", True),
            f"{name}.add_generation_prompt",
        ),
        chat_template_kwargs=chat_template_kwargs,
        chat_template_replacements=chat_template_replacements,
        assistant_prefix=_plain_string(
            body.get("assistant_prefix", ""), f"{name}.assistant_prefix"
        ),
    )


def _selection(raw: Any, name: str) -> DatasetSelectionSpec:
    body = _mapping(raw, name)
    unexpected = set(body) - {"num_prompts", "selection_seed"}
    if unexpected:
        fields = ", ".join(sorted(unexpected))
        raise ConfigError(f"{name} contains task-owned or unknown fields: {fields}")
    return DatasetSelectionSpec(
        num_prompts=_prompt_count(body.get("num_prompts"), f"{name}.num_prompts"),
        selection_seed=_integer(
            body.get("selection_seed", DEFAULT_SELECTION_SEED),
            f"{name}.selection_seed",
        ),
    )


def _generation(
    raw: Any,
    name: str,
    *,
    default_temperature: float | None = None,
    default_top_p: float | None = None,
    default_max_new_tokens: int | None = None,
    default_samples_per_prompt: int | None = None,
) -> GenerationSpec:
    body = _mapping(raw, name)
    allowed = {
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_new_tokens",
        "samples_per_prompt",
        "provenance",
        "stop_token_ids",
        "stop_strings",
    }
    unexpected = set(body) - allowed
    if unexpected:
        raise ConfigError(
            f"{name} contains unknown fields: {', '.join(sorted(unexpected))}"
        )
    stop_token_ids = _list(body.get("stop_token_ids", []), f"{name}.stop_token_ids")
    stop_strings = _list(body.get("stop_strings", []), f"{name}.stop_strings")
    return GenerationSpec(
        temperature=_number(
            body.get("temperature", default_temperature), f"{name}.temperature"
        ),
        top_p=_number(body.get("top_p", default_top_p), f"{name}.top_p"),
        top_k=_integer(body.get("top_k", -1), f"{name}.top_k"),
        repetition_penalty=_number(
            body.get("repetition_penalty", 1.0), f"{name}.repetition_penalty"
        ),
        max_new_tokens=_integer(
            body.get("max_new_tokens", default_max_new_tokens),
            f"{name}.max_new_tokens",
        ),
        samples_per_prompt=_integer(
            body.get("samples_per_prompt", default_samples_per_prompt),
            f"{name}.samples_per_prompt",
        ),
        provenance=_optional_string(body.get("provenance"), f"{name}.provenance"),
        stop_token_ids=tuple(stop_token_ids),
        stop_strings=tuple(stop_strings),
    )


def parse_config(raw: Any) -> OffPolicyConfig:
    root = _mapping(raw, "config")
    models = _mapping(root.get("models"), "models")
    unexpected_models = set(models) - {
        "student",
        "teacher",
        "revision",
        "subfolder",
        "trust_remote_code",
        "context_length",
        "config_overrides",
    }
    if unexpected_models:
        fields = ", ".join(sorted(unexpected_models))
        raise ConfigError(f"models contains unknown fields: {fields}")
    revision_raw = _mapping(models.get("revision", {}), "models.revision")
    unexpected_revision_roles = set(revision_raw) - {"student", "teacher"}
    if unexpected_revision_roles:
        fields = ", ".join(sorted(unexpected_revision_roles))
        raise ConfigError(f"models.revision contains unknown fields: {fields}")
    revision = {
        role: _optional_string(revision_raw.get(role), f"models.revision.{role}")
        for role in ("student", "teacher")
    }
    subfolder_raw = _mapping(models.get("subfolder", {}), "models.subfolder")
    unexpected_subfolder_roles = set(subfolder_raw) - {"student", "teacher"}
    if unexpected_subfolder_roles:
        fields = ", ".join(sorted(unexpected_subfolder_roles))
        raise ConfigError(f"models.subfolder contains unknown fields: {fields}")
    subfolder = {
        role: _optional_string(subfolder_raw.get(role), f"models.subfolder.{role}")
        for role in ("student", "teacher")
    }
    trust_remote_code_raw = _mapping(
        models.get("trust_remote_code", {}), "models.trust_remote_code"
    )
    unexpected_trust_roles = set(trust_remote_code_raw) - {"student", "teacher"}
    if unexpected_trust_roles:
        fields = ", ".join(sorted(unexpected_trust_roles))
        raise ConfigError(f"models.trust_remote_code contains unknown fields: {fields}")
    trust_remote_code = {
        role: _boolean(
            trust_remote_code_raw.get(role, False),
            f"models.trust_remote_code.{role}",
        )
        for role in ("student", "teacher")
    }
    context_length_raw = _mapping(
        models.get("context_length", {}), "models.context_length"
    )
    unexpected_context_roles = set(context_length_raw) - {"student", "teacher"}
    if unexpected_context_roles:
        fields = ", ".join(sorted(unexpected_context_roles))
        raise ConfigError(f"models.context_length contains unknown fields: {fields}")
    context_length = {
        role: _optional_integer(
            context_length_raw.get(role), f"models.context_length.{role}"
        )
        for role in ("student", "teacher")
    }
    config_overrides_raw = _mapping(
        models.get("config_overrides", {}), "models.config_overrides"
    )
    unexpected_override_roles = set(config_overrides_raw) - {"student", "teacher"}
    if unexpected_override_roles:
        fields = ", ".join(sorted(unexpected_override_roles))
        raise ConfigError(f"models.config_overrides contains unknown fields: {fields}")
    config_overrides = {
        role: _mapping(
            config_overrides_raw.get(role, {}),
            f"models.config_overrides.{role}",
        )
        for role in ("student", "teacher")
    }
    if "templates" in root:
        raise ConfigError(
            "legacy multiple templates are not supported; use one template"
        )
    template = _template(root.get("template"), "template")
    if "sampling_dataset" in root:
        raise ConfigError("sampling_dataset moved to task adapter; use data.sampling")
    data_raw = _mapping(root.get("data"), "data")
    data_evaluation_raw = _mapping(data_raw.get("evaluation"), "data.evaluation")
    sampling_raw = _mapping(root.get("sampling"), "sampling")
    distill_raw = _mapping(root.get("distillation"), "distillation")
    prompt_init_raw = _mapping(
        distill_raw.get("prompt_init"), "distillation.prompt_init"
    )
    insertion_raw = _mapping(distill_raw.get("insertion"), "distillation.insertion")
    schedule_raw = _mapping(distill_raw.get("schedule"), "distillation.schedule")
    evaluation_raw = _mapping(root.get("evaluation"), "evaluation")
    if "datasets" in evaluation_raw:
        raise ConfigError(
            "evaluation.datasets moved to task adapter; use data.evaluation"
        )
    unexpected_evaluation = set(evaluation_raw) - {"arms", "benchmarks", "template"}
    if unexpected_evaluation:
        fields = ", ".join(sorted(unexpected_evaluation))
        raise ConfigError(f"evaluation contains unknown fields: {fields}")
    arms_raw = _mapping(evaluation_raw.get("arms"), "evaluation.arms")
    benchmarks_raw = _mapping(
        evaluation_raw.get("benchmarks", {}), "evaluation.benchmarks"
    )

    sampling = SamplingSpec(
        generation=_generation(
            sampling_raw.get("generation"),
            "sampling.generation",
            default_temperature=DEFAULT_SAMPLING_TEMPERATURE,
            default_top_p=DEFAULT_SAMPLING_TOP_P,
            default_max_new_tokens=DEFAULT_SAMPLING_MAX_NEW_TOKENS,
            default_samples_per_prompt=DEFAULT_SAMPLING_SAMPLES_PER_PROMPT,
        ),
        trajectory_filter=str(
            sampling_raw.get("trajectory_filter", DEFAULT_SAMPLING_TRAJECTORY_FILTER)
        ),
    )
    if "template" in sampling_raw:
        raise ConfigError(
            "sampling.template is redundant; all stages use the shared template"
        )
    distillation = DistillationSpec(
        objective=str(distill_raw.get("objective", "")),
        attention_implementation=str(
            distill_raw.get(
                "attention_implementation",
                DEFAULT_TRAIN_ATTENTION_IMPLEMENTATION,
            )
        ),
        activation_checkpointing=_boolean(
            distill_raw.get(
                "activation_checkpointing",
                DEFAULT_ACTIVATION_CHECKPOINTING,
            ),
            "distillation.activation_checkpointing",
        ),
        checkpoint_every_updates=_optional_integer(
            distill_raw.get(
                "checkpoint_every_updates",
                DEFAULT_CHECKPOINT_EVERY_UPDATES,
            ),
            "distillation.checkpoint_every_updates",
        ),
        prompt_length=_integer(
            distill_raw.get("prompt_length"), "distillation.prompt_length"
        ),
        max_sequence_length=_optional_integer(
            distill_raw.get("max_sequence_length"),
            "distillation.max_sequence_length",
        ),
        overlength_policy=str(distill_raw.get("overlength_policy", "error")),
        max_trajectories_per_question=_optional_integer(
            distill_raw.get("max_trajectories_per_question"),
            "distillation.max_trajectories_per_question",
        ),
        prompt_init=PromptInitSpec(
            method=str(prompt_init_raw.get("method", "")),
            seed=_integer(
                prompt_init_raw.get("seed", DEFAULT_PROMPT_INIT_SEED),
                "distillation.prompt_init.seed",
            ),
            std=_optional_number(
                prompt_init_raw.get("std"), "distillation.prompt_init.std"
            ),
        ),
        insertion=PromptInsertionSpec(
            mode=str(insertion_raw.get("mode", "")),
            marker=str(insertion_raw.get("marker", "")),
            sentinel_id=_integer(
                insertion_raw.get("sentinel_id"),
                "distillation.insertion.sentinel_id",
            ),
            occurrence=str(insertion_raw.get("occurrence", "")),
        ),
        optimizer=str(distill_raw.get("optimizer", "")),
        learning_rate=_number(
            distill_raw.get("learning_rate"), "distillation.learning_rate"
        ),
        schedule=ScheduleSpec(
            name=str(schedule_raw.get("name", "")),
            warmup_steps=_integer(
                schedule_raw.get("warmup_steps"),
                "distillation.schedule.warmup_steps",
            ),
            min_lr_ratio=_number(
                schedule_raw.get("min_lr_ratio"),
                "distillation.schedule.min_lr_ratio",
            ),
        ),
        effective_batch_size=_integer(
            distill_raw.get("effective_batch_size"),
            "distillation.effective_batch_size",
        ),
        micro_batch_size=_integer(
            distill_raw.get("micro_batch_size", DEFAULT_DISTILLATION_MICRO_BATCH_SIZE),
            "distillation.micro_batch_size",
        ),
        cross_entropy_backend=str(
            distill_raw.get("cross_entropy_backend", DEFAULT_CROSS_ENTROPY_BACKEND)
        ),
        loss_normalization=str(
            distill_raw.get("loss_normalization", DEFAULT_LOSS_NORMALIZATION)
        ),
        batch_seed=_integer(
            distill_raw.get("batch_seed", DEFAULT_DISTILLATION_BATCH_SEED),
            "distillation.batch_seed",
        ),
        max_steps=_optional_integer(
            distill_raw.get("max_steps"), "distillation.max_steps"
        ),
        epochs=_optional_integer(distill_raw.get("epochs"), "distillation.epochs"),
        weight_decay=_number(
            distill_raw.get("weight_decay"), "distillation.weight_decay"
        ),
        max_grad_norm=_optional_number(
            distill_raw.get("max_grad_norm"), "distillation.max_grad_norm"
        ),
    )
    if "template" in distill_raw:
        raise ConfigError(
            "distillation.template is redundant; all stages use the shared template"
        )
    if "trajectory_filter" in distill_raw:
        raise ConfigError(
            "distillation.trajectory_filter is redundant; training consumes the sampled snapshot"
        )
    arms = {
        str(name): EvalArmSpec(
            generation=_generation(
                _mapping(body, f"evaluation.arms.{name}").get("generation"),
                f"evaluation.arms.{name}.generation",
            ),
        )
        for name, body in arms_raw.items()
    }
    benchmarks = {
        str(label): _generation(
            _mapping(body, f"evaluation.benchmarks.{label}").get("generation"),
            f"evaluation.benchmarks.{label}.generation",
        )
        for label, body in benchmarks_raw.items()
    }
    for label, body in benchmarks_raw.items():
        unexpected = set(_mapping(body, f"evaluation.benchmarks.{label}")) - {
            "generation"
        }
        if unexpected:
            raise ConfigError(
                f"evaluation.benchmarks.{label} contains unknown fields: "
                f"{', '.join(sorted(unexpected))}"
            )
    for name, body in arms_raw.items():
        if "template" in _mapping(body, f"evaluation.arms.{name}"):
            raise ConfigError(
                f"evaluation.arms.{name}.template is redundant; all arms use the shared template"
            )
    config = OffPolicyConfig(
        schema_version=_integer(root.get("schema_version"), "schema_version"),
        name=str(root.get("name", "")),
        task=str(root.get("task", "")),
        student=_model(
            models.get("student"),
            "models.student",
            revision=revision["student"],
            subfolder=subfolder["student"],
            trust_remote_code=trust_remote_code["student"],
            context_length=context_length["student"],
            config_overrides=config_overrides["student"],
        ),
        teacher=_model(
            models.get("teacher"),
            "models.teacher",
            revision=revision["teacher"],
            subfolder=subfolder["teacher"],
            trust_remote_code=trust_remote_code["teacher"],
            context_length=context_length["teacher"],
            config_overrides=config_overrides["teacher"],
        ),
        template=template,
        data=DataSpec(
            sampling=_selection(data_raw.get("sampling"), "data.sampling"),
            evaluation={
                str(label): _selection(body, f"data.evaluation.{label}")
                for label, body in data_evaluation_raw.items()
            },
        ),
        sampling=sampling,
        distillation=distillation,
        evaluation=EvaluationSpec(
            arms=arms,
            benchmarks=benchmarks,
            template=(
                _template(evaluation_raw["template"], "evaluation.template")
                if "template" in evaluation_raw
                else None
            ),
        ),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> OffPolicyConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    try:
        return parse_config(raw)
    except ConfigError as exc:
        raise ConfigError(f"invalid config {path}: {exc}") from exc


__all__ = [
    "ConfigError",
    "OffPolicyConfig",
    "load_config",
    "parse_config",
]
