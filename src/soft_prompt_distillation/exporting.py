"""Export a learned soft prompt as input-only virtual token rows."""

from __future__ import annotations

import gc
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from .common import (
    RunOptions,
    load_template_tokenizer,
    load_tokenizer,
    merge_model_identities,
    model_source,
    require_matching_tokenizer_vocabulary,
    sha256_file,
    write_manifest,
)
from .config import load_config
from .virtual_tokens import install_virtual_prompt, write_virtual_prompt_metadata


def _repair_generation_config(model: Any, checkpoint: str) -> None:
    """Make a published generation_config self-consistent enough to save.

    save_pretrained validates generation_config and refuses to write one that
    sets temperature or top_p without do_sample. allenai/Olmo-3-7B-Instruct-DPO
    ships exactly that combination, which aborted the export of every soft-prompt
    checkpoint after the run had already spent eight hours sampling, training and
    evaluating both baselines.

    The stored defaults are close to inert for us -- every evaluation passes its
    sampling parameters explicitly -- so the repair keeps the published values and
    only sets the flag they imply, rather than discarding the model's documented
    decoding recipe.
    """

    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    samples = (
        getattr(generation_config, "temperature", None) is not None
        or getattr(generation_config, "top_p", None) is not None
    )
    if samples and not getattr(generation_config, "do_sample", False):
        generation_config.do_sample = True
        print(
            f"export: {checkpoint} sets temperature/top_p without do_sample; "
            "setting do_sample=True so the derived checkpoint can be saved",
            flush=True,
        )


def _restore_legacy_rope_keys(output: Path, base_config: Any | None = None) -> None:
    """Write rope settings in the form SGLang reads from a local directory.

    transformers 5.3 saves the modern `rope_parameters` dict and writes null for
    the legacy `rope_theta` and `rope_scaling`. SGLang rebuilds rope_parameters
    from those legacy keys when the model is a local directory, so it sees None
    and Olmo3 dies with `'NoneType' object is not subscriptable`. Loading the same
    model by hub id takes a different path and works, which is why the student and
    teacher arms pass and only the derived checkpoints fail -- an export is always
    a local directory.

    Verified against SGLang in the runtime container: with the legacy keys
    restored, ModelConfig reports the full yarn parameters again.

    SGLang's OLMo-3 compatibility config still consumes the legacy keys.  It
    also subclasses the Transformers 5 config base in a way that validates a
    modern ``rope_parameters`` payload before the subclass has installed
    ``max_position_embeddings``.  Once the equivalent legacy fields have been
    restored, remove that redundant modern field for OLMo-3 exports so SGLang
    takes its working compatibility path.  Transformers reconstructs the
    modern representation from these legacy fields when it loads the export.

    Transformers 5 may also omit ``max_position_embeddings`` from a derived
    OLMo-3 config even though it is present in the Hub config, so copy it from
    the immutable base config when the saved export lost it.

    Only missing keys are filled in, so a config that already carries them, or one
    with no rope settings at all, is left exactly as transformers wrote it.
    """

    config_path = output / "config.json"
    if not config_path.is_file():
        return
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base_payload = base_config.to_dict() if base_config is not None else {}
    if payload.get("max_position_embeddings") is None:
        max_positions = base_payload.get("max_position_embeddings")
        if max_positions is not None:
            payload["max_position_embeddings"] = max_positions
    parameters = payload.get("rope_parameters")
    if not isinstance(parameters, dict) or "rope_theta" not in parameters:
        return
    if payload.get("rope_theta") is None:
        payload["rope_theta"] = parameters["rope_theta"]
    scaling = {
        key: value for key, value in parameters.items() if key != "rope_theta"
    }
    if payload.get("rope_scaling") is None and len(scaling) > 1:
        payload["rope_scaling"] = scaling
    if payload.get("model_type") == "olmo3":
        payload.pop("rope_parameters", None)
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _complete_rope_overrides(
    config_overrides: Mapping[str, Any], base_config: Any
) -> dict[str, Any]:
    """Carry the base theta into a partial YaRN override.

    Transformers 5.3 normalizes Qwen3 RoPE settings into ``rope_parameters``.
    When a caller supplies a replacement ``rope_scaling`` dict without
    ``rope_theta``, that normalization discards the theta published by the base
    checkpoint and inserts ``None`` instead.  The model then fails in its
    constructor before the exporter can save or repair the derived config.

    Configs intentionally specify only the scaling delta, so fill the missing
    theta from the unmodified base config.  Explicit caller values always win.
    """

    completed = deepcopy(dict(config_overrides))
    scaling = completed.get("rope_scaling")
    if not isinstance(scaling, dict) or scaling.get("rope_theta") is not None:
        return completed
    payload = base_config.to_dict()
    parameters = payload.get("rope_parameters")
    theta = parameters.get("rope_theta") if isinstance(parameters, dict) else None
    if theta is None:
        theta = payload.get("rope_theta")
    if theta is not None:
        scaling["rope_theta"] = theta
    return completed


def export_soft_prompt_model(options: RunOptions, checkpoint: Path) -> Path:
    options.output.mkdir(parents=True, exist_ok=False)
    options.cache_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(options.config)
    student_source, student_revision = model_source(options, config, "student")
    tokenizer, runtime_identity = load_template_tokenizer(options, config)
    _student_tokenizer, student_identity = load_tokenizer(options, config, "student")
    require_matching_tokenizer_vocabulary(runtime_identity, student_identity)
    merge_model_identities(runtime_identity, student_identity)
    base_model_config = AutoConfig.from_pretrained(
        student_source,
        revision=student_revision,
        trust_remote_code=config.student.trust_remote_code,
    )
    config_overrides = _complete_rope_overrides(
        config.student.config_overrides, base_model_config
    )
    model_config = AutoConfig.from_pretrained(
        student_source,
        revision=student_revision,
        trust_remote_code=config.student.trust_remote_code,
        **config_overrides,
    )
    model = AutoModelForCausalLM.from_pretrained(
        student_source,
        revision=student_revision,
        cache_dir=options.cache_dir,
        config=model_config,
        dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=config.student.trust_remote_code,
    ).eval()
    try:
        tensors = load_file(checkpoint)
        if "prompt_embeddings" not in tensors:
            raise RuntimeError("soft-prompt checkpoint lacks prompt_embeddings")
        spec = install_virtual_prompt(model, tokenizer, tensors["prompt_embeddings"])
        # For Hub repositories that publish stage checkpoints in subfolders,
        # the resolved local subfolder is the actual source of generation
        # metadata.  Passing the repository root loses that distinction.
        _repair_generation_config(model, student_source)
        model.save_pretrained(options.output, safe_serialization=True)
        _restore_legacy_rope_keys(options.output, base_model_config)
        tokenizer.save_pretrained(options.output)
        metadata = write_virtual_prompt_metadata(
            options.output,
            spec,
            base_model=config.student.checkpoint,
            soft_prompt_sha256=sha256_file(checkpoint),
        )
        runtime_identity["input_checkpoint"] = {
            "path": str(checkpoint),
            "content_sha256": sha256_file(checkpoint),
        }
        runtime_identity["base_student"] = config.student.checkpoint
        runtime_identity["virtual_token_ids"] = list(spec.token_ids)
        write_manifest(
            options,
            config,
            "export",
            {"virtual_tokens": len(spec.token_ids)},
            [metadata, options.output / "config.json"],
            runtime_identity,
            stage_source=Path(__file__),
        )
    finally:
        del model
        gc.collect()
    return options.output


__all__ = ["export_soft_prompt_model"]
