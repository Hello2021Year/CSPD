"""Response-only soft-prompt distillation stage."""

from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, TextIO

import torch
from safetensors.torch import save_file

from .common import (
    RunOptions,
    load_model,
    load_template_tokenizer,
    load_tokenizer,
    merge_model_identities,
    require_matching_tokenizer_vocabulary,
    model_source,
    iter_dataset_rows,
    sha256_file,
    snapshot_template_identity,
    write_manifest,
)
from .config import load_config
from .objective import make_response_sft_example
from .progress import log_progress
from .rendering import (
    RenderedPrompt,
    render_for_execution,
    resolve_insertion_index,
)
from .schedule import learning_rate_for_update
from .training import (
    SoftPromptModel,
    accumulate_response_loss,
    batch_indices_for_update,
    collate_examples,
    finish_prompt_update,
    initialize_prompt_embeddings,
    response_token_count,
)


def scheduled_checkpoint_updates(
    total_updates: int, checkpoint_every_updates: int | None
) -> tuple[int, ...]:
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    updates = {total_updates}
    if checkpoint_every_updates is not None:
        if checkpoint_every_updates <= 0:
            raise ValueError("checkpoint_every_updates must be positive or null")
        updates.update(
            range(checkpoint_every_updates, total_updates + 1, checkpoint_every_updates)
        )
    return tuple(sorted(updates))


def validate_snapshot_prompt_reuse(
    snapshot_rendered: RenderedPrompt,
    current_rendered: RenderedPrompt,
    snapshot_prompt_ids: tuple[int, ...],
) -> None:
    """Allow position changes while requiring the same marker-free prompt text."""

    if snapshot_prompt_ids != snapshot_rendered.token_ids:
        raise RuntimeError(
            "snapshot prompt IDs differ from its recorded rendering config"
        )
    if snapshot_rendered.text_sha256 != current_rendered.text_sha256:
        raise RuntimeError(
            "current marker-free prompt text differs from teacher snapshot"
        )


def record_training_metric(
    handle: TextIO,
    metric: dict[str, int | float],
    *,
    total_updates: int,
    checkpoint: bool,
) -> None:
    """Persist one update immediately and report saved-checkpoint progress."""

    handle.write(json.dumps(metric, ensure_ascii=False) + "\n")
    handle.flush()
    if checkpoint:
        log_progress(
            "stage=train event=checkpoint "
            f"update={metric['update']}/{total_updates} "
            f"loss={metric['loss']:.8g} "
            f"gradient_norm={metric['gradient_norm']:.8g} "
            f"learning_rate={metric['learning_rate']:.8g}"
        )


def training_sample_payload(
    trajectory: Mapping[str, Any],
    rendered: RenderedPrompt,
    *,
    trajectory_index: int,
    insert_at: int,
    soft_prompt_length: int,
) -> dict[str, Any]:
    """Build one auditable example of the exact training token contract."""

    if rendered.control_token_ids is None:
        raise RuntimeError("training sample rendering lacks control token IDs")
    response_ids = [int(token_id) for token_id in trajectory["response_ids"]]
    prompt_ids = [int(token_id) for token_id in rendered.token_ids]
    payload = {
        "selection": "first_global_batch_update_1",
        "trajectory_index": trajectory_index,
        "question_id": trajectory.get("question_id"),
        "source_row_index": trajectory.get("source_row_index"),
        "sample_index": trajectory.get("sample_index"),
        "messages": [dict(message) for message in rendered.messages],
        "rendered_prompt_text": rendered.text,
        "response_text": trajectory.get("completion"),
        "tokenization_strategy": rendered.tokenization_strategy,
        "prompt_token_ids": prompt_ids,
        "control_prompt_token_ids": [
            int(token_id) for token_id in rendered.control_token_ids
        ],
        "response_token_ids": response_ids,
        "soft_prompt_insert_at": insert_at,
        "soft_prompt_length": soft_prompt_length,
        "prompt_token_count": len(prompt_ids),
        "response_token_count": len(response_ids),
        "training_sequence_length_with_soft_prompt": (
            len(prompt_ids) + soft_prompt_length + len(response_ids)
        ),
        "prompt_token_ids_sha256": rendered.token_ids_sha256,
        "tokenization_matches_canonical": rendered.tokenization_matches_canonical,
        "canonical_prompt_token_count": rendered.canonical_token_count,
        "canonical_prompt_token_ids_sha256": (
            rendered.canonical_token_ids_sha256
        ),
    }
    return payload


def log_training_sample(payload: Mapping[str, Any]) -> None:
    """Emit one physical, immediately flushed JSON log line."""

    encoded = json.dumps(
        dict(payload), ensure_ascii=True, separators=(",", ":")
    )
    log_progress(f"stage=train event=sample payload={encoded}")


def filter_trajectories_by_sequence_length(
    trajectories: list[dict[str, object]],
    *,
    prompt_length: int,
    max_sequence_length: int | None,
    overlength_policy: str,
) -> tuple[list[dict[str, object]], dict[str, int | str | None]]:
    """Apply the configured limit to text tokens plus inserted soft tokens."""

    lengths = [
        len(trajectory["prompt_ids"])
        + len(trajectory["response_ids"])
        + prompt_length
        for trajectory in trajectories
    ]
    if not lengths:
        raise RuntimeError("snapshot contains no trajectories")
    if max_sequence_length is None:
        kept = trajectories
    else:
        overlength_indices = [
            index for index, length in enumerate(lengths)
            if length > max_sequence_length
        ]
        if overlength_indices and overlength_policy == "error":
            index = overlength_indices[0]
            raise RuntimeError(
                f"trajectory {index} sequence length {lengths[index]} exceeds "
                f"distillation.max_sequence_length={max_sequence_length}"
            )
        kept = [
            trajectory
            for trajectory, length in zip(trajectories, lengths, strict=True)
            if length <= max_sequence_length
        ]
    filtered = len(trajectories) - len(kept)
    if not kept:
        raise RuntimeError(
            "sequence-length filtering removed every trajectory: "
            f"input={len(trajectories)}, limit={max_sequence_length}"
        )
    return kept, {
        "input_trajectories": len(trajectories),
        "kept_trajectories": len(kept),
        "filtered_trajectories": filtered,
        "max_observed_sequence_length": max(lengths),
        "max_kept_sequence_length": max(
            len(trajectory["prompt_ids"])
            + len(trajectory["response_ids"])
            + prompt_length
            for trajectory in kept
        ),
        "max_sequence_length": max_sequence_length,
        "overlength_policy": overlength_policy,
    }


def cap_trajectories_per_question(
    trajectories: list[dict[str, object]],
    *,
    max_trajectories_per_question: int | None,
) -> tuple[list[dict[str, object]], dict[str, int | None]]:
    """Keep at most the first configured number of trajectories per question."""

    seen: dict[str, int] = {}
    kept = []
    for index, trajectory in enumerate(trajectories):
        question_id = trajectory.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise RuntimeError(
                f"trajectory {index} lacks a non-empty question_id required for capping"
            )
        count = seen.get(question_id, 0)
        seen[question_id] = count + 1
        if (
            max_trajectories_per_question is None
            or count < max_trajectories_per_question
        ):
            kept.append(trajectory)
    if not kept:
        raise RuntimeError("per-question trajectory cap removed every trajectory")
    return kept, {
        "input_trajectories": len(trajectories),
        "kept_trajectories": len(kept),
        "capped_trajectories": len(trajectories) - len(kept),
        "question_count": len(seen),
        "max_observed_trajectories_per_question": max(seen.values()),
        "max_trajectories_per_question": max_trajectories_per_question,
    }


def load_training_trajectories(
    snapshot: Path,
    *,
    prompt_length: int,
    max_sequence_length: int | None,
    overlength_policy: str,
    max_trajectories_per_question: int | None,
) -> tuple[
    list[dict[str, object]],
    dict[str, int | str | None],
    dict[str, int | None],
]:
    """Stream a snapshot while preserving filter-then-cap semantics.

    Teacher snapshots can be many gigabytes.  Reading the whole JSONL text and
    only then filtering it multiplies that text, its split lines, and every
    decoded token list in each distributed rank.  This loader retains only
    rows that survive both configured filters while producing the same order
    and manifest statistics as the two list-based helpers above.
    """

    trajectories: list[dict[str, object]] = []
    seen: dict[str, int] = {}
    input_count = 0
    sequence_kept_count = 0
    filtered_count = 0
    capped_count = 0
    max_observed_length: int | None = None
    max_sequence_kept_length: int | None = None

    for input_index, trajectory in enumerate(iter_dataset_rows(snapshot)):
        input_count += 1
        length = (
            len(trajectory["prompt_ids"])
            + len(trajectory["response_ids"])
            + prompt_length
        )
        max_observed_length = (
            length
            if max_observed_length is None
            else max(max_observed_length, length)
        )
        if max_sequence_length is not None and length > max_sequence_length:
            if overlength_policy == "error":
                raise RuntimeError(
                    f"trajectory {input_index} sequence length {length} exceeds "
                    f"distillation.max_sequence_length={max_sequence_length}"
                )
            filtered_count += 1
            continue

        sequence_index = sequence_kept_count
        sequence_kept_count += 1
        max_sequence_kept_length = (
            length
            if max_sequence_kept_length is None
            else max(max_sequence_kept_length, length)
        )
        question_id = trajectory.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise RuntimeError(
                f"trajectory {sequence_index} lacks a non-empty question_id "
                "required for capping"
            )
        count = seen.get(question_id, 0)
        seen[question_id] = count + 1
        if (
            max_trajectories_per_question is not None
            and count >= max_trajectories_per_question
        ):
            capped_count += 1
            continue
        trajectories.append(trajectory)

    if input_count == 0:
        raise RuntimeError("snapshot contains no trajectories")
    if sequence_kept_count == 0:
        raise RuntimeError(
            "sequence-length filtering removed every trajectory: "
            f"input={input_count}, limit={max_sequence_length}"
        )
    if not trajectories:
        raise RuntimeError("per-question trajectory cap removed every trajectory")
    assert max_observed_length is not None
    assert max_sequence_kept_length is not None
    return trajectories, {
        "input_trajectories": input_count,
        "kept_trajectories": sequence_kept_count,
        "filtered_trajectories": filtered_count,
        "max_observed_sequence_length": max_observed_length,
        "max_kept_sequence_length": max_sequence_kept_length,
        "max_sequence_length": max_sequence_length,
        "overlength_policy": overlength_policy,
    }, {
        "input_trajectories": sequence_kept_count,
        "kept_trajectories": len(trajectories),
        "capped_trajectories": capped_count,
        "question_count": len(seen),
        "max_observed_trajectories_per_question": max(seen.values()),
        "max_trajectories_per_question": max_trajectories_per_question,
    }


TRAINING_STATE_FILENAME = "training_state.pt"


def _save_training_state(
    output: Path, update: int, optimizer: object, metrics_count: int
) -> None:
    """Persist what a restart cannot recompute: optimizer moments and position.

    Batch order is a pure function of (batch_seed, update), so no sampler state
    is needed. Written after the checkpoint it names, so it can never point at a
    checkpoint that does not exist.
    """

    import torch

    path = output / TRAINING_STATE_FILENAME
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "update": int(update),
            "optimizer": optimizer.state_dict(),
            "metrics_count": int(metrics_count),
        },
        temporary,
    )
    temporary.replace(path)


def _load_training_state(output: Path) -> dict | None:
    import torch

    path = output / TRAINING_STATE_FILENAME
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_resume_step(
    state: Mapping[str, Any], metrics_path: Path, total_updates: int
) -> tuple[int, int]:
    """Verify that saved optimizer position and durable metrics agree."""

    completed = state.get("update")
    metrics_count = state.get("metrics_count")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise RuntimeError("training state update must be an integer")
    if completed <= 0 or completed > total_updates:
        raise RuntimeError(
            f"training state update {completed} is outside 1..{total_updates}"
        )
    if isinstance(metrics_count, bool) or not isinstance(metrics_count, int):
        raise RuntimeError("training state metrics_count must be an integer")
    if metrics_count != completed:
        raise RuntimeError(
            f"training state update {completed} disagrees with metrics_count "
            f"{metrics_count}"
        )
    if not metrics_path.is_file():
        raise RuntimeError(
            f"training state records update {completed} but metrics are missing: "
            f"{metrics_path}"
        )

    rows = [
        line
        for line in metrics_path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    if len(rows) < metrics_count:
        raise RuntimeError(
            f"training state records {metrics_count} metrics but only "
            f"{len(rows)} rows exist"
        )
    for expected_update, line in enumerate(rows[:metrics_count], start=1):
        try:
            metric = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid training metric row {expected_update}: {metrics_path}"
            ) from exc
        if metric.get("update") != expected_update:
            raise RuntimeError(
                f"training metric row {expected_update} records update "
                f"{metric.get('update')!r}"
            )
    return completed, metrics_count


def _truncate_metrics(path: Path, keep: int) -> None:
    """Drop metric rows written after the last saved state, so the metrics file
    and the resumed optimizer describe the same updates."""

    if not path.is_file():
        return
    rows = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    if len(rows) <= keep:
        return
    path.write_text("\n".join(rows[:keep]) + "\n", encoding="utf-8")


def _save_prompt_checkpoint(path: Path, prompt_parameter: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(
        {"prompt_embeddings": prompt_parameter.detach().cpu()},
        temporary,
    )
    temporary.replace(path)


def train(options: RunOptions, snapshot: Path) -> Path:
    resume = bool(getattr(options, "resume_training", False))
    distributed = torch.distributed
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if options.train_model_placement == "balanced" and world_size != 1:
        raise RuntimeError(
            "balanced train model placement is single-process model sharding; "
            "do not launch it with torchrun"
        )
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed training requires CUDA")
        torch.cuda.set_device(local_rank)
        distributed.init_process_group("nccl")
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=resume)
    if world_size > 1:
        distributed.barrier()
    config = load_config(options.config)
    snapshot_config_path = snapshot.parent / "config.yaml"
    if not snapshot_config_path.is_file():
        raise RuntimeError(
            f"snapshot rendering config does not exist: {snapshot_config_path}"
        )
    snapshot_config = load_config(snapshot_config_path)
    (
        trajectories,
        sequence_length_filter,
        trajectory_cap,
    ) = load_training_trajectories(
        snapshot,
        prompt_length=config.distillation.prompt_length,
        max_sequence_length=config.distillation.max_sequence_length,
        overlength_policy=config.distillation.overlength_policy,
        max_trajectories_per_question=(
            config.distillation.max_trajectories_per_question
        ),
    )
    if config.distillation.effective_batch_size % world_size != 0:
        raise RuntimeError(
            "drop_last distributed training requires effective_batch_size "
            f"({config.distillation.effective_batch_size}) to be divisible by "
            f"world_size ({world_size})"
        )
    template_tokenizer, runtime_identity = load_template_tokenizer(options, config)
    _student_tokenizer, student_identity = load_tokenizer(options, config, "student")
    require_matching_tokenizer_vocabulary(runtime_identity, student_identity)
    merge_model_identities(runtime_identity, student_identity)
    snapshot_identity = snapshot_template_identity(snapshot)
    snapshot_template_tokenizer = template_tokenizer
    snapshot_runtime_identity = runtime_identity
    if snapshot_config.template != config.template:
        snapshot_template_tokenizer, snapshot_runtime_identity = (
            load_template_tokenizer(options, snapshot_config)
        )
    if (
        snapshot_identity["tokenizer_vocab_sha256"]
        != snapshot_runtime_identity["tokenizer_vocab_sha256"]
    ):
        raise RuntimeError(
            "canonical template tokenizer vocabulary differs from snapshot identity"
        )
    if (
        snapshot_identity["template_chat_sha256"]
        != snapshot_runtime_identity["template_chat_sha256"]
    ):
        raise RuntimeError("canonical chat template differs from snapshot identity")
    if snapshot_identity["template_source"] not in {
        None,
        snapshot_config.template.source,
    }:
        raise RuntimeError("canonical template source differs from snapshot identity")
    if snapshot_identity["template_render_mode"] not in {
        None,
        snapshot_config.template.render_mode,
    }:
        raise RuntimeError("template render mode differs from snapshot identity")
    if (
        snapshot_runtime_identity["tokenizer_vocab_sha256"]
        != runtime_identity["tokenizer_vocab_sha256"]
    ):
        raise RuntimeError(
            "current template tokenizer vocabulary differs from snapshot tokenizer"
        )
    runtime_identity["input_snapshot"] = {
        "path": str(snapshot),
        "content_sha256": sha256_file(snapshot),
    }
    runtime_identity["distributed"] = {
        "world_size": world_size,
        "model_placement": options.train_model_placement,
        "effective_batch_size": config.distillation.effective_batch_size,
        "micro_batch_size_per_rank": config.distillation.micro_batch_size,
        "cross_entropy_backend": config.distillation.cross_entropy_backend,
        "loss_normalization": config.distillation.loss_normalization,
        "batch_seed": config.distillation.batch_seed,
    }
    total_updates = config.distillation.updates_for_trajectory_count(len(trajectories))
    config.distillation.schedule.validate("distillation.schedule", total_updates)
    if rank == 0:
        sample_index = batch_indices_for_update(
            dataset_size=len(trajectories),
            effective_batch_size=config.distillation.effective_batch_size,
            batch_seed=config.distillation.batch_seed,
            update=1,
        )[0]
        sample_trajectory = trajectories[sample_index]
        sample_rendered = render_for_execution(
            config,
            execution_model="student",
            row=sample_trajectory,
            tokenizer=template_tokenizer,
        )
        sample_snapshot_rendered = (
            sample_rendered
            if snapshot_config.template == config.template
            else render_for_execution(
                snapshot_config,
                execution_model="teacher",
                row=sample_trajectory,
                tokenizer=snapshot_template_tokenizer,
            )
        )
        validate_snapshot_prompt_reuse(
            sample_snapshot_rendered,
            sample_rendered,
            tuple(
                int(token_id) for token_id in sample_trajectory["prompt_ids"]
            ),
        )
        sample_insert_at = resolve_insertion_index(config, sample_rendered)
        log_training_sample(
            training_sample_payload(
                sample_trajectory,
                sample_rendered,
                trajectory_index=sample_index,
                insert_at=sample_insert_at,
                soft_prompt_length=config.distillation.prompt_length,
            )
        )
    student_source, student_revision = model_source(options, config, "student")
    student = load_model(
        student_source,
        student_revision,
        placement=options.train_model_placement,
        attention_implementation=config.distillation.attention_implementation,
        trust_remote_code=config.student.trust_remote_code,
        config_overrides=config.student.config_overrides,
    )
    training_device = student.get_input_embeddings().weight.device
    prompt = initialize_prompt_embeddings(
        student,
        template_tokenizer,
        config.distillation.prompt_init,
        config.distillation.prompt_length,
    )
    wrapper = SoftPromptModel(
        student,
        prompt,
        activation_checkpointing=config.distillation.activation_checkpointing,
        cross_entropy_backend=config.distillation.cross_entropy_backend,
        loss_normalization=config.distillation.loss_normalization,
    )
    optimizer = torch.optim.AdamW(
        [wrapper.prompt_parameter],
        lr=config.distillation.learning_rate,
        weight_decay=config.distillation.weight_decay,
    )
    checkpoint_update_set = set(
        scheduled_checkpoint_updates(
            total_updates, config.distillation.checkpoint_every_updates
        )
    )
    checkpoint_paths: list[Path] = []
    metrics_path = options.output / "training_metrics.jsonl"
    metrics_handle = None
    metrics_count = 0
    start_update = 1
    state = _load_training_state(options.output) if resume else None
    if state is not None:
        from safetensors.torch import load_file

        completed, metrics_count = _validate_resume_step(
            state, metrics_path, total_updates
        )
        restored = (
            options.output / "checkpoints" / f"update_{completed:08d}.safetensors"
        )
        if not restored.is_file():
            raise RuntimeError(
                f"training state records update {completed} but its checkpoint is "
                f"missing: {restored}"
            )
        if completed >= total_updates:
            raise RuntimeError(
                f"training state already covers all {total_updates} updates"
            )
        with torch.no_grad():
            wrapper.prompt_parameter.copy_(
                load_file(restored)["prompt_embeddings"].to(
                    wrapper.prompt_parameter.device
                )
            )
        optimizer.load_state_dict(state["optimizer"])
        start_update = completed + 1
        if rank == 0:
            _truncate_metrics(metrics_path, metrics_count)
            metrics_handle = metrics_path.open("a", encoding="utf-8")
            checkpoint_paths.extend(
                sorted((options.output / "checkpoints").glob("update_*.safetensors"))
            )
            print(
                f"resuming training at update {start_update}/{total_updates} "
                f"from {restored.name}",
                flush=True,
            )
    elif rank == 0:
        metrics_handle = metrics_path.open("w", encoding="utf-8")
        initial_checkpoint = (
            options.output / "checkpoints" / "update_00000000.safetensors"
        )
        _save_prompt_checkpoint(initial_checkpoint, wrapper.prompt_parameter)
        checkpoint_paths.append(initial_checkpoint)

    for update in range(start_update, total_updates + 1):
        global_indices = batch_indices_for_update(
            dataset_size=len(trajectories),
            effective_batch_size=config.distillation.effective_batch_size,
            batch_seed=config.distillation.batch_seed,
            update=update,
        )
        local_indices = global_indices[rank::world_size]
        learning_rate = learning_rate_for_update(
            base_learning_rate=config.distillation.learning_rate,
            schedule=config.distillation.schedule,
            update_step=update,
            total_updates=total_updates,
        )
        optimizer.zero_grad(set_to_none=True)
        local_loss_sum = 0.0
        local_batches = []
        for offset in range(
            0, len(local_indices), config.distillation.micro_batch_size
        ):
            micro_indices = local_indices[
                offset : offset + config.distillation.micro_batch_size
            ]
            examples = []
            positions = []
            for trajectory in (trajectories[index] for index in micro_indices):
                current = render_for_execution(
                    config,
                    execution_model="student",
                    row=trajectory,
                    tokenizer=template_tokenizer,
                )
                snapshot_rendered = (
                    current
                    if snapshot_config.template == config.template
                    else render_for_execution(
                        snapshot_config,
                        execution_model="teacher",
                        row=trajectory,
                        tokenizer=snapshot_template_tokenizer,
                    )
                )
                validate_snapshot_prompt_reuse(
                    snapshot_rendered,
                    current,
                    tuple(int(token_id) for token_id in trajectory["prompt_ids"]),
                )
                examples.append(
                    make_response_sft_example(
                        current.token_ids, trajectory["response_ids"]
                    )
                )
                # The response IDs are reusable across insertion-position
                # ablations because removing the control marker yields the same
                # canonical prompt IDs.  The insertion offset itself is owned by
                # the current config, not by the snapshot that happened to
                # generate those responses.
                positions.append(resolve_insertion_index(config, current))
            local_batches.append(
                collate_examples(
                    examples,
                    insert_at=positions,
                    pad_token_id=template_tokenizer.eos_token_id,
                )
            )
        local_response_tokens = sum(
            response_token_count(batch) for batch in local_batches
        )
        local_loss_denominator = (
            local_response_tokens
            if config.distillation.loss_normalization == "token"
            else sum(int(batch.input_ids.size(0)) for batch in local_batches)
        )
        global_token_tensor = torch.tensor(
            local_response_tokens, dtype=torch.int64, device=training_device
        )
        if world_size > 1:
            distributed.all_reduce(global_token_tensor, op=distributed.ReduceOp.SUM)
        global_response_tokens = int(global_token_tensor.item())
        if global_response_tokens <= 0:
            raise RuntimeError("global batch contains no response loss tokens")
        global_denominator_tensor = torch.tensor(
            local_loss_denominator, dtype=torch.int64, device=training_device
        )
        if world_size > 1:
            distributed.all_reduce(
                global_denominator_tensor, op=distributed.ReduceOp.SUM
            )
        global_loss_denominator = int(global_denominator_tensor.item())
        if global_loss_denominator <= 0:
            raise RuntimeError("global loss denominator must be positive")
        for cpu_batch in local_batches:
            batch = type(cpu_batch)(
                cpu_batch.input_ids.to(training_device),
                cpu_batch.attention_mask.to(training_device),
                cpu_batch.labels.to(training_device),
                cpu_batch.insert_at.to(training_device),
            )
            loss_sum, token_count = accumulate_response_loss(
                wrapper,
                batch,
                global_loss_denominator=global_loss_denominator,
            )
            local_loss_sum += loss_sum
            if token_count != response_token_count(cpu_batch):
                raise RuntimeError(
                    "response-token count changed during forward alignment"
                )
        result = finish_prompt_update(
            wrapper,
            optimizer,
            local_loss_sum=local_loss_sum,
            local_response_tokens=local_response_tokens,
            local_loss_denominator=local_loss_denominator,
            learning_rate=learning_rate,
            max_grad_norm=config.distillation.max_grad_norm,
        )
        if rank == 0:
            assert metrics_handle is not None
            steps_per_epoch = (
                len(trajectories) // config.distillation.effective_batch_size
            )
            metric = {
                "update": update,
                "epoch": (update - 1) // steps_per_epoch + 1,
                "trajectories": len(global_indices),
                "loss": result.loss,
                "response_tokens": result.response_tokens,
                "gradient_norm": result.gradient_norm,
                "learning_rate": result.learning_rate,
            }
            is_checkpoint_update = update in checkpoint_update_set
            if is_checkpoint_update:
                update_checkpoint = (
                    options.output / "checkpoints" / f"update_{update:08d}.safetensors"
                )
                _save_prompt_checkpoint(update_checkpoint, wrapper.prompt_parameter)
                checkpoint_paths.append(update_checkpoint)
            record_training_metric(
                metrics_handle,
                metric,
                total_updates=total_updates,
                checkpoint=is_checkpoint_update,
            )
            metrics_count += 1
            if is_checkpoint_update:
                # After the checkpoint and after its metric row, so a resume
                # restores an optimizer and a metrics file that agree.
                metrics_handle.flush()
                _save_training_state(
                    options.output, update, optimizer, metrics_count
                )
    checkpoint = options.output / "soft_prompt.safetensors"
    if rank == 0:
        assert metrics_handle is not None
        metrics_handle.close()
        _save_prompt_checkpoint(checkpoint, wrapper.prompt_parameter)
        runtime_identity["distillation"] = asdict(config.distillation)
        runtime_identity["distillation"]["resolved_total_updates"] = total_updates
        runtime_identity["sequence_length_filter"] = sequence_length_filter
        runtime_identity["trajectory_cap"] = trajectory_cap
        write_manifest(
            options,
            config,
            "train",
            {
                "trajectories": len(trajectories),
                "input_trajectories": sequence_length_filter["input_trajectories"],
                "filtered_overlength": sequence_length_filter["filtered_trajectories"],
                "capped_per_question": trajectory_cap["capped_trajectories"],
                "updates": metrics_count,
                "checkpoints": len(checkpoint_paths),
            },
            [checkpoint, metrics_path, *checkpoint_paths],
            runtime_identity,
            stage_source=Path(__file__),
        )
    if world_size > 1:
        distributed.barrier()
        distributed.destroy_process_group()
    del optimizer, wrapper, student, prompt
    gc.collect()
    torch.cuda.empty_cache()
    return checkpoint


__all__ = [
    "cap_trajectories_per_question",
    "filter_trajectories_by_sequence_length",
    "load_training_trajectories",
    "log_training_sample",
    "scheduled_checkpoint_updates",
    "train",
    "training_sample_payload",
]
