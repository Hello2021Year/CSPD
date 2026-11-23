"""Frozen-teacher trajectory sampling stage."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

from .common import (
    RunOptions,
    download_rows,
    load_template_tokenizer,
    model_source,
    read_json_rows,
    resolved_request_batch_size,
    sha256_file,
    snapshot_config,
    write_jsonl,
    write_manifest,
)
from .config import load_config
from .progress import log_progress
from .rendering import render_for_execution, resolve_insertion_index
from .sglang_backend import SGLangRuntime, resolved_attention_backend
from .tasks import get_task


_PART_SCHEMA_VERSION = 1


def _sampling_seed(
    row_index: int, sample_index: int, samples_per_prompt: int
) -> int:
    return 1000 + row_index * samples_per_prompt + sample_index


def _validate_source_prompt_ids(row: dict[str, Any], rendered: Any) -> None:
    source_prompt_ids = row.get("source_prompt_ids")
    if source_prompt_ids is None:
        return
    if tuple(source_prompt_ids) != tuple(rendered.token_ids):
        raise RuntimeError(
            "rendered sampling prompt does not match source prompt token IDs: "
            f"question_id={row.get('question_id')!r}"
        )


def _deduplicate_source_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int | str]]:
    """Exclude unverifiable rows and keep one problem when duplicate labels agree."""

    groups: dict[str, list[dict[str, Any]]] = {}
    empty_answer_rows = 0
    for row in rows:
        problem = row.get("problem")
        answer = row.get("answer")
        if not isinstance(problem, str) or not problem:
            raise RuntimeError("normalized sampling row has no problem")
        if not isinstance(answer, str):
            raise RuntimeError("normalized sampling row has a non-string answer")
        if not answer.strip():
            empty_answer_rows += 1
            continue
        groups.setdefault(problem, []).append(row)

    kept = []
    conflicting_groups = 0
    conflicting_rows = 0
    for group in groups.values():
        if len({row["answer"] for row in group}) != 1:
            conflicting_groups += 1
            conflicting_rows += len(group)
            continue
        kept.append(group[0])

    counts: dict[str, int | str] = {
        "key": "exact_problem_text",
        "policy": (
            "exclude_empty_answers_then_keep_first_if_all_exact_answers_agree_"
            "else_exclude_group"
        ),
        "raw_rows": len(rows),
        "empty_answer_rows_excluded": empty_answer_rows,
        "unique_problem_groups": len(groups),
        "duplicate_rows": len(rows) - empty_answer_rows - len(groups),
        "conflicting_problem_groups": conflicting_groups,
        "conflicting_rows_excluded": conflicting_rows,
        "rows_after_deduplication": len(kept),
    }
    if len({row["problem"] for row in kept}) != len(kept):
        raise RuntimeError("source deduplication failed to produce unique problems")
    return kept, counts


def _validate_question_id_mapping(
    rows: list[dict[str, Any]], task_name: str
) -> None:
    """Require each question id to identify exactly one normalized problem."""

    problem_by_question_id: dict[str, str] = {}
    for row in rows:
        question_id = row.get("question_id")
        problem = row.get("problem")
        if not isinstance(question_id, str) or not question_id:
            raise RuntimeError(
                f"task {task_name} produced an empty or non-string question_id"
            )
        if not isinstance(problem, str) or not problem:
            raise RuntimeError(f"task {task_name} produced an invalid problem")
        previous_problem = problem_by_question_id.setdefault(question_id, problem)
        if previous_problem != problem:
            raise RuntimeError(
                f"task {task_name} maps question_id {question_id!r} to multiple "
                "different problems; per-question trajectory caps would silently "
                "mix unrelated examples"
            )


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _part_specs(
    question_count: int, questions_per_part: int, samples_per_prompt: int
) -> list[dict[str, int]]:
    specs = []
    for part_index, row_start in enumerate(
        range(0, question_count, questions_per_part)
    ):
        row_end = min(question_count, row_start + questions_per_part)
        specs.append(
            {
                "part_index": part_index,
                "row_start": row_start,
                "row_end": row_end,
                "request_count": (row_end - row_start) * samples_per_prompt,
            }
        )
    return specs


def _pending_part_specs(
    source_row_indices: list[int],
    questions_per_part: int,
    samples_per_prompt: int,
) -> list[dict[str, Any]]:
    """Partition an arbitrary global-row subset without overlapping questions."""

    specs: list[dict[str, Any]] = []
    for part_index, pending_start in enumerate(
        range(0, len(source_row_indices), questions_per_part)
    ):
        pending_end = min(
            len(source_row_indices), pending_start + questions_per_part
        )
        indices = source_row_indices[pending_start:pending_end]
        specs.append(
            {
                "part_index": part_index,
                "row_start": pending_start,
                "row_end": pending_end,
                "source_row_indices": indices,
                "request_count": len(indices) * samples_per_prompt,
            }
        )
    return specs


def _sampling_generation_identity(config: Any) -> dict[str, Any]:
    identity = asdict(config.sampling.generation)
    identity.pop("provenance", None)
    return identity


def _load_reused_trajectories(
    options: RunOptions,
    config: Any,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[int], dict[str, Any]]:
    """Reuse completed exact-question samples and remap them to target row indices."""

    reuse_dir = options.reuse_sample_dir
    if reuse_dir is None:
        return [], set(), {}
    reuse_dir = reuse_dir.resolve()
    manifest_path = reuse_dir / "manifest.json"
    snapshot_path = reuse_dir / "teacher_trajectories.jsonl"
    config_path = reuse_dir / "config.yaml"
    for path in (manifest_path, snapshot_path, config_path):
        if not path.is_file():
            raise RuntimeError(f"reuse sample is incomplete: missing {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "sample":
        raise RuntimeError(f"reuse manifest is not a sample stage: {manifest_path}")
    expected_sha256 = manifest.get("content_sha256", {}).get(
        "teacher_trajectories.jsonl"
    )
    if expected_sha256 != sha256_file(snapshot_path):
        raise RuntimeError(f"reuse snapshot checksum mismatch: {snapshot_path}")

    previous_config = load_config(config_path)
    if previous_config.task != config.task:
        raise RuntimeError("reuse sample task differs from target task")
    if previous_config.teacher != config.teacher:
        raise RuntimeError("reuse sample teacher differs from target teacher")
    if previous_config.template != config.template:
        raise RuntimeError("reuse sample template differs from target template")
    if previous_config.sampling.trajectory_filter != "correct_only":
        raise RuntimeError("reuse currently requires a correct_only source snapshot")
    if config.sampling.trajectory_filter != "correct_only":
        raise RuntimeError("reuse currently requires a correct_only target")
    if _sampling_generation_identity(previous_config) != _sampling_generation_identity(
        config
    ):
        raise RuntimeError("reuse sample generation semantics differ from target")

    previous_options = RunOptions(
        config=config_path,
        output=reuse_dir,
        cache_dir=options.cache_dir,
        student_model_path=options.student_model_path,
        teacher_model_path=options.teacher_model_path,
    )
    task = get_task(config.task)
    previous_raw_rows, previous_dataset_identities = download_rows(
        previous_options, previous_config, task, "sampling"
    )
    previous_rows, previous_deduplication = _deduplicate_source_rows(
        previous_raw_rows
    )
    _validate_question_id_mapping(previous_rows, previous_config.task)
    previous_keys = {
        (row["problem"], row["answer"]) for row in previous_rows
    }
    target_index_by_key = {
        (row["problem"], row["answer"]): index
        for index, row in enumerate(rows)
    }
    reusable_keys = previous_keys & target_index_by_key.keys()
    reused_indices = {target_index_by_key[key] for key in reusable_keys}

    samples_per_prompt = config.sampling.generation.samples_per_prompt
    reused_trajectories = []
    seen_sample_keys: set[tuple[int, int]] = set()
    seen_responses: dict[int, set[tuple[int, ...]]] = {}
    for record in read_json_rows(snapshot_path):
        key = (record.get("problem"), record.get("answer"))
        if key not in reusable_keys:
            continue
        source_row_index = target_index_by_key[key]
        sample_index = record.get("sample_index")
        response_ids = record.get("response_ids")
        if (
            not isinstance(sample_index, int)
            or not 0 <= sample_index < samples_per_prompt
            or not isinstance(response_ids, list)
            or not all(isinstance(token_id, int) for token_id in response_ids)
        ):
            raise RuntimeError("reuse snapshot contains an invalid sampling record")
        sample_key = (source_row_index, sample_index)
        if sample_key in seen_sample_keys:
            raise RuntimeError(f"reuse snapshot duplicates request key {sample_key}")
        seen_sample_keys.add(sample_key)
        response_key = tuple(response_ids)
        question_responses = seen_responses.setdefault(source_row_index, set())
        if response_key in question_responses:
            raise RuntimeError(
                f"reuse snapshot duplicates a response for row {source_row_index}"
            )
        question_responses.add(response_key)
        remapped = dict(record)
        remapped["source_row_index"] = source_row_index
        remapped["question_id"] = rows[source_row_index]["question_id"]
        reused_trajectories.append(remapped)
    reused_trajectories.sort(
        key=lambda record: (record["source_row_index"], record["sample_index"])
    )
    return reused_trajectories, reused_indices, {
        "source_directory": str(reuse_dir),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_snapshot_sha256": expected_sha256,
        "source_dataset_identities": previous_dataset_identities,
        "source_deduplication": previous_deduplication,
        "source_questions": len(previous_rows),
        "overlapping_questions_reused": len(reused_indices),
        "nonoverlapping_source_questions": len(previous_keys - reusable_keys),
        "correct_trajectories_reused": len(reused_trajectories),
        "generation_requests_reused": len(reused_indices) * samples_per_prompt,
        "question_key": "exact_problem_text_and_exact_answer",
    }


def _part_directory(parts_dir: Path, part_index: int) -> Path:
    return parts_dir / f"part-{part_index:05d}"


def _load_part_manifest(
    parts_dir: Path,
    spec: dict[str, Any],
    identity_sha256: str,
) -> dict[str, Any] | None:
    directory = _part_directory(parts_dir, spec["part_index"])
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise RuntimeError(f"sampling part path is not a directory: {directory}")
    data_path = directory / "trajectories.jsonl"
    manifest_path = directory / "manifest.json"
    if not data_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"incomplete completed-part directory: {directory}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read sampling part manifest: {manifest_path}") from exc
    expected = {
        "schema_version": _PART_SCHEMA_VERSION,
        **spec,
        "identity_sha256": identity_sha256,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"sampling part identity mismatch for {directory}: {key}: "
                f"{manifest.get(key)!r} != {value!r}"
            )
    if manifest.get("content_sha256") != sha256_file(data_path):
        raise RuntimeError(f"sampling part checksum mismatch: {data_path}")
    counts = manifest.get("record_counts")
    if not isinstance(counts, dict) or any(
        not isinstance(counts.get(key), int) or counts[key] < 0
        for key in (
            "correct_trajectories_before_deduplication",
            "duplicate_responses_filtered",
            "trajectories",
        )
    ):
        raise RuntimeError(f"sampling part has invalid record counts: {manifest_path}")
    if not isinstance(manifest.get("runtime_version"), str):
        raise RuntimeError(f"sampling part has no runtime version: {manifest_path}")
    return manifest


def _write_part(
    parts_dir: Path,
    spec: dict[str, Any],
    identity_sha256: str,
    runtime_version: str,
    trajectories: list[dict[str, Any]],
    correct_before_deduplication: int,
    duplicates_filtered: int,
) -> dict[str, Any]:
    final_directory = _part_directory(parts_dir, spec["part_index"])
    if final_directory.exists():
        raise RuntimeError(f"refusing to overwrite sampling part: {final_directory}")
    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{final_directory.name}.tmp-",
            dir=parts_dir,
        )
    )
    data_path = temporary_directory / "trajectories.jsonl"
    write_jsonl(data_path, trajectories)
    manifest = {
        "schema_version": _PART_SCHEMA_VERSION,
        **spec,
        "identity_sha256": identity_sha256,
        "runtime_version": runtime_version,
        "record_counts": {
            "correct_trajectories_before_deduplication": (
                correct_before_deduplication
            ),
            "duplicate_responses_filtered": duplicates_filtered,
            "trajectories": len(trajectories),
        },
        "content_sha256": sha256_file(data_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (temporary_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_directory.replace(final_directory)
    return manifest


def _merge_parts(
    parts_dir: Path,
    specs: list[dict[str, Any]],
    snapshot: Path,
    samples_per_prompt: int,
    reused_trajectories: list[dict[str, Any]] | None = None,
) -> int:
    temporary = snapshot.with_suffix(".jsonl.tmp")
    trajectory_count = 0
    seen_sample_keys: set[tuple[int, int]] = set()
    with temporary.open("w", encoding="utf-8") as output:
        for record in reused_trajectories or []:
            row_index = record.get("source_row_index")
            sample_index = record.get("sample_index")
            if not isinstance(row_index, int) or not isinstance(sample_index, int):
                raise RuntimeError("invalid reused trajectory identity")
            sample_key = (row_index, sample_index)
            if sample_key in seen_sample_keys:
                raise RuntimeError(f"duplicate sampling request key: {sample_key}")
            seen_sample_keys.add(sample_key)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            trajectory_count += 1
        for spec in specs:
            data_path = (
                _part_directory(parts_dir, spec["part_index"])
                / "trajectories.jsonl"
            )
            replies_by_question: dict[int, set[tuple[int, ...]]] = {}
            allowed_row_indices = set(
                spec.get(
                    "source_row_indices",
                    range(spec["row_start"], spec["row_end"]),
                )
            )
            with data_path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"invalid sampling part row {data_path}:{line_number}"
                        ) from exc
                    row_index = record.get("source_row_index")
                    sample_index = record.get("sample_index")
                    response_ids = record.get("response_ids")
                    if (
                        not isinstance(row_index, int)
                        or row_index not in allowed_row_indices
                        or not isinstance(sample_index, int)
                        or not 0 <= sample_index < samples_per_prompt
                        or not isinstance(response_ids, list)
                        or not all(isinstance(value, int) for value in response_ids)
                    ):
                        raise RuntimeError(
                            f"invalid sampling part identity row {data_path}:{line_number}"
                        )
                    sample_key = (row_index, sample_index)
                    if sample_key in seen_sample_keys:
                        raise RuntimeError(f"duplicate sampling request key: {sample_key}")
                    seen_sample_keys.add(sample_key)
                    response_key = tuple(response_ids)
                    seen_replies = replies_by_question.setdefault(row_index, set())
                    if response_key in seen_replies:
                        raise RuntimeError(
                            f"duplicate response survived for source row {row_index}"
                        )
                    seen_replies.add(response_key)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    trajectory_count += 1
    temporary.replace(snapshot)
    return trajectory_count


def sample(options: RunOptions) -> Path:
    if options.sample_shard_count <= 0:
        raise RuntimeError("--sample-shard-count must be positive")
    if not 0 <= options.sample_shard_index < options.sample_shard_count:
        raise RuntimeError(
            "--sample-shard-index must be in [0, --sample-shard-count)"
        )
    options.output.mkdir(parents=True, exist_ok=True)
    options.cache_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(options.config)
    task = get_task(config.task)
    template_tokenizer, runtime_identity = load_template_tokenizer(options, config)
    raw_rows, dataset_identities = download_rows(options, config, task, "sampling")

    rows, source_deduplication = _deduplicate_source_rows(raw_rows)
    _validate_question_id_mapping(rows, config.task)
    samples_per_prompt = config.sampling.generation.samples_per_prompt
    reused_trajectories, reused_row_indices, reuse_identity = (
        _load_reused_trajectories(options, config, rows)
    )
    pending_row_indices = [
        index for index in range(len(rows)) if index not in reused_row_indices
    ]
    if len(reused_row_indices) + len(pending_row_indices) != len(rows):
        raise RuntimeError("reuse and pending question sets do not partition target rows")
    runtime_identity["datasets"] = dataset_identities
    runtime_identity["sampling"] = {
        "trajectory_filter": config.sampling.trajectory_filter,
        "generation": asdict(config.sampling.generation),
        "reply_uniqueness": {
            "scope": "within_source_problem",
            "key": "source_row_index,response_ids",
            "enforced": True,
        },
        "source_deduplication": source_deduplication,
    }
    if reuse_identity:
        runtime_identity["sampling"]["reuse"] = reuse_identity
    if options.sglang_tp_size <= 0:
        raise RuntimeError("--sglang-tp-size must be positive")
    if options.sglang_dp_size <= 0:
        raise RuntimeError("--sglang-dp-size must be positive")
    request_count = len(rows) * samples_per_prompt
    generation_request_count = len(pending_row_indices) * samples_per_prompt
    request_batch_size = resolved_request_batch_size(
        options.sglang_batch_size,
        generation_request_count if generation_request_count else request_count,
    )
    questions_per_part = max(1, request_batch_size // samples_per_prompt)
    resolved_part_request_count = min(
        generation_request_count, questions_per_part * samples_per_prompt
    )
    specs = (
        _pending_part_specs(
            pending_row_indices, questions_per_part, samples_per_prompt
        )
        if reuse_identity
        else _part_specs(len(rows), questions_per_part, samples_per_prompt)
    )
    eos_token_id = template_tokenizer.eos_token_id
    teacher_source, _teacher_revision = model_source(options, config, "teacher")
    teacher_spec = getattr(config, "teacher", None)
    runtime_identity["sampling"]["backend"] = {
        "name": "sglang",
        "attention_backend": resolved_attention_backend(),
        "tensor_parallel_size": options.sglang_tp_size,
        "data_parallel_size": options.sglang_dp_size,
        "requested_batch_size": options.sglang_batch_size,
        "resolved_batch_size": resolved_part_request_count,
        "log_level": options.sglang_log_level,
        "context_length": getattr(teacher_spec, "context_length", None),
        "model_config_overrides": dict(
            getattr(teacher_spec, "config_overrides", {})
        ),
        "deterministic_inference": False,
        "data_parallel_routing": {
            "mode": "explicit_problem_modulo",
            "formula": "deduplicated_row_index % data_parallel_size",
            "all_samples_for_problem_share_rank": True,
        },
    }
    runtime_identity["sampling"]["checkpointing"] = {
        "schema_version": _PART_SCHEMA_VERSION,
        "unit": "whole_questions",
        "questions_per_part": questions_per_part,
        "requests_per_full_part": resolved_part_request_count,
        "part_count": len(specs),
        "engine_lifecycle": "restart_per_part",
        "commit": "atomic_directory_rename",
    }
    if reuse_identity or options.sample_shard_count > 1:
        runtime_identity["sampling"]["checkpointing"].update(
            {
                "generated_questions": len(pending_row_indices),
                "reused_questions": len(reused_row_indices),
                "part_shard_count": options.sample_shard_count,
                "part_shard_formula": "part_index % part_shard_count",
            }
        )
    config_snapshot = snapshot_config(options)
    identity_sha256 = _sha256_json(
        {
            "config_sha256": sha256_file(config_snapshot),
            "runtime_identity": runtime_identity,
            "rows_sha256": _sha256_json(
                [
                    {
                        "question_id": row["question_id"],
                        "problem": row["problem"],
                        "answer": row["answer"],
                    }
                    for row in rows
                ]
            ),
            "sampling_source_sha256": sha256_file(Path(__file__)),
        }
    )
    parts_dir = options.output / "parts"
    parts_dir.mkdir(exist_ok=True)
    completed: dict[int, dict[str, Any]] = {}
    for spec in specs:
        manifest = _load_part_manifest(parts_dir, spec, identity_sha256)
        if manifest is not None:
            completed[spec["part_index"]] = manifest
    temporary_parts = sorted(parts_dir.glob(".part-*.tmp-*"))
    if temporary_parts:
        log_progress(
            "stage=sample event=ignored_incomplete_parts "
            f"count={len(temporary_parts)}"
        )
    completed_requests = sum(
        specs[index]["request_count"] for index in completed
    )
    kept_trajectories = sum(
        manifest["record_counts"]["trajectories"]
        for manifest in completed.values()
    )
    log_progress(
        "stage=sample event=generation_start "
        f"total={generation_request_count} completed={completed_requests} "
        f"reused_requests={len(reused_row_indices) * samples_per_prompt} "
        f"parts={len(completed)}/{len(specs)}"
    )
    completed_versions = {
        manifest["runtime_version"] for manifest in completed.values()
    }
    if len(completed_versions) > 1:
        raise RuntimeError("completed sampling parts use different SGLang versions")

    owned_specs = [
        spec
        for spec in specs
        if spec["part_index"] % options.sample_shard_count
        == options.sample_shard_index
    ]

    def grade_part(
        completions: list[str],
        pending: list[tuple[dict[str, Any], Any, int, int, int, int]],
    ) -> list[Any]:
        if config.sampling.trajectory_filter != "correct_only":
            return [None] * len(completions)
        if hasattr(task, "grade_sampling_batch"):
            return task.grade_sampling_batch(
                completions,
                [row for row, *_rest in pending],
                verifier_command=options.code_verifier_command,
                verifier_timeout_seconds=options.code_verifier_timeout_seconds,
            )
        return [
            SimpleNamespace(
                correct=task.grade(completion, row["answer"]),
                grader=f"{type(task).__name__}.grade",
                metadata={},
            )
            for completion, (row, *_rest) in zip(
                completions, pending, strict=True
            )
        ]

    def finalize_part(
        spec: dict[str, Any],
        pending: list[tuple[dict[str, Any], Any, int, int, int, int]],
        generated: list[Any],
        raw_completions: list[str],
        completions: list[str],
        runtime_version: str,
        grade_results: list[Any],
    ) -> None:
        nonlocal completed_requests, kept_trajectories
        part_trajectories = []
        seen_reply_keys: set[tuple[int, tuple[int, ...]]] = set()
        correct_before_deduplication = 0
        duplicates_filtered = 0
        for (
            row,
            rendered,
            _seed,
            _rank,
            source_row_index,
            sample_index,
        ), result, raw_completion, completion, grade_result in zip(
            pending,
            generated,
            raw_completions,
            completions,
            grade_results,
            strict=True,
        ):
            correct = grade_result.correct if grade_result is not None else None
            if config.sampling.trajectory_filter == "correct_only" and not correct:
                continue
            correct_before_deduplication += 1
            reply_key = (
                source_row_index,
                tuple(int(token_id) for token_id in result.token_ids),
            )
            if reply_key in seen_reply_keys:
                duplicates_filtered += 1
                continue
            seen_reply_keys.add(reply_key)
            part_trajectories.append(
                {
                    "question_id": row["question_id"],
                    "source_row_index": source_row_index,
                    "sample_index": sample_index,
                    "problem": row["problem"],
                    "answer": row["answer"],
                    "prompt_ids": list(rendered.token_ids),
                    "response_ids": result.token_ids,
                    "raw_completion": raw_completion,
                    "completion": completion,
                    "finish_reason": result.finish_reason,
                    "correct": correct,
                    "grader": (
                        grade_result.grader if grade_result is not None else None
                    ),
                    "grader_metadata": (
                        dict(grade_result.metadata)
                        if grade_result is not None
                        else None
                    ),
                    "insert_at": resolve_insertion_index(config, rendered),
                }
            )
        manifest = _write_part(
            parts_dir,
            spec,
            identity_sha256,
            runtime_version,
            part_trajectories,
            correct_before_deduplication,
            duplicates_filtered,
        )
        completed[spec["part_index"]] = manifest
        completed_requests += spec["request_count"]
        kept_trajectories += len(part_trajectories)
        log_progress(
            "stage=sample event=generation_progress "
            f"finished={completed_requests} total={generation_request_count} "
            f"kept={kept_trajectories} "
            f"parts={len(completed)}/{len(specs)}"
        )

    pipeline_verifier = bool(
        config.sampling.trajectory_filter == "correct_only"
        and options.code_verifier_command is not None
        and getattr(task, "requires_external_verifier", False)
    )
    verifier_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="code-verifier")
        if pipeline_verifier
        else None
    )
    inflight: deque[
        tuple[
            tuple[
                dict[str, Any],
                list[tuple[dict[str, Any], Any, int, int, int, int]],
                list[Any],
                list[str],
                list[str],
                str,
            ],
            Future[list[Any]],
        ]
    ] = deque()

    def drain_one_verifier_part() -> None:
        context, future = inflight.popleft()
        finalize_part(*context, future.result())

    try:
        for spec in owned_specs:
            part_index = spec["part_index"]
            if part_index in completed:
                continue
            if options.sample_finalize_only:
                continue
            log_progress(
                "stage=sample event=part_start "
                f"part={part_index + 1}/{len(specs)} "
                f"requests={spec['request_count']}"
            )
            pending = []
            source_row_indices = spec.get(
                "source_row_indices", range(spec["row_start"], spec["row_end"])
            )
            for row_index in source_row_indices:
                row = rows[row_index]
                rendered = render_for_execution(
                    config,
                    execution_model="teacher",
                    row=row,
                    tokenizer=template_tokenizer,
                )
                _validate_source_prompt_ids(row, rendered)
                for sample_index in range(samples_per_prompt):
                    pending.append(
                        (
                            row,
                            rendered,
                            _sampling_seed(
                                row_index, sample_index, samples_per_prompt
                            ),
                            row_index % options.sglang_dp_size,
                            row_index,
                            sample_index,
                        )
                    )
            runtime_options = {}
            if getattr(
                getattr(config, "teacher", None), "trust_remote_code", False
            ):
                runtime_options["trust_remote_code"] = True
            if getattr(teacher_spec, "context_length", None) is not None:
                runtime_options["context_length"] = teacher_spec.context_length
            if getattr(teacher_spec, "config_overrides", None):
                runtime_options["model_config_overrides"] = (
                    teacher_spec.config_overrides
                )
            runtime = SGLangRuntime.start(
                teacher_source,
                options.sglang_tp_size,
                options.sglang_dp_size,
                options.cache_dir,
                options.sglang_log_level,
                **runtime_options,
            )
            if completed_versions and runtime.version not in completed_versions:
                runtime.shutdown()
                raise RuntimeError(
                    "refusing to mix sampling parts from different SGLang versions"
                )
            try:
                generated = list(
                    runtime.generate_token_ids(
                        [item.token_ids for _row, item, *_rest in pending],
                        config.sampling.generation,
                        [seed for _row, _item, seed, *_rest in pending],
                        eos_token_id,
                        progress_label=f"sample_part_{part_index + 1}",
                        data_parallel_ranks=[
                            rank
                            for _row, _item, _seed, rank, *_rest in pending
                        ],
                    )
                )
                # SGLang Engine.shutdown() terminates descendant processes.
                # The isolated verifier launcher is also a descendant of this
                # sampling process, so let the previous part's verifier finish
                # before shutting down the engine for the current part.  This
                # still overlaps CPU verification of part N with GPU generation
                # of part N+1 without killing the verifier's local srun client.
                if inflight:
                    drain_one_verifier_part()
            finally:
                runtime.shutdown()
            completed_versions.add(runtime.version)
            raw_completions = [
                template_tokenizer.decode(
                    result.token_ids, skip_special_tokens=False
                )
                for result in generated
            ]
            completions = [
                template_tokenizer.decode(
                    result.token_ids, skip_special_tokens=True
                )
                for result in generated
            ]
            context = (
                spec,
                pending,
                generated,
                raw_completions,
                completions,
                runtime.version,
            )
            if verifier_executor is None:
                finalize_part(*context, grade_part(completions, pending))
            else:
                inflight.append(
                    (
                        context,
                        verifier_executor.submit(
                            grade_part, completions, pending
                        ),
                    )
                )
        while inflight:
            drain_one_verifier_part()
    finally:
        if verifier_executor is not None:
            verifier_executor.shutdown(wait=True, cancel_futures=True)

    missing_owned = [
        spec["part_index"]
        for spec in owned_specs
        if spec["part_index"] not in completed
    ]
    if missing_owned and not options.sample_finalize_only:
        raise RuntimeError(f"owned sampling parts are incomplete: {missing_owned}")
    if len(completed) != len(specs):
        if options.sample_finalize_only:
            missing = sorted(set(range(len(specs))) - completed.keys())
            raise RuntimeError(f"sampling parts are incomplete: {missing}")
        if options.sample_shard_count > 1:
            log_progress(
                "stage=sample event=shard_complete "
                f"shard={options.sample_shard_index}/{options.sample_shard_count} "
                f"parts={len(owned_specs)}"
            )
            return parts_dir
        raise RuntimeError("sampling parts are incomplete after generation")
    runtime_identity["sampling"]["backend"]["version"] = next(
        iter(completed_versions)
    )
    snapshot = options.output / "teacher_trajectories.jsonl"
    trajectory_count = _merge_parts(
        parts_dir,
        specs,
        snapshot,
        samples_per_prompt,
        reused_trajectories=reused_trajectories,
    )
    correct_trajectories_before_deduplication = sum(
        manifest["record_counts"]["correct_trajectories_before_deduplication"]
        for manifest in completed.values()
    ) + len(reused_trajectories)
    duplicate_responses_filtered = sum(
        manifest["record_counts"]["duplicate_responses_filtered"]
        for manifest in completed.values()
    )
    if trajectory_count != kept_trajectories + len(reused_trajectories):
        raise RuntimeError("merged sampling trajectory count does not match parts")
    record_counts = {
        "source_rows": len(raw_rows),
        "questions": len(rows),
        "generation_requests": request_count,
        "correct_trajectories_before_deduplication": (
            correct_trajectories_before_deduplication
        ),
        "duplicate_responses_filtered": duplicate_responses_filtered,
        "trajectories": trajectory_count,
        "parts": len(specs),
    }
    if reuse_identity:
        record_counts.update(
            {
                "generation_requests_reused": (
                    len(reused_row_indices) * samples_per_prompt
                ),
                "generation_requests_executed": generation_request_count,
                "questions_reused": len(reused_row_indices),
                "questions_generated": len(pending_row_indices),
            }
        )
    write_manifest(
        options,
        config,
        "sample",
        record_counts,
        [snapshot],
        runtime_identity,
        stage_source=Path(__file__),
    )
    return snapshot


__all__ = ["sample"]
