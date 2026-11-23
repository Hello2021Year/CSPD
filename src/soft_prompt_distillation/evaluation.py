"""Single-arm SGLang evaluation stage."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
import transformers
from transformers import AutoTokenizer

from .common import (
    RunOptions,
    download_rows,
    load_template_tokenizer,
    load_tokenizer,
    merge_model_identities,
    model_source,
    resolved_request_batch_size,
    write_jsonl,
    write_manifest,
)
from .config import GenerationSpec, OffPolicyConfig, load_config
from .rendering import RenderedPrompt, render_for_execution, resolve_insertion_index
from .sglang_backend import SGLangRuntime, resolved_attention_backend
from .tasks import TaskAdapter, get_task
from .virtual_tokens import insert_virtual_token_ids, load_virtual_prompt_metadata


EVALUATION_SEED_OFFSET_ENV = "SOFT_PROMPT_EVALUATION_SEED_OFFSET"


def _evaluation_seed_offset() -> int:
    raw = os.environ.get(EVALUATION_SEED_OFFSET_ENV, "0")
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{EVALUATION_SEED_OFFSET_ENV} must be an integer"
        ) from exc


def _evaluation_seed(request_index: int) -> int:
    """Use one random stream for the same request across every evaluation arm."""

    return 2000 + _evaluation_seed_offset() + request_index


def _score_rows(
    requests: list[tuple[dict[str, Any], int, GenerationSpec]],
    rendered_inputs: list[RenderedPrompt],
    generated: Iterable[tuple[list[int], str]],
    tokenizer: Any,
    task: TaskAdapter,
    *,
    verifier_command: Path | None = None,
    verifier_timeout_seconds: int | None = None,
) -> list[dict[str, Any]]:
    scored = []
    normalized_rows = []
    raw_completions = []
    completions = []
    for (row, sample_index, _spec), rendered, (response_ids, finish) in zip(
        requests, rendered_inputs, generated, strict=True
    ):
        raw_completion = tokenizer.decode(response_ids, skip_special_tokens=False)
        completion = tokenizer.decode(response_ids, skip_special_tokens=True)
        parsed = task.parse_completion(completion)
        normalized_rows.append(row)
        raw_completions.append(raw_completion)
        completions.append(completion)
        scored.append(
            {
                "dataset": row["dataset"],
                "question_id": row["question_id"],
                "sample_index": sample_index,
                "input_messages": [dict(message) for message in rendered.messages],
                "input_text": rendered.text,
                "input_tokenization": rendered.tokenization_strategy,
                "input_token_count": len(rendered.token_ids),
                "input_token_ids_sha256": rendered.token_ids_sha256,
                "canonical_input_token_count": rendered.canonical_token_count,
                "canonical_input_token_ids_sha256": (
                    rendered.canonical_token_ids_sha256
                ),
                "input_tokenization_matches_canonical": (
                    rendered.tokenization_matches_canonical
                ),
                "response_ids": list(response_ids),
                "raw_completion": raw_completion,
                "completion": completion,
                "reasoning_text": parsed.reasoning_text,
                "final_text": parsed.final_text,
                "extracted_answer": parsed.extracted_answer,
                "extractor": parsed.extractor,
                "task_adapter": type(task).__name__,
                "finish_reason": finish,
            }
        )
    if hasattr(task, "grade_evaluation_batch"):
        grades = task.grade_evaluation_batch(
            completions,
            normalized_rows,
            verifier_command=verifier_command,
            verifier_timeout_seconds=verifier_timeout_seconds,
        )
    else:
        grades = [
            SimpleNamespace(
                correct=task.grade_evaluation(completion, row),
                grader=task.evaluation_grader(row),
                metadata={},
            )
            for completion, row in zip(
                completions, normalized_rows, strict=True
            )
        ]
    if len(grades) != len(scored):
        raise RuntimeError("task evaluation grader returned the wrong result count")
    for row, grade in zip(scored, grades, strict=True):
        row["grader"] = grade.grader
        row["grader_metadata"] = dict(grade.metadata)
        row["correct"] = grade.correct
    return scored


def _benchmark_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)
    return {
        dataset: {
            "questions": len({str(row["question_id"]) for row in dataset_rows}),
            "samples": len(dataset_rows),
            "accuracy": sum(bool(row["correct"]) for row in dataset_rows)
            / len(dataset_rows),
        }
        for dataset, dataset_rows in sorted(grouped.items())
    }


def _macro_accuracy(benchmarks: dict[str, dict[str, Any]]) -> float:
    if not benchmarks:
        raise RuntimeError("cannot aggregate an empty benchmark set")
    return sum(float(summary["accuracy"]) for summary in benchmarks.values()) / len(
        benchmarks
    )


def _tokenization_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    questions: dict[tuple[str, str], tuple[Any, ...]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["question_id"]))
        value = (
            row["input_tokenization"],
            int(row["input_token_count"]),
            row["input_token_ids_sha256"],
            int(row["canonical_input_token_count"]),
            row["canonical_input_token_ids_sha256"],
            bool(row["input_tokenization_matches_canonical"]),
        )
        previous = questions.setdefault(key, value)
        if previous != value:
            raise RuntimeError(
                "repeated evaluation samples use inconsistent prompt tokenization"
            )
    strategies = {value[0] for value in questions.values()}
    if len(strategies) != 1:
        raise RuntimeError("evaluation inputs use multiple tokenization strategies")
    matches = sum(bool(value[-1]) for value in questions.values())
    return {
        "strategy": next(iter(strategies)),
        "questions": len(questions),
        "matches_canonical": matches,
        "differs_from_canonical": len(questions) - matches,
    }


def _generation_for_dataset(
    config: OffPolicyConfig, arm: str, dataset: str
) -> GenerationSpec:
    benchmarks = getattr(config.evaluation, "benchmarks", {})
    return benchmarks.get(dataset, config.evaluation.arms[arm].generation)


def _generation_identity(spec: GenerationSpec) -> dict[str, Any]:
    return {
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "top_k": spec.top_k,
        "repetition_penalty": spec.repetition_penalty,
        "max_new_tokens": spec.max_new_tokens,
        "samples_per_prompt": spec.samples_per_prompt,
        "stop_token_ids": list(spec.stop_token_ids),
        "stop_strings": list(spec.stop_strings),
        "provenance": spec.provenance,
    }


def _generate_and_score(
    options: RunOptions,
    config: OffPolicyConfig,
    task: TaskAdapter,
    rows: list[dict[str, Any]],
    template_tokenizer: Any,
    execution_tokenizer: Any,
    arm: str,
    soft_prompt_model_path: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if options.sglang_tp_size <= 0:
        raise RuntimeError("--sglang-tp-size must be positive")
    if options.sglang_dp_size <= 0:
        raise RuntimeError("--sglang-dp-size must be positive")
    if arm not in config.evaluation.arms:
        raise RuntimeError(f"evaluation arm {arm!r} is not declared in the config")
    requests = [
        (row, sample_index, _generation_for_dataset(config, arm, row["dataset"]))
        for row in rows
        for sample_index in range(
            _generation_for_dataset(config, arm, row["dataset"]).samples_per_prompt
        )
    ]
    # SGLang non-deterministic DP workers may start with identical RNG state.
    # Keep all repetitions of a question on one worker, as training sampling
    # already does, including when repetitions cross a request-batch boundary.
    request_ranks = [
        row_index % options.sglang_dp_size
        for row_index, row in enumerate(rows)
        for _ in range(
            _generation_for_dataset(config, arm, row["dataset"]).samples_per_prompt
        )
    ]
    request_batch_size = resolved_request_batch_size(
        options.sglang_batch_size, len(requests)
    )
    eos_token_id = execution_tokenizer.eos_token_id
    role = "teacher" if arm == "teacher" else "student"
    forbidden_ids: tuple[int, ...] = ()
    virtual_prompt = None
    if arm == "student_soft_prompt":
        if soft_prompt_model_path is None:
            raise RuntimeError(
                "student_soft_prompt evaluation requires --soft-prompt-model-path"
            )
        virtual_prompt = load_virtual_prompt_metadata(soft_prompt_model_path)
        derived_tokenizer = AutoTokenizer.from_pretrained(
            soft_prompt_model_path,
            trust_remote_code=getattr(config.student, "trust_remote_code", False),
        )
        resolved_virtual_ids = tuple(
            int(derived_tokenizer.convert_tokens_to_ids(token))
            for token in virtual_prompt.tokens
        )
        if resolved_virtual_ids != virtual_prompt.token_ids:
            raise RuntimeError(
                "derived tokenizer does not match soft-prompt token metadata"
            )
        source = str(soft_prompt_model_path)
        forbidden_ids = virtual_prompt.token_ids
    else:
        source = model_source(options, config, role)[0]

    runtime_options = {}
    if getattr(
        getattr(config, role, None), "trust_remote_code", False
    ):
        runtime_options["trust_remote_code"] = True
    model_spec = getattr(config, role, None)
    if getattr(model_spec, "context_length", None) is not None:
        runtime_options["context_length"] = model_spec.context_length
    if getattr(model_spec, "config_overrides", None):
        runtime_options["model_config_overrides"] = model_spec.config_overrides
    if options.generation_backend == "sglang":
        runtime_type = SGLangRuntime
    elif options.generation_backend == "vllm":
        if arm == "student_soft_prompt":
            raise RuntimeError(
                "vLLM is currently supported for frozen student/teacher baselines only"
            )
        from .vllm_backend import VLLMRuntime

        runtime_type = VLLMRuntime
    else:
        raise RuntimeError(
            f"unsupported generation backend: {options.generation_backend!r}"
        )
    runtime = runtime_type.start(
        source,
        options.sglang_tp_size,
        options.sglang_dp_size,
        options.cache_dir,
        options.sglang_log_level,
        **runtime_options,
    )
    pipeline_verifier = bool(
        options.code_verifier_command is not None
        and getattr(task, "requires_external_verifier", False)
    )
    verifier_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="code-verifier")
        if pipeline_verifier
        else None
    )
    inflight: deque[Future[list[dict[str, Any]]]] = deque()
    scored: list[dict[str, Any]] = []

    try:
        for offset in range(0, len(requests), request_batch_size):
            batch_requests = requests[offset : offset + request_batch_size]
            batch_rows = [row for row, _sample_index, _spec in batch_requests]
            batch_specs = [spec for _row, _sample_index, spec in batch_requests]
            rendered = [
                render_for_execution(
                    config,
                    execution_model=role,
                    row=row,
                    tokenizer=template_tokenizer,
                    execution_tokenizer=execution_tokenizer,
                    template=getattr(config.evaluation, "template", None),
                )
                for row in batch_rows
            ]
            prompts = [list(item.token_ids) for item in rendered]
            if virtual_prompt is not None:
                prompts = [
                    insert_virtual_token_ids(
                        item.token_ids,
                        resolve_insertion_index(config, item),
                        virtual_prompt.token_ids,
                    )
                    for item in rendered
                ]
            seeds = [
                _evaluation_seed(index)
                for index in range(offset, offset + len(batch_rows))
            ]
            routing_kwargs = (
                {"data_parallel_ranks": request_ranks[offset : offset + len(batch_rows)]}
                if options.generation_backend == "sglang"
                else {}
            )
            generated = list(
                runtime.generate_token_ids(
                    prompts,
                    batch_specs,
                    seeds,
                    eos_token_id,
                    forbidden_token_ids=forbidden_ids,
                    **routing_kwargs,
                )
            )
            score_args = (
                batch_requests,
                rendered,
                [(item.token_ids, item.finish_reason) for item in generated],
                execution_tokenizer,
                task,
            )
            score_kwargs = {
                "verifier_command": options.code_verifier_command,
                "verifier_timeout_seconds": (
                    options.code_verifier_timeout_seconds
                ),
            }
            if verifier_executor is None:
                scored.extend(_score_rows(*score_args, **score_kwargs))
            else:
                inflight.append(
                    verifier_executor.submit(
                        _score_rows, *score_args, **score_kwargs
                    )
                )
                # One CPU-verifier batch overlaps the next GPU-generation batch.
                if len(inflight) >= 2:
                    scored.extend(inflight.popleft().result())
    finally:
        runtime.shutdown()
        if verifier_executor is not None:
            verifier_executor.shutdown(wait=True, cancel_futures=True)
    while inflight:
        scored.extend(inflight.popleft().result())
    backend = {
        "name": options.generation_backend,
        "attention_backend": (
            resolved_attention_backend()
            if options.generation_backend == "sglang"
            else None
        ),
        "version": runtime.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "arm": arm,
        "model_source": source,
        "tensor_parallel_size": options.sglang_tp_size,
        "data_parallel_size": options.sglang_dp_size,
        "requested_batch_size": options.sglang_batch_size,
        "resolved_batch_size": request_batch_size,
        "default_generation": _generation_identity(
            config.evaluation.arms[arm].generation
        ),
        "samples_per_prompt": config.evaluation.arms[
            arm
        ].generation.samples_per_prompt,
        "benchmark_generation": {
            dataset: _generation_identity(
                _generation_for_dataset(config, arm, dataset)
            )
            for dataset in sorted({str(row["dataset"]) for row in rows})
        },
        "request_count": len(requests),
        "evaluation_seed_offset": _evaluation_seed_offset(),
        "evaluation_seed_formula": (
            "2000 + evaluation_seed_offset + request_index"
        ),
        "log_level": options.sglang_log_level,
        "context_length": getattr(model_spec, "context_length", None),
        "model_config_overrides": dict(
            getattr(model_spec, "config_overrides", {})
        ),
    }
    if options.generation_backend == "sglang":
        backend["data_parallel_routing"] = {
            "mode": "explicit_problem_modulo",
            "formula": "evaluation_row_index % data_parallel_size",
            "all_samples_for_problem_share_rank": True,
        }
    if soft_prompt_model_path is not None and arm == "student_soft_prompt":
        backend["soft_prompt_input"] = "virtual_token_ids"
        backend["soft_prompt_model_path"] = str(soft_prompt_model_path)
    return scored, backend


def evaluate(options: RunOptions, arm: str) -> Path:
    if arm not in {"student", "teacher", "student_soft_prompt"}:
        raise RuntimeError(f"unsupported evaluation arm: {arm!r}")
    if arm == "student_soft_prompt" and options.soft_prompt_model_path is None:
        raise RuntimeError(
            "student_soft_prompt evaluation requires --soft-prompt-model-path"
        )
    if arm != "student_soft_prompt" and options.soft_prompt_model_path is not None:
        raise RuntimeError(
            "--soft-prompt-model-path is only valid for student_soft_prompt evaluation"
        )
    options.output.mkdir(parents=True, exist_ok=False)
    config = load_config(options.config)
    task = get_task(config.task)
    rows, dataset_identities = download_rows(options, config, task, "evaluation")
    role = "teacher" if arm == "teacher" else "student"
    template_tokenizer, runtime_identity = load_template_tokenizer(options, config)
    if arm == "student_soft_prompt":
        execution_tokenizer = AutoTokenizer.from_pretrained(
            options.soft_prompt_model_path,
            trust_remote_code=getattr(config.student, "trust_remote_code", False),
        )
        _student_tokenizer, execution_identity = load_tokenizer(
            options, config, "student"
        )
    else:
        execution_tokenizer, execution_identity = load_tokenizer(
            options, config, role
        )
    merge_model_identities(runtime_identity, execution_identity)
    runtime_identity["datasets"] = dataset_identities
    arm_rows, backend_identity = _generate_and_score(
        options,
        config,
        task,
        rows,
        template_tokenizer,
        execution_tokenizer,
        arm,
        options.soft_prompt_model_path,
    )
    runtime_identity["evaluation_backend"] = backend_identity
    result_path = options.output / "results.jsonl"
    write_jsonl(result_path, arm_rows)
    benchmarks = _benchmark_summary(arm_rows)
    summary = {
        "status": "complete",
        "arm": arm,
        "questions": len(rows),
        "samples": len(arm_rows),
        "default_samples_per_prompt": config.evaluation.arms[
            arm
        ].generation.samples_per_prompt,
        "samples_per_prompt": config.evaluation.arms[
            arm
        ].generation.samples_per_prompt,
        "samples_per_prompt_by_benchmark": {
            dataset: _generation_for_dataset(
                config, arm, dataset
            ).samples_per_prompt
            for dataset in sorted(benchmarks)
        },
        "accuracy": _macro_accuracy(benchmarks),
        "aggregation": "macro_average_over_benchmarks",
        "benchmarks": benchmarks,
        "input_tokenization": _tokenization_summary(arm_rows),
    }
    summary_path = options.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_manifest(
        options,
        config,
        f"evaluate:{arm}",
        {"questions": len(rows), "samples": len(arm_rows)},
        [result_path, summary_path],
        runtime_identity,
        stage_source=Path(__file__),
    )
    return summary_path


__all__ = ["evaluate"]
