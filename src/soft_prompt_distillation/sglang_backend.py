"""Small, fail-closed adapter for SGLang generation."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .progress import log_progress


PROGRESS_INTERVAL_SECONDS = 60.0
PROGRESS_INTERVAL_ENV = "SOFT_PROMPT_SGLANG_PROGRESS_INTERVAL_SECONDS"
ATTENTION_BACKEND_ENV = "SOFT_PROMPT_SGLANG_ATTENTION_BACKEND"
SGLANG_LONG_CONTEXT_ENV = "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"
ENGINE_START_ATTEMPTS = 3
ENGINE_START_RETRY_DELAY_SECONDS = 5.0


def resolved_attention_backend() -> str | None:
    """Return the normalized SGLang backend selected by the environment."""

    attention_backend = os.environ.get(ATTENTION_BACKEND_ENV)
    if attention_backend is None:
        return None
    attention_backend = attention_backend.strip()
    if not attention_backend:
        raise RuntimeError(
            f"{ATTENTION_BACKEND_ENV} must be non-empty when set"
        )
    return attention_backend


def _progress_interval_seconds() -> float:
    raw = os.environ.get(PROGRESS_INTERVAL_ENV)
    if raw is None:
        return PROGRESS_INTERVAL_SECONDS
    try:
        interval_seconds = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{PROGRESS_INTERVAL_ENV} must be a positive finite number"
        ) from exc
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise RuntimeError(
            f"{PROGRESS_INTERVAL_ENV} must be a positive finite number"
        )
    return interval_seconds


@dataclass(frozen=True, slots=True)
class GenerationResult:
    token_ids: list[int]
    finish_reason: str


def _sampling_params(
    spec: Any,
    seed: int,
    eos_token_id: int | None,
    forbidden_token_ids: Sequence[int],
) -> dict[str, Any]:
    configured_stop_ids = list(getattr(spec, "stop_token_ids", ()))
    if eos_token_id is not None:
        configured_stop_ids.insert(0, eos_token_id)
    stop_token_ids = list(dict.fromkeys(configured_stop_ids))
    result = {
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "top_k": getattr(spec, "top_k", -1),
        "repetition_penalty": getattr(spec, "repetition_penalty", 1.0),
        "max_new_tokens": spec.max_new_tokens,
        "stop_token_ids": stop_token_ids,
        "sampling_seed": seed,
        # Keep the terminating token in output_ids so snapshots and evaluation
        # artifacts have an explicit, auditable EOS/stop policy.
        "no_stop_trim": True,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    }
    stop_strings = tuple(getattr(spec, "stop_strings", ()))
    if stop_strings:
        result["stop"] = list(stop_strings)
    if forbidden_token_ids:
        result["logit_bias"] = {
            int(token_id): -100.0 for token_id in forbidden_token_ids
        }
    return result


def _parse_results(
    raw: Any, expected: int, eos_token_id: int | None
) -> list[GenerationResult]:
    items = raw if isinstance(raw, list) else [raw]
    if len(items) != expected:
        raise RuntimeError(
            f"SGLang returned {len(items)} results for {expected} requests"
        )
    results = []
    for item in items:
        if not isinstance(item, dict) or "output_ids" not in item:
            raise RuntimeError("SGLang response does not contain output_ids")
        token_ids = [int(value) for value in item["output_ids"]]
        reason = item.get("meta_info", {}).get("finish_reason")
        if not isinstance(reason, dict):
            raise RuntimeError(f"unrecognized SGLang finish reason: {reason!r}")
        reason_type = reason.get("type")
        if reason_type == "stop":
            matched = reason.get("matched")
            if matched is None:
                raise RuntimeError(
                    f"SGLang stop reason does not identify what matched: {reason!r}"
                )
            matched_ids = [matched] if isinstance(matched, int) else matched
            if isinstance(matched_ids, list) and all(
                isinstance(value, int) for value in matched_ids
            ):
                if token_ids[-len(matched_ids) :] != matched_ids:
                    raise RuntimeError(
                        "SGLang no_stop_trim response does not end with its "
                        f"matched stop token IDs: {reason!r}"
                    )
                finish = (
                    "eos"
                    if eos_token_id is not None and eos_token_id in matched_ids
                    else "stop"
                )
            elif isinstance(matched, str):
                finish = "stop"
            else:
                raise RuntimeError(f"unrecognized SGLang stop match: {matched!r}")
        elif reason_type == "length":
            finish = "length"
        elif reason_type == "abort":
            finish = "abort"
        else:
            raise RuntimeError(f"unrecognized SGLang finish reason: {reason!r}")
        results.append(GenerationResult(token_ids, finish))
    return results


class SGLangRuntime:
    """Own one in-process SGLang engine and expose token/embedding batches."""

    def __init__(
        self,
        engine: Any,
        version: str,
        data_parallel_size: int = 1,
        context_length: int | None = None,
        reserved_tokens: int = 0,
    ) -> None:
        if data_parallel_size <= 0:
            raise ValueError("SGLang data parallel size must be positive")
        if context_length is not None and context_length <= 0:
            raise ValueError("SGLang context length must be positive")
        if reserved_tokens < 0:
            raise ValueError("SGLang reserved token count must be nonnegative")
        self._engine = engine
        self.version = version
        self.data_parallel_size = data_parallel_size
        self.context_length = context_length
        self.reserved_tokens = reserved_tokens
        engine_loop = getattr(engine, "loop", None)
        if engine_loop is not None and engine_loop.is_closed():
            raise RuntimeError("SGLang Engine event loop is already closed")

        def loop_factory() -> asyncio.AbstractEventLoop:
            if engine_loop is None:
                loop = asyncio.new_event_loop()
            else:
                loop = engine_loop
            asyncio.set_event_loop(loop)
            return loop

        # SGLang's TokenizerManager binds itself to the first event loop used
        # by async_generate.  One persistent Runner is therefore required when
        # callers split a large sampling workload into multiple request batches.
        self._runner = asyncio.Runner(loop_factory=loop_factory)

    @classmethod
    def start(
        cls,
        model_path: str,
        tensor_parallel_size: int,
        data_parallel_size: int,
        download_dir: Path,
        log_level: str,
        trust_remote_code: bool = False,
        context_length: int | None = None,
        model_config_overrides: Mapping[str, Any] | None = None,
    ) -> "SGLangRuntime":
        if tensor_parallel_size <= 0:
            raise ValueError("SGLang tensor parallel size must be positive")
        if data_parallel_size <= 0:
            raise ValueError("SGLang data parallel size must be positive")
        if log_level not in {"debug", "info", "warning", "error", "critical"}:
            raise ValueError(f"unsupported SGLang log level: {log_level!r}")
        try:
            import sglang
            from sglang import Engine
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "SGLang generation requires a CUDA-capable SGLang installation"
            ) from exc
        engine_options: dict[str, Any] = {}
        if trust_remote_code:
            engine_options["trust_remote_code"] = True
        if context_length is not None:
            if context_length <= 0:
                raise ValueError("SGLang context length must be positive")
            engine_options["context_length"] = context_length
        if model_config_overrides:
            engine_options["json_model_override_args"] = json.dumps(
                dict(model_config_overrides),
                sort_keys=True,
                separators=(",", ":"),
            )
        attention_backend = resolved_attention_backend()
        if attention_backend is not None:
            engine_options["attention_backend"] = attention_backend
        # SGLang validates an explicit context_length against the model's
        # unmodified config before applying json_model_override_args.  A model
        # with an explicit RoPE scaling override (for example Nemotron's
        # published YaRN-2x recipe) therefore needs this narrowly scoped opt-in
        # during Engine construction.  Restore the caller's environment after
        # child processes have inherited it so unrelated engines are unchanged.
        rope_scaling = (
            model_config_overrides.get("rope_scaling")
            if model_config_overrides
            else None
        )
        allow_longer_context = context_length is not None and bool(rope_scaling)
        previous_long_context = os.environ.get(SGLANG_LONG_CONTEXT_ENV)
        if allow_longer_context:
            os.environ[SGLANG_LONG_CONTEXT_ENV] = "1"
        try:
            for attempt in range(1, ENGINE_START_ATTEMPTS + 1):
                try:
                    engine = Engine(
                        model_path=model_path,
                        tp_size=tensor_parallel_size,
                        dp_size=data_parallel_size,
                        download_dir=str(download_dir),
                        log_level=log_level,
                        **engine_options,
                    )
                    break
                except RuntimeError:
                    if attempt == ENGINE_START_ATTEMPTS:
                        raise
                    # SGLang can return from a previous Engine.shutdown before
                    # all data-parallel listener sockets have retired. A fresh
                    # randomly allocated rendezvous port succeeds once those
                    # child processes finish exiting.
                    time.sleep(ENGINE_START_RETRY_DELAY_SECONDS)
        finally:
            if allow_longer_context:
                if previous_long_context is None:
                    os.environ.pop(SGLANG_LONG_CONTEXT_ENV, None)
                else:
                    os.environ[SGLANG_LONG_CONTEXT_ENV] = previous_long_context
        for attribute in ("async_generate", "shutdown"):
            if not hasattr(engine, attribute):
                engine.shutdown()
                raise RuntimeError(
                    f"installed SGLang lacks required Engine.{attribute} interface"
                )
        tokenizer_manager = getattr(engine, "tokenizer_manager", None)
        resolved_context_length = getattr(tokenizer_manager, "context_len", None)
        reserved_tokens = int(
            getattr(tokenizer_manager, "num_reserved_tokens", 0)
        )
        return cls(
            engine,
            str(sglang.__version__),
            data_parallel_size,
            context_length=(
                int(resolved_context_length)
                if resolved_context_length is not None
                else context_length
            ),
            reserved_tokens=reserved_tokens,
        )

    def generate_token_ids(
        self,
        prompts: Sequence[Sequence[int]],
        spec: Any | Sequence[Any],
        seeds: Sequence[int],
        eos_token_id: int | None,
        forbidden_token_ids: Sequence[int] = (),
        progress_label: str = "sglang",
        data_parallel_ranks: Sequence[int] | None = None,
    ) -> list[GenerationResult]:
        _validate_batch(
            prompts,
            seeds,
            data_parallel_ranks,
            data_parallel_size=self.data_parallel_size,
        )
        specs = _generation_specs(spec, len(prompts))
        params = [
            _sampling_params(item_spec, seed, eos_token_id, forbidden_token_ids)
            for item_spec, seed in zip(specs, seeds, strict=True)
        ]
        if self.context_length is not None:
            for prompt, item_params in zip(prompts, params, strict=True):
                available_tokens = (
                    self.context_length - len(prompt) - self.reserved_tokens
                )
                if available_tokens <= 0:
                    raise ValueError(
                        "prompt plus SGLang reserved tokens exhausts the model "
                        f"context length of {self.context_length} tokens"
                    )
                item_params["max_new_tokens"] = min(
                    item_params["max_new_tokens"], available_tokens
                )
        ranks: Sequence[int | None] = (
            [None] * len(prompts)
            if data_parallel_ranks is None
            else data_parallel_ranks
        )
        raw = self._runner.run(
            self._generate_token_ids_concurrently(
                prompts, params, ranks, progress_label=progress_label
            )
        )
        return _parse_results(raw, len(prompts), eos_token_id)

    async def _generate_token_ids_concurrently(
        self,
        prompts: Sequence[Sequence[int]],
        params: Sequence[dict[str, Any]],
        data_parallel_ranks: Sequence[int | None],
        *,
        progress_label: str,
    ) -> list[dict[str, Any]]:
        progress_interval_seconds = _progress_interval_seconds()
        # Register this request chunk at once. SGLang's scheduler, rather than a
        # second Python-side chunk barrier, controls admission within the chunk.
        tasks = []
        for prompt, param, rank in zip(
            prompts, params, data_parallel_ranks, strict=True
        ):
            kwargs = {
                "input_ids": list(prompt),
                "sampling_params": param,
            }
            if rank is not None:
                kwargs["data_parallel_rank"] = rank
            tasks.append(asyncio.create_task(self._engine.async_generate(**kwargs)))
        reporter = asyncio.create_task(
            _report_task_progress(
                tasks, progress_label, progress_interval_seconds
            )
        )
        try:
            return await asyncio.gather(*tasks)
        finally:
            reporter.cancel()
            with suppress(asyncio.CancelledError):
                await reporter

    def shutdown(self) -> None:
        try:
            self._engine.shutdown()
        finally:
            self._runner.close()


def _validate_batch(
    prompts: Sequence[Any],
    seeds: Sequence[int],
    data_parallel_ranks: Sequence[int] | None,
    *,
    data_parallel_size: int,
) -> None:
    if not prompts:
        raise ValueError("SGLang generation batch must not be empty")
    if len(prompts) != len(seeds):
        raise ValueError("SGLang generation requires one seed per prompt")
    if data_parallel_ranks is None:
        return
    if len(prompts) != len(data_parallel_ranks):
        raise ValueError("SGLang generation requires one DP rank per prompt")
    if any(
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or not 0 <= rank < data_parallel_size
        for rank in data_parallel_ranks
    ):
        raise ValueError(
            "SGLang data-parallel ranks must be integers in "
            f"[0, {data_parallel_size})"
        )


def _generation_specs(spec: Any | Sequence[Any], expected: int) -> list[Any]:
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        specs = list(spec)
        if len(specs) != expected:
            raise ValueError("SGLang generation requires one spec per prompt")
        return specs
    return [spec] * expected


async def _report_task_progress(
    tasks: Sequence[asyncio.Task[Any]],
    label: str,
    interval_seconds: float,
) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        finished = sum(task.done() for task in tasks)
        log_progress(
            f"stage={label} event=sglang_progress "
            f"finished={finished} total={len(tasks)}"
        )




__all__ = [
    "GenerationResult",
    "SGLangRuntime",
]
