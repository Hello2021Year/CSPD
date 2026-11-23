"""Small offline-vLLM adapter used for backend reproducibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class GenerationResult:
    token_ids: list[int]
    finish_reason: str


class VLLMRuntime:
    """Own one offline vLLM engine and expose the token-ID interface we audit."""

    def __init__(self, engine: Any, version: str) -> None:
        self._engine = engine
        self.version = version

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
    ) -> "VLLMRuntime":
        if tensor_parallel_size <= 0:
            raise ValueError("vLLM tensor parallel size must be positive")
        if data_parallel_size != 1:
            raise ValueError(
                "offline vLLM evaluation currently requires data_parallel_size=1; "
                "use tensor parallelism for multiple GPUs"
            )
        try:
            import vllm
            from vllm import LLM
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "vLLM generation requires a CUDA-capable vLLM installation"
            ) from exc
        engine_options: dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "download_dir": str(download_dir),
            "trust_remote_code": trust_remote_code,
            "disable_log_stats": log_level not in {"debug", "info"},
        }
        if context_length is not None:
            engine_options["max_model_len"] = context_length
        if model_config_overrides:
            engine_options["hf_overrides"] = dict(model_config_overrides)
        return cls(LLM(**engine_options), str(vllm.__version__))

    def generate_token_ids(
        self,
        prompts: Sequence[Sequence[int]],
        specs: Sequence[Any],
        seeds: Sequence[int],
        eos_token_id: int | None,
        *,
        forbidden_token_ids: Sequence[int] = (),
    ) -> list[GenerationResult]:
        if not (len(prompts) == len(specs) == len(seeds)):
            raise ValueError("vLLM generation inputs must have equal lengths")
        if forbidden_token_ids:
            raise RuntimeError(
                "vLLM backend does not yet support soft-prompt token suppression"
            )
        from vllm import SamplingParams

        sampling_params = []
        for spec, seed in zip(specs, seeds, strict=True):
            stop_token_ids = list(getattr(spec, "stop_token_ids", ()))
            if eos_token_id is not None:
                stop_token_ids.insert(0, eos_token_id)
            sampling_params.append(
                SamplingParams(
                    temperature=spec.temperature,
                    top_p=spec.top_p,
                    top_k=getattr(spec, "top_k", -1),
                    min_p=0.0,
                    repetition_penalty=getattr(spec, "repetition_penalty", 1.0),
                    max_tokens=spec.max_new_tokens,
                    stop=list(getattr(spec, "stop_strings", ())) or None,
                    stop_token_ids=list(dict.fromkeys(stop_token_ids)) or None,
                    seed=int(seed),
                    skip_special_tokens=False,
                    spaces_between_special_tokens=False,
                    include_stop_str_in_output=True,
                )
            )
        outputs = self._engine.generate(
            prompts=[{"prompt_token_ids": list(prompt)} for prompt in prompts],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} results for {len(prompts)} requests"
            )
        results = []
        for request in outputs:
            if len(request.outputs) != 1:
                raise RuntimeError("vLLM returned an unexpected number of candidates")
            candidate = request.outputs[0]
            reason = str(candidate.finish_reason)
            if reason == "stop":
                finish = "eos" if candidate.stop_reason == eos_token_id else "stop"
            elif reason == "length":
                finish = "length"
            elif reason == "abort":
                finish = "abort"
            else:
                raise RuntimeError(f"unrecognized vLLM finish reason: {reason!r}")
            results.append(GenerationResult(list(candidate.token_ids), finish))
        return results

    def shutdown(self) -> None:
        # Offline LLM has no stable public shutdown method across vLLM releases.
        self._engine = None


__all__ = ["GenerationResult", "VLLMRuntime"]
