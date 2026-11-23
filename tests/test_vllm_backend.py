from __future__ import annotations

from types import SimpleNamespace

import pytest

from soft_prompt_distillation.vllm_backend import VLLMRuntime


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _LLM:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.calls = []

    def generate(self, *, prompts, sampling_params, use_tqdm):
        self.calls.append((prompts, sampling_params, use_tqdm))
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        token_ids=[seed.kwargs["seed"], 2],
                        finish_reason="stop",
                        stop_reason=2,
                    )
                ]
            )
            for seed in sampling_params
        ]


def _fake_vllm(monkeypatch):
    module = SimpleNamespace(
        __version__="0.10.2", LLM=_LLM, SamplingParams=_SamplingParams
    )
    monkeypatch.setitem(__import__("sys").modules, "vllm", module)


def test_vllm_runtime_preserves_token_ids_and_per_request_seeds(
    monkeypatch, tmp_path
) -> None:
    _fake_vllm(monkeypatch)
    runtime = VLLMRuntime.start(
        "student",
        tensor_parallel_size=2,
        data_parallel_size=1,
        download_dir=tmp_path,
        log_level="warning",
    )
    spec = SimpleNamespace(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        max_new_tokens=32768,
        stop_strings=(),
        stop_token_ids=(),
    )
    generated = runtime.generate_token_ids(
        [[10], [11]], [spec, spec], [102000, 102001], 2
    )

    assert runtime.version == "0.10.2"
    assert runtime._engine.options["tensor_parallel_size"] == 2
    assert [item.token_ids for item in generated] == [
        [102000, 2],
        [102001, 2],
    ]
    assert [item.finish_reason for item in generated] == ["eos", "eos"]
    prompts, params, use_tqdm = runtime._engine.calls[0]
    assert prompts == [
        {"prompt_token_ids": [10]},
        {"prompt_token_ids": [11]},
    ]
    assert [item.kwargs["seed"] for item in params] == [102000, 102001]
    assert all(item.kwargs["min_p"] == 0.0 for item in params)
    assert use_tqdm is False


def test_vllm_runtime_rejects_offline_data_parallel(monkeypatch, tmp_path) -> None:
    _fake_vllm(monkeypatch)
    with pytest.raises(ValueError, match="data_parallel_size=1"):
        VLLMRuntime.start(
            "student",
            tensor_parallel_size=1,
            data_parallel_size=2,
            download_dir=tmp_path,
            log_level="warning",
        )
