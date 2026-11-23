from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from soft_prompt_distillation import sglang_backend
from soft_prompt_distillation.sglang_backend import SGLangRuntime


SPEC = SimpleNamespace(temperature=0.6, top_p=0.95, max_new_tokens=32)


def test_resolved_attention_backend_normalizes_environment(monkeypatch) -> None:
    monkeypatch.delenv(sglang_backend.ATTENTION_BACKEND_ENV, raising=False)
    assert sglang_backend.resolved_attention_backend() is None
    monkeypatch.setenv(sglang_backend.ATTENTION_BACKEND_ENV, " flashinfer ")
    assert sglang_backend.resolved_attention_backend() == "flashinfer"
    monkeypatch.setenv(sglang_backend.ATTENTION_BACKEND_ENV, " ")
    with pytest.raises(RuntimeError, match="must be non-empty"):
        sglang_backend.resolved_attention_backend()


class _Engine:
    def __init__(self):
        self.token_calls = []
        self.stopped = False

    async def async_generate(self, **kwargs):
        self.token_calls.append(kwargs)
        return {
            "output_ids": [7],
            "meta_info": {"finish_reason": {"type": "length"}},
        }

    def shutdown(self):
        self.stopped = True


def test_sglang_runtime_keeps_token_ids_seeds_and_eos_policy() -> None:
    engine = _Engine()
    runtime = SGLangRuntime(engine, "test")
    result = runtime.generate_token_ids([[1, 2]], SPEC, [44], eos_token_id=2)
    assert result[0].token_ids == [7]
    assert result[0].finish_reason == "length"
    assert engine.token_calls[0]["input_ids"] == [1, 2]
    params = engine.token_calls[0]["sampling_params"]
    assert params["sampling_seed"] == 44
    assert params["top_k"] == -1
    assert params["repetition_penalty"] == 1.0
    assert params["stop_token_ids"] == [2]
    assert params["no_stop_trim"] is True


def test_sglang_runtime_allows_tokenizer_without_declared_eos() -> None:
    engine = _Engine()
    runtime = SGLangRuntime(engine, "test")

    result = runtime.generate_token_ids(
        [[1, 2]], SPEC, [44], eos_token_id=None
    )

    assert result[0].finish_reason == "length"
    assert engine.token_calls[0]["sampling_params"]["stop_token_ids"] == []


def test_sglang_runtime_forwards_top_k_and_repetition_penalty() -> None:
    engine = _Engine()
    runtime = SGLangRuntime(engine, "test")
    spec = SimpleNamespace(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.05,
        max_new_tokens=32,
    )

    runtime.generate_token_ids([[1, 2]], spec, [44], eos_token_id=2)

    params = engine.token_calls[0]["sampling_params"]
    assert params["top_k"] == 20
    assert params["repetition_penalty"] == 1.05


def test_sglang_runtime_supports_per_request_specs_and_extra_stops() -> None:
    engine = _Engine()
    runtime = SGLangRuntime(engine, "test")
    first = SimpleNamespace(
        temperature=0.6,
        top_p=0.95,
        max_new_tokens=32,
        stop_token_ids=(9,),
        stop_strings=("</final>",),
    )
    second = SimpleNamespace(
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=64,
        stop_token_ids=(),
        stop_strings=(),
    )

    runtime.generate_token_ids(
        [[1], [2]], [first, second], [41, 42], eos_token_id=3
    )

    first_params = engine.token_calls[0]["sampling_params"]
    second_params = engine.token_calls[1]["sampling_params"]
    assert first_params["stop_token_ids"] == [3, 9]
    assert first_params["stop"] == ["</final>"]
    assert first_params["max_new_tokens"] == 32
    assert second_params["stop_token_ids"] == [3]
    assert "stop" not in second_params
    assert second_params["max_new_tokens"] == 64


def test_sglang_runtime_caps_each_request_to_remaining_context() -> None:
    engine = _Engine()
    runtime = SGLangRuntime(
        engine,
        "test",
        context_length=10,
        reserved_tokens=1,
    )
    spec = SimpleNamespace(
        temperature=0.6,
        top_p=0.95,
        max_new_tokens=8,
    )

    runtime.generate_token_ids(
        [[1, 2], [3, 4, 5, 6]],
        spec,
        [41, 42],
        eos_token_id=2,
    )

    assert engine.token_calls[0]["sampling_params"]["max_new_tokens"] == 7
    assert engine.token_calls[1]["sampling_params"]["max_new_tokens"] == 5


def test_sglang_runtime_rejects_prompt_that_exhausts_context() -> None:
    runtime = SGLangRuntime(
        _Engine(),
        "test",
        context_length=3,
        reserved_tokens=1,
    )

    with pytest.raises(ValueError, match="exhausts the model context"):
        runtime.generate_token_ids(
            [[1, 2]], SPEC, [44], eos_token_id=2
        )


def test_sglang_runtime_rejects_wrong_number_of_generation_specs() -> None:
    runtime = SGLangRuntime(_Engine(), "test")
    with pytest.raises(ValueError, match="one spec per prompt"):
        runtime.generate_token_ids(
            [[1], [2]], [SPEC], [41, 42], eos_token_id=3
        )


def test_sglang_runtime_routes_requests_to_explicit_dp_ranks() -> None:
    engine = _Engine()
    runtime = SGLangRuntime(engine, "test", data_parallel_size=3)
    runtime.generate_token_ids(
        [[1], [2]],
        SPEC,
        [41, 42],
        eos_token_id=2,
        data_parallel_ranks=[2, 0],
    )
    assert [call["data_parallel_rank"] for call in engine.token_calls] == [2, 0]


@pytest.mark.parametrize("ranks", [[0], [0, 3], [0, -1], [0, True]])
def test_sglang_runtime_rejects_invalid_dp_ranks(ranks) -> None:
    runtime = SGLangRuntime(_Engine(), "test", data_parallel_size=3)
    with pytest.raises(ValueError, match="DP rank|data-parallel ranks"):
        runtime.generate_token_ids(
            [[1], [2]],
            SPEC,
            [41, 42],
            eos_token_id=2,
            data_parallel_ranks=ranks,
        )


def test_virtual_tokens_are_forbidden_as_generated_output() -> None:
    engine = _Engine()
    runtime = SGLangRuntime(engine, "test")
    runtime.generate_token_ids(
        [[1, 20, 21]], SPEC, [44], eos_token_id=2, forbidden_token_ids=[20, 21]
    )
    assert engine.token_calls[0]["sampling_params"]["logit_bias"] == {
        20: -100.0,
        21: -100.0,
    }


def test_token_id_requests_are_registered_concurrently() -> None:
    class ConcurrentEngine(_Engine):
        def __init__(self):
            super().__init__()
            self.started = 0
            self.all_started = None

        async def async_generate(self, **kwargs):
            if self.all_started is None:
                self.all_started = asyncio.Event()
            self.started += 1
            if self.started == 3:
                self.all_started.set()
            await self.all_started.wait()
            return {
                "output_ids": [7],
                "meta_info": {"finish_reason": {"type": "length"}},
            }

    engine = ConcurrentEngine()
    runtime = SGLangRuntime(engine, "test")
    results = runtime.generate_token_ids(
        [[1], [2], [3]], SPEC, [41, 42, 43], eos_token_id=2
    )
    assert engine.started == 3
    assert len(results) == 3


def test_sglang_runtime_reuses_engine_loop_across_request_batches() -> None:
    class SingleLoopEngine(_Engine):
        def __init__(self):
            super().__init__()
            self.loop = asyncio.new_event_loop()
            self.chosen_loop = None

        async def async_generate(self, **kwargs):
            current_loop = asyncio.get_running_loop()
            if self.chosen_loop is None:
                self.chosen_loop = current_loop
            else:
                assert current_loop is self.chosen_loop
            return await super().async_generate(**kwargs)

    engine = SingleLoopEngine()
    runtime = SGLangRuntime(engine, "test")
    try:
        first = runtime.generate_token_ids([[1]], SPEC, [41], eos_token_id=2)
        second = runtime.generate_token_ids([[2]], SPEC, [42], eos_token_id=2)
        assert [result.token_ids for result in first + second] == [[7], [7]]
        assert engine.chosen_loop is engine.loop
    finally:
        runtime.shutdown()

    assert engine.stopped is True
    assert engine.loop.is_closed()


def test_sglang_progress_heartbeat_preserves_result_order(
    monkeypatch, capsys
) -> None:
    class StaggeredEngine(_Engine):
        async def async_generate(self, **kwargs):
            token = kwargs["input_ids"][0]
            await asyncio.sleep(0.02 if token == 1 else 0.005)
            return {
                "output_ids": [token],
                "meta_info": {"finish_reason": {"type": "length"}},
            }

    monkeypatch.setenv(
        sglang_backend.PROGRESS_INTERVAL_ENV,
        "0.001",
    )
    runtime = SGLangRuntime(StaggeredEngine(), "test")

    results = runtime.generate_token_ids(
        [[1], [2]], SPEC, [41, 42], eos_token_id=3, progress_label="test"
    )

    assert [result.token_ids for result in results] == [[1], [2]]
    output = capsys.readouterr().out
    assert "stage=test event=sglang_progress" in output
    assert "total=2" in output


@pytest.mark.parametrize("value", ["", "0", "-1", "nan", "inf", "not-a-number"])
def test_sglang_progress_interval_environment_rejects_invalid_values(
    monkeypatch, value
) -> None:
    monkeypatch.setenv(sglang_backend.PROGRESS_INTERVAL_ENV, value)
    runtime = SGLangRuntime(_Engine(), "test")

    with pytest.raises(RuntimeError, match="positive finite number"):
        runtime.generate_token_ids([[1]], SPEC, [41], eos_token_id=3)


def test_finish_reason_distinguishes_non_eos_stop_token() -> None:
    engine = _Engine()
    async def generate(**_kwargs):
        return {
            "output_ids": [7, 9],
            "meta_info": {
                "finish_reason": {"type": "stop", "matched": 9}
            },
        }

    engine.async_generate = generate
    runtime = SGLangRuntime(engine, "test")
    result = runtime.generate_token_ids([[1]], SPEC, [44], eos_token_id=2)
    assert result[0].finish_reason == "stop"


def test_official_length_reason_wins_even_if_last_token_is_eos() -> None:
    engine = _Engine()
    async def generate(**_kwargs):
        return {
            "output_ids": [7, 2],
            "meta_info": {
                "finish_reason": {"type": "length", "length": 2}
            },
        }

    engine.async_generate = generate
    runtime = SGLangRuntime(engine, "test")
    result = runtime.generate_token_ids([[1]], SPEC, [44], eos_token_id=2)
    assert result[0].finish_reason == "length"


def test_stop_reason_must_include_matched_value() -> None:
    engine = _Engine()
    async def generate(**_kwargs):
        return {
            "output_ids": [7, 2],
            "meta_info": {"finish_reason": {"type": "stop"}},
        }

    engine.async_generate = generate
    runtime = SGLangRuntime(engine, "test")
    with pytest.raises(RuntimeError, match="does not identify"):
        runtime.generate_token_ids([[1]], SPEC, [44], eos_token_id=2)


def test_sglang_batch_requires_one_seed_per_prompt() -> None:
    runtime = SGLangRuntime(_Engine(), "test")
    with pytest.raises(ValueError, match="one seed per prompt"):
        runtime.generate_token_ids([[1], [2]], SPEC, [1], eos_token_id=2)


def test_sglang_start_passes_runtime_options(monkeypatch, tmp_path) -> None:
    captured = {}
    captured_long_context_env = None

    class FakeEngine:
        def __init__(self, **kwargs):
            nonlocal captured_long_context_env
            captured.update(kwargs)
            captured_long_context_env = os.environ.get(
                sglang_backend.SGLANG_LONG_CONTEXT_ENV
            )

        async def async_generate(self, **_kwargs):
            return {}

        def shutdown(self):
            pass

    fake_sglang = SimpleNamespace(Engine=FakeEngine, __version__="test")
    monkeypatch.setitem(sys.modules, "sglang", fake_sglang)
    monkeypatch.setenv(
        sglang_backend.ATTENTION_BACKEND_ENV,
        "flashinfer",
    )

    runtime = SGLangRuntime.start(
        "org/model",
        2,
        3,
        Path(tmp_path),
        "warning",
        trust_remote_code=True,
        context_length=131072,
        model_config_overrides={
            "max_position_embeddings": 131072,
            "rope_scaling": {"rope_type": "yarn", "factor": 4.0},
        },
    )
    assert runtime.version == "test"
    assert runtime.data_parallel_size == 3
    assert captured_long_context_env == "1"
    assert sglang_backend.SGLANG_LONG_CONTEXT_ENV not in os.environ
    assert captured == {
        "model_path": "org/model",
        "tp_size": 2,
        "dp_size": 3,
        "download_dir": str(tmp_path),
        "log_level": "warning",
        "attention_backend": "flashinfer",
        "trust_remote_code": True,
        "context_length": 131072,
        "json_model_override_args": (
            '{"max_position_embeddings":131072,'
            '"rope_scaling":{"factor":4.0,"rope_type":"yarn"}}'
        ),
    }


def test_sglang_start_rejects_empty_attention_backend(
    monkeypatch, tmp_path
) -> None:
    class FakeEngine:
        def __init__(self, **_kwargs):
            raise AssertionError("engine must not start")

    monkeypatch.setitem(
        sys.modules,
        "sglang",
        SimpleNamespace(Engine=FakeEngine, __version__="test"),
    )
    monkeypatch.setenv(sglang_backend.ATTENTION_BACKEND_ENV, "  ")

    with pytest.raises(RuntimeError, match="must be non-empty"):
        SGLangRuntime.start("org/model", 1, 1, Path(tmp_path), "warning")


def test_sglang_start_retries_transient_engine_runtime_error(
    monkeypatch, tmp_path
) -> None:
    attempts = []
    sleeps = []

    class FakeEngine:
        def __init__(self, **_kwargs):
            attempts.append(True)
            if len(attempts) == 1:
                raise RuntimeError("scheduler rendezvous failed")

        async def async_generate(self, **_kwargs):
            return {}

        def shutdown(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "sglang",
        SimpleNamespace(Engine=FakeEngine, __version__="test"),
    )
    monkeypatch.setattr(sglang_backend.time, "sleep", sleeps.append)

    runtime = SGLangRuntime.start(
        "org/model", 1, 1, Path(tmp_path), "warning"
    )
    runtime.shutdown()

    assert len(attempts) == 2
    assert sleeps == [sglang_backend.ENGINE_START_RETRY_DELAY_SECONDS]


def test_sglang_start_rejects_nonpositive_context_length(
    monkeypatch, tmp_path
) -> None:
    class FakeEngine:
        def __init__(self, **_kwargs):
            raise AssertionError("engine must not start")

    monkeypatch.setitem(
        sys.modules,
        "sglang",
        SimpleNamespace(Engine=FakeEngine, __version__="test"),
    )
    with pytest.raises(ValueError, match="context length must be positive"):
        SGLangRuntime.start(
            "org/model",
            1,
            1,
            Path(tmp_path),
            "warning",
            context_length=0,
        )


def test_sglang_start_rejects_unknown_log_level(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported SGLang log level"):
        SGLangRuntime.start("org/model", 1, 1, Path(tmp_path), "verbose")
