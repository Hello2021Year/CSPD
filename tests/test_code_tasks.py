from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import pickle
import zlib

import pytest

from soft_prompt_distillation.config import load_config
from soft_prompt_distillation.code_verifier import (
    LCB_GRADER_ID,
    PYTHON_ASSERTIONS_GRADER_ID,
    RESPONSE_SCHEMA,
    run_external_code_verifier,
)
from soft_prompt_distillation.code_verifier_worker import (
    LCB_CANDIDATE_MEMORY_LIMIT_BYTES,
    _grade_assertions,
    _json_safe,
    _prepare_livecodebench_payload,
    _safe_livecodebench_check,
    _worker_count,
)
from soft_prompt_distillation.tasks import get_task
from soft_prompt_distillation.tasks.code_tasks import (
    MellumCodeTask,
    MimoCodeTask,
    NemotronCodeTask,
    AREAL_DATASET_REVISION,
    AREAL_UPSTREAM_COMMIT,
    LCB_DATASET_REVISION,
    LCB_RELEASE_V5_REVISION,
    LCB_STARTER_FORMAT,
    LCB_STDIN_FORMAT,
    LCB_SYSTEM_PROMPT,
    OLMES_UPSTREAM_COMMIT,
    areal_qwen3_user_content,
    extract_areal_python_code,
    extract_livecodebench_code,
    extract_open_instruct_python_code,
    livecodebench_user_prompt,
    olmo3_livecodebench_user_prompt,
)
from soft_prompt_distillation.tasks.klear_contamination import (
    KLEAR_LCB_V1_V6_NEAR_DUPLICATE_IDS,
)


def test_python_assertion_verifier_marks_embedded_null_candidate_incorrect() -> None:
    correct, metadata = _grade_assertions(
        "def add(a, b):\n    return a + b\x00",
        {"assertions": ["assert add(2, 3) == 5"]},
    )

    assert correct is False
    assert metadata["outcome"] == "fail"
    assert metadata["returncode"] != 0
    assert "null bytes" in metadata["stderr_tail"]


def test_python_assertion_verifier_runs_each_test_in_fresh_process() -> None:
    correct, metadata = _grade_assertions(
        "state = 0\n"
        "def first_call():\n"
        "    global state\n"
        "    state += 1\n"
        "    return state",
        {"assertions": ["assert first_call() == 1", "assert first_call() == 1"]},
    )

    assert correct is True
    assert metadata["test_results"] == [True, True]
    assert metadata["pass_rate"] == 1.0


def test_livecodebench_prompt_matches_pinned_upstream_without_starter() -> None:
    assert livecodebench_user_prompt("Add two integers.") == (
        "### Question:\nAdd two integers.\n\n"
        f"### Format: {LCB_STDIN_FORMAT}\n"
        "```python\n# YOUR CODE HERE\n```\n\n"
        "### Answer: (use the provided format with backticks)\n\n"
    )


def test_livecodebench_prompt_matches_pinned_upstream_with_starter() -> None:
    assert livecodebench_user_prompt("Increment x.", "def solve(x):\n    pass") == (
        "### Question:\nIncrement x.\n\n"
        f"### Format: {LCB_STARTER_FORMAT}\n"
        "```python\ndef solve(x):\n    pass\n```\n\n"
        "### Answer: (use the provided format with backticks)\n\n"
    )


def test_olmo3_livecodebench_prompt_matches_pinned_olmes_adapt() -> None:
    assert OLMES_UPSTREAM_COMMIT == "7bebf0eb148cd89d1d8a8096505f99d4fc48c209"
    assert olmo3_livecodebench_user_prompt("Add two integers.") == (
        "### Question:\nAdd two integers.\n\n"
        "### Format:\n"
        "Provide CONCISE reasoning on how to arrive at the answer in the "
        "<think> </think> tag.\n"
        f"{LCB_STDIN_FORMAT}\n"
        "```python\n# YOUR CODE HERE\n```\n\n"
        "### Answer: (use the provided format with backticks)\n\n"
    )


def test_livecodebench_extractor_uses_last_fenced_block() -> None:
    completion = (
        "reasoning\n```python\nprint('draft')\n```\n"
        "final\n```python\nprint('answer')\n```\n"
    )
    assert extract_livecodebench_code(completion) == "print('answer')"
    assert extract_livecodebench_code("print('unfenced')") == ""


def test_safe_lcb_check_reaps_candidate_and_returns_response(monkeypatch) -> None:
    class Receiver:
        def poll(self, timeout=None) -> bool:
            assert timeout == 12
            assert process.join_calls == []
            return True

        def recv(self):
            return ("ok", [True], {"execution time": 0.1})

        def close(self) -> None:
            pass

    class Sender:
        def close(self) -> None:
            pass

    class Process:
        exitcode = 0

        def __init__(self, **_kwargs) -> None:
            self.join_calls = []
            self.killed = False

        def start(self) -> None:
            pass

        def join(self, timeout=None) -> None:
            self.join_calls.append(timeout)

        def is_alive(self) -> bool:
            return False

        def kill(self) -> None:
            self.killed = True

    process = Process()

    class Context:
        def Pipe(self, *, duplex):
            assert duplex is False
            return Receiver(), Sender()

        def Process(self, **_kwargs):
            return process

    monkeypatch.setattr(
        "soft_prompt_distillation.code_verifier_worker.multiprocessing.get_context",
        lambda method: Context() if method == "fork" else None,
    )
    results, metadata = _safe_livecodebench_check(
        {"input_output": json.dumps({"inputs": ["1\n"], "outputs": ["1\n"]})},
        "print(1)",
        6,
    )
    assert results == [True]
    assert metadata == {"execution time": 0.1}
    assert process.join_calls == [0, None]
    assert process.killed is False


def test_safe_lcb_check_kills_and_marks_timed_out_candidate(monkeypatch) -> None:
    class Endpoint:
        def poll(self, timeout=None) -> bool:
            assert timeout == 19
            return False

        def close(self) -> None:
            pass

    class Process:
        exitcode = -9

        def __init__(self, **_kwargs) -> None:
            self.alive = True
            self.killed = False

        def start(self) -> None:
            pass

        def join(self, _timeout=None) -> None:
            pass

        def is_alive(self) -> bool:
            return self.alive

        def kill(self) -> None:
            self.killed = True
            self.alive = False

    process = Process()

    class Context:
        def Pipe(self, *, duplex):
            assert duplex is False
            return Endpoint(), Endpoint()

        def Process(self, **_kwargs):
            return process

    monkeypatch.setattr(
        "soft_prompt_distillation.code_verifier_worker.multiprocessing.get_context",
        lambda method: Context() if method == "fork" else None,
    )
    results, metadata = _safe_livecodebench_check(
        {
            "input_output": json.dumps(
                {"inputs": ["1\n", "2\n"], "outputs": ["1\n", "2\n"]}
            )
        },
        "while True: pass",
        6,
    )
    assert results == [-1, -1]
    assert process.killed is True
    assert metadata == {
        "error_code": -5,
        "error_message": "global_timeout",
        "candidate_exitcode": -9,
        "memory_limit_bytes": LCB_CANDIDATE_MEMORY_LIMIT_BYTES,
    }


def test_safe_lcb_check_drains_large_response_before_join(monkeypatch) -> None:
    blob = "x" * 1_000_000

    def send_large_response(_sample, _code, _timeout, connection) -> None:
        connection.send(("ok", [True], {"blob": blob}))
        connection.close()

    monkeypatch.setattr(
        "soft_prompt_distillation.code_verifier_worker._run_livecodebench_candidate",
        send_large_response,
    )
    results, metadata = _safe_livecodebench_check(
        {"input_output": json.dumps({"inputs": ["1\n"], "outputs": ["1\n"]})},
        "print(1)",
        1,
    )
    assert results == [True]
    assert metadata == {"blob": blob}


def test_open_instruct_code_extractor_uses_last_python_fence_or_full_output() -> None:
    completion = (
        "draft\n```python\nprint('draft')\n```\n"
        "final\n```\nprint('answer')\n```\n"
    )
    assert extract_open_instruct_python_code(completion) == "print('answer')"
    assert extract_open_instruct_python_code("print('unfenced')") == (
        "print('unfenced')"
    )


def test_klear_code_normalization_retains_tests() -> None:
    task = get_task("klear_code")
    row = task.normalize_sampling_row(
        {
            "hash": "question-1",
            "prompt": [{"role": "user", "content": "Solve it."}],
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(
                    {"inputs": ["2 3\n"], "outputs": ["5\n"]}
                ),
            },
        }
    )
    assert row["problem"] == "Solve it."
    assert row["question_id"] == "question-1"
    assert row["verifier_protocol"] == "livecodebench"
    assert row["verifier_payload"] == {
        "input_output": {
            "inputs": ["2 3\n"],
            "outputs": ["5\n"],
            "fn_name": None,
        }
    }
    assert json.loads(row["answer"]) == row["verifier_payload"]


def test_klear_code_normalization_retains_functional_assertions() -> None:
    task = get_task("klear_code")
    row = task.normalize_sampling_row(
        {
            "hash": "question-functional",
            "prompt": [{"role": "user", "content": "Implement add(a, b)."}],
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(
                    {"functional": ["assert add(2, 3) == 5"]}
                ),
            },
        }
    )
    assert row["problem"] == "Implement add(a, b)."
    assert row["question_id"] == "question-functional"
    assert row["verifier_protocol"] == "python_assertions"
    assert row["verifier_payload"] == {
        "assertions": ["assert add(2, 3) == 5"]
    }
    assert json.loads(row["answer"]) == row["verifier_payload"]


def test_klear_code_contract_pins_models_data_windows_and_generation() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root
        / "configs"
        / "code/klear-reasoner-8b.yaml"
    )
    task = get_task("klear_code")
    assert config.student.revision == "18e3d4f77405b05c6c12498d65348b7bd6112c25"
    assert config.teacher.revision == "ce412688b6181fc2fe2e51a3e86c60f08c08b7dc"
    assert task.sampling_dataset.revision == (
        "92b67015a1e84cb596dadcdc09462006c83c8a9a"
    )
    assert task.evaluation_windows == {
        "livecodebench_v5": ("2024-08-01", "2025-01-31"),
        "livecodebench_v6": ("2025-02-01", "2025-05-01"),
    }
    assert config.template.system.text == LCB_SYSTEM_PROMPT
    assert config.template.chat_template_kwargs == {"enable_thinking": True}
    for arm in config.evaluation.arms.values():
        assert arm.generation.samples_per_prompt == 8
        assert arm.generation.max_new_tokens == 32768
        assert arm.generation.stop_token_ids == (151645, 151643)


def test_klear_code_windows_are_disjoint() -> None:
    task = get_task("klear_code")
    assert task.include_evaluation_row(
        {"contest_date": "2025-01-31T00:00:00"}, "livecodebench_v5"
    )
    assert not task.include_evaluation_row(
        {"contest_date": "2025-02-01T00:00:00"}, "livecodebench_v5"
    )
    assert task.include_evaluation_row(
        {"contest_date": "2025-02-01T00:00:00"}, "livecodebench_v6"
    )


def test_mimo_code_sampling_fails_closed_without_unpublished_data() -> None:
    task = MimoCodeTask()
    with pytest.raises(RuntimeError, match="fixed 130K.*unpublished"):
        task.get_sampling_datasets()
    with pytest.raises(RuntimeError, match="no published source row contract"):
        task.normalize_sampling_row({})


def test_mellum_code_sampling_fails_closed_without_fixed_rlvr_mix() -> None:
    task = MellumCodeTask()
    with pytest.raises(RuntimeError, match="unpublished JetBrains code collection"):
        task.get_sampling_datasets()
    with pytest.raises(RuntimeError, match="no published row contract"):
        task.normalize_sampling_row({})


def test_olmo_code_contract_pins_v3_data_and_official_avg10() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root
        / "configs"
        / "code/olmo3-7b.yaml"
    )
    task = get_task("olmo_code")
    assert config.student.revision == "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    assert config.teacher.revision == "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
    assert config.template.system.mode == "model_default"
    assert config.template.system.text is None
    assert config.evaluation.template is not None
    assert config.evaluation.template.system.mode == "text"
    assert config.evaluation.template.system.text == LCB_SYSTEM_PROMPT
    assert tuple(task.evaluation_datasets) == ("livecodebench_v3",)
    assert len(task.get_evaluation_datasets("livecodebench_v3")) == 3
    normalized = task.normalize_evaluation_row(
        {
            "question_content": "Add two integers.",
            "starter_code": "",
            "question_id": "q1",
            "public_test_cases": "[]",
            "private_test_cases": "eJyLjjbQMzBWio0FAA4ZAqI=",
            "metadata": "{}",
        },
        "livecodebench_v3",
    )
    assert normalized["problem"] == olmo3_livecodebench_user_prompt(
        "Add two integers."
    )
    assert all(
        source.revision == "1a8d79b1012d77e4bb1fd7f699c5c7eb33ef931e"
        for source in task.get_sampling_datasets()
    )
    for arm in config.evaluation.arms.values():
        assert arm.generation.temperature == 0.6
        assert arm.generation.top_p == 0.95
        assert arm.generation.max_new_tokens == 32768
        assert arm.generation.samples_per_prompt == 10
        assert arm.generation.stop_token_ids == (100265, 100257)
        assert arm.generation.stop_strings == ()


def test_olmo_code_sampling_uses_open_instruct_extractor_and_grader(
    tmp_path: Path,
) -> None:
    task = get_task("olmo_code")
    row = task.normalize_sampling_row(
        {
            "id": "question-1",
            "dataset": ["code"],
            "prompt": "user: Implement add(a, b).",
            "input_ids_prompt": [1, 2, 3],
            "ground_truth": [json.dumps(["assert add(2, 3) == 5"])],
        }
    )
    assert row["source_prompt_ids"] == [1, 2, 3]
    runner = _fake_runner(
        tmp_path / "runner.py",
        grader_expression="request['grader_requested']",
    )
    results = task.grade_sampling_batch(
        ["print('ok')"],
        [row],
        verifier_command=runner,
        verifier_timeout_seconds=10,
    )
    # The fake runner only accepts exactly print('ok'). This proves the
    # Open-Instruct no-fence fallback sent the full output instead of the
    # LiveCodeBench empty extraction.
    assert results[0].correct is True
    assert results[0].grader == PYTHON_ASSERTIONS_GRADER_ID


def test_nemotron_code_sampling_fails_closed_without_unpublished_data() -> None:
    task = NemotronCodeTask()
    with pytest.raises(RuntimeError, match="9.8K CodeRL.*unpublished"):
        task.get_sampling_datasets()
    with pytest.raises(RuntimeError, match="no published source row contract"):
        task.normalize_sampling_row({})


@pytest.mark.parametrize(
    ("task_name", "config_name"),
    [
        (
            "nemotron_klear_proxy_code",
            "code/nemotron-cascade-8b.yaml",
        ),
        
        (
            "mellum_klear_proxy_code",
            "code/mellum2-12b-a2p5b.yaml",
        ),
    ],
)
def test_missing_code_data_pairs_use_explicit_pinned_klear_proxy(
    task_name: str, config_name: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / config_name)
    task = get_task(task_name)
    sources = task.get_sampling_datasets()
    assert config.task == task_name
    assert config.data.sampling.num_prompts == 8192
    assert len(sources) == 1
    assert sources[0].repo == "Kwai-Klear/KlearReasoner-CodeSub-15K"
    assert sources[0].filename == "train_code_rllm_clean.json"
    assert sources[0].revision == (
        "92b67015a1e84cb596dadcdc09462006c83c8a9a"
    )
    assert "proxy" in (config.sampling.generation.provenance or "").lower()
    assert "not " in (config.sampling.generation.provenance or "").lower()
    assert config.sampling.generation.samples_per_prompt == 4
    assert config.distillation.max_trajectories_per_question == 2

    source_row = {
        "hash": "proxy-question",
        "prompt": [{"role": "user", "content": "Add two integers."}],
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(
                {"inputs": ["2 3\n"], "outputs": ["5\n"]}
            ),
        },
    }
    assert task.normalize_sampling_row(source_row) == get_task(
        "klear_code"
    ).normalize_sampling_row(source_row)


def test_areal_prompt_decoder_preserves_published_user_content_exactly() -> None:
    user_content = f"{LCB_SYSTEM_PROMPT}\n\nExact prompt.\n\n"
    raw = f"<｜User｜>{user_content}<｜Assistant｜><think>\n"
    assert areal_qwen3_user_content(raw) == user_content
    with pytest.raises(ValueError, match="unknown raw prompt template"):
        areal_qwen3_user_content("<|im_start|>user\nnot-the-release-format")


def test_areal_extractor_matches_v03_last_valid_fence_contract() -> None:
    completion = (
        "draft ```python\nx = 1\n```\n"
        "final ```cpp\n#include <iostream>\nint main() { return 0; }\n```"
    )
    assert extract_areal_python_code(completion) == (
        "#include <iostream>\nint main() { return 0; }"
    )
    assert extract_areal_python_code("```python\nprint(1)\n```") == ""


def test_areal_code_contract_pins_models_data_template_and_generation() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root
        / "configs"
        / "code/areal-boba2-8b.yaml"
    )
    task = get_task("areal_code")
    assert config.student.revision == "b968826d9c46dd6066d109eabc6255188de91218"
    assert config.teacher.revision == "0a2d6b38229e58fac8213d3ce24789f796693858"
    assert task.sampling_dataset.revision == AREAL_DATASET_REVISION
    eval_sources = task.get_evaluation_datasets("livecodebench_v5_2410_2502")
    assert len(eval_sources) == 1
    assert eval_sources[0].revision == AREAL_DATASET_REVISION
    assert eval_sources[0].filename == (
        "code_benchmark/lcb_v5_2410_2502/test.jsonl"
    )
    codecontests_sources = task.get_evaluation_datasets("codecontests")
    assert len(codecontests_sources) == 1
    assert codecontests_sources[0].revision == AREAL_DATASET_REVISION
    assert codecontests_sources[0].filename == (
        "code_benchmark/code_contest_all/test.jsonl"
    )
    assert config.template.source == "teacher"
    assert config.template.system.mode == "model_default"
    assert config.template.user.suffix == "\n/think"
    assert config.template.assistant_prefix == "<think>"
    assert config.template.chat_template_kwargs == {"enable_thinking": True}
    assert config.data.sampling.num_prompts == 7500
    for arm in config.evaluation.arms.values():
        assert arm.generation.temperature == 1.0
        assert arm.generation.top_p == 0.95
        assert arm.generation.top_k == -1
        assert arm.generation.samples_per_prompt == 8
        assert arm.generation.max_new_tokens == 32768
        assert arm.generation.stop_token_ids == (151645, 151643)
        assert arm.generation.stop_strings == (
            "</s>",
            "<|im_end|>",
            "<|endoftext|>",
        )


def test_areal_normalization_uses_raw_training_prompt_and_exact_eval_file() -> None:
    task = get_task("areal_code")
    user_content = f"{LCB_SYSTEM_PROMPT}\n\nExact released spacing.\n\n"
    sampling = task.normalize_sampling_row(
        {
            "prompt": f"<｜User｜>{user_content}<｜Assistant｜><think>\n",
            "query_id": "deepcoder_lcb_example",
            "input_output": json.dumps(
                {"inputs": ["2 3\n"], "outputs": ["5\n"]}
            ),
        }
    )
    assert sampling["problem"] == user_content
    assert sampling["question_id"] == "deepcoder_lcb_example"

    question = f"{LCB_SYSTEM_PROMPT}\n\nOfficial evaluation question."
    evaluation = task.normalize_evaluation_row(
        {
            "question": question,
            "input_output": json.dumps(
                {"inputs": [""], "outputs": ["ok\n"]}
            ),
        },
        "livecodebench_v5_2410_2502",
    )
    assert evaluation["problem"] == question
    assert evaluation["question_id"] == hashlib.sha256(
        question.encode("utf-8")
    ).hexdigest()
    parsed = task.parse_completion(
        "```python\nprint('long enough for AReaL')\n```"
    )
    assert parsed.extractor == (
        f"areal@{AREAL_UPSTREAM_COMMIT}:last-valid-fenced-block"
    )


def test_areal_codecontests_normalization_uses_published_question_and_tests() -> None:
    task = get_task("areal_code")
    question = "Determine whether the string is square."
    normalized = task.normalize_evaluation_row(
        {
            "question": question,
            "source": 2,
            "input_output": json.dumps(
                {
                    "inputs": ["2\naa\naba\n"],
                    "outputs": ["YES\nNO\n"],
                }
            ),
        },
        "codecontests",
    )
    assert normalized["dataset"] == "codecontests"
    assert normalized["problem"] == question
    assert normalized["question_id"] == hashlib.sha256(
        question.encode("utf-8")
    ).hexdigest()
    assert normalized["verifier_protocol"] == "livecodebench"
    assert normalized["verifier_payload"] == {
        "input_output": {
            "inputs": ["2\naa\naba\n"],
            "outputs": ["YES\nNO\n"],
            "fn_name": None,
        }
    }


def test_areal_14b_code_contract_reuses_exact_open_data_and_eval() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root
        / "configs"
        / "code/areal-boba2-14b.yaml"
    )
    assert config.student.checkpoint == "Qwen/Qwen3-14B"
    assert config.student.revision == "40c069824f4251a91eefaf281ebe4c544efd3e18"
    assert config.teacher.checkpoint == "inclusionAI/AReaL-boba-2-14B-Open"
    assert config.teacher.revision == "e9a5e3785c29014dda82bbb65d271c0f260e9ea9"
    assert config.student.context_length == 40960
    assert config.teacher.context_length == 40960
    assert tuple(config.data.evaluation) == ("livecodebench_v5_2410_2502",)
    assert config.template.user.suffix == "\n/think"
    assert config.template.assistant_prefix == "<think>"
    assert config.data.sampling.num_prompts == 7500
    for arm in config.evaluation.arms.values():
        assert arm.generation.temperature == 1.0
        assert arm.generation.top_p == 0.95
        assert arm.generation.top_k == -1
        assert arm.generation.samples_per_prompt == 8
        assert arm.generation.max_new_tokens == 32768


def _fake_runner(
    path: Path,
    *,
    backend: str = "slurm_pyxis",
    grader_expression: str = "request['grader_requested']",
) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys
requests = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines()]
payloads = [json.loads(line) for line in pathlib.Path(sys.argv[1]).with_name('payloads.jsonl').read_text().splitlines()]
with pathlib.Path(sys.argv[2]).open('w') as handle:
    for request in requests:
        response = {
            'schema': 'soft-prompt-code-verifier-response-v1',
            'request_id': request['request_id'],
            'correct': request['code'].strip() == "print('ok')",
            'grader': GRADER_VALUE,
            'metadata': {
                'outcome': 'fake-parity',
                'payload_rows': len(payloads),
                'request_has_inline_payload': 'payload' in request,
            },
            'isolation': {'backend': 'BACKEND', 'image': 'sha256:test-image'},
        }
        handle.write(json.dumps(response) + '\\n')
""".replace("BACKEND", backend).replace("GRADER_VALUE", grader_expression),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_executable_task_grades_only_through_batch_runner(tmp_path: Path) -> None:
    task = get_task("klear_code")
    row = task.normalize_sampling_row(
        {
            "hash": "question-1",
            "prompt": [{"role": "user", "content": "Solve it."}],
            "reward_model": {
                "ground_truth": json.dumps(
                    {"inputs": [""], "outputs": ["ok\n"]}
                )
            },
        }
    )
    with pytest.raises(RuntimeError, match="code-verifier-command"):
        task.grade_sampling_batch(
            ["```python\nprint('ok')\n```"],
            [row],
            verifier_timeout_seconds=10,
        )

    results = task.grade_sampling_batch(
        ["```python\nprint('ok')\n```", "no fenced code"],
        [row, row],
        verifier_command=_fake_runner(tmp_path / "runner.py"),
        verifier_timeout_seconds=10,
    )
    assert [result.correct for result in results] == [True, False]
    assert all(result.grader == LCB_GRADER_ID for result in results)
    assert results[0].metadata["isolation"] == {
        "backend": "slurm_pyxis",
        "image": "sha256:test-image",
    }


def test_external_verifier_rejects_non_isolated_receipt(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsupported isolation backend"):
        run_external_code_verifier(
            [
                {
                    "protocol": "livecodebench",
                    "code": "print('ok')",
                    "payload": {"input_output": {}},
                    "grader_requested": LCB_GRADER_ID,
                }
            ],
            command=_fake_runner(tmp_path / "runner.py", backend="process"),
            timeout_seconds=10,
        )


def test_external_verifier_deduplicates_payload_sidecar(tmp_path: Path) -> None:
    payload = {"input_output": {"inputs": [""], "outputs": ["ok\n"]}}
    results = run_external_code_verifier(
        [
            {
                "protocol": "livecodebench",
                "code": "print('ok')",
                "payload": payload,
                "payload_identity": "same-question",
                "grader_requested": LCB_GRADER_ID,
            },
            {
                "protocol": "livecodebench",
                "code": "print('ok')",
                "payload": payload,
                "payload_identity": "same-question",
                "grader_requested": LCB_GRADER_ID,
            },
        ],
        command=_fake_runner(tmp_path / "runner.py"),
        timeout_seconds=10,
    )
    assert [result.metadata["payload_rows"] for result in results] == [1, 1]
    assert not any(
        result.metadata["request_has_inline_payload"] for result in results
    )


def test_external_verifier_rejects_grader_substitution(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="grader does not match"):
        run_external_code_verifier(
            [
                {
                    "protocol": "livecodebench",
                    "code": "print('ok')",
                    "payload": {"input_output": {}},
                    "grader_requested": LCB_GRADER_ID,
                }
            ],
            command=_fake_runner(
                tmp_path / "runner.py", grader_expression="'wrong-grader'"
            ),
            timeout_seconds=10,
        )


def test_external_verifier_preserves_opt_in_failure_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    runner = tmp_path / "failing-runner.py"
    runner.write_text("#!/usr/bin/env python3\nraise SystemExit(9)\n", encoding="utf-8")
    runner.chmod(0o755)
    failure_root = tmp_path / "failures"
    monkeypatch.setenv(
        "SOFT_PROMPT_CODE_VERIFIER_FAILURE_DIR", str(failure_root)
    )
    with pytest.raises(RuntimeError, match="request bundle:"):
        run_external_code_verifier(
            [
                {
                    "protocol": "livecodebench",
                    "code": "print('preserve me')",
                    "payload": {"input_output": {"inputs": [""], "outputs": [""]}},
                    "grader_requested": LCB_GRADER_ID,
                }
            ],
            command=runner,
            timeout_seconds=10,
        )
    bundles = list(failure_root.glob("failure-*"))
    assert len(bundles) == 1
    requests = (bundles[0] / "requests.jsonl").read_text(encoding="utf-8")
    assert "print('preserve me')" in requests
    assert (bundles[0] / "payloads.jsonl").is_file()
    failure = json.loads((bundles[0] / "failure.json").read_text(encoding="utf-8"))
    assert failure["failure_kind"] == "nonzero_exit"
    assert failure["returncode"] == 9


def test_response_schema_constant_is_stable() -> None:
    assert RESPONSE_SCHEMA == "soft-prompt-code-verifier-response-v1"


def test_verifier_worker_reserves_two_cpus_per_candidate(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
    assert _worker_count(100) == 16
    assert _worker_count(3) == 3
    monkeypatch.setenv("SOFT_PROMPT_VERIFIER_WORKERS", "7")
    assert _worker_count(100) == 7


def test_verifier_metadata_is_json_safe() -> None:
    class Scalar:
        def item(self):
            return 4

    assert _json_safe({"nested": (Scalar(), {1: Scalar()})}) == {
        "nested": [4, {"1": 4}]
    }


def test_worker_decodes_raw_livecodebench_payload_once_per_group() -> None:
    private = json.dumps([{"input": "2 3\n", "output": "5\n"}])
    encoded = base64.b64encode(zlib.compress(pickle.dumps(private))).decode()
    prepared = _prepare_livecodebench_payload(
        {
            "livecodebench_raw": {
                "public_test_cases": json.dumps(
                    [{"input": "1 1\n", "output": "2\n"}]
                ),
                "private_test_cases": encoded,
                "metadata": json.dumps({"func_name": None}),
            }
        }
    )
    assert prepared == {
        "input_output": {
            "inputs": ["1 1\n", "2 3\n"],
            "outputs": ["2\n", "5\n"],
            "fn_name": None,
        }
    }


def test_safe_lcb_check_reaps_child_with_lingering_thread(monkeypatch) -> None:
    import threading
    import time

    def send_then_linger(_sample, _code, _timeout, connection):
        threading.Thread(target=time.sleep, args=(30,), daemon=False).start()
        connection.send(("ok", [False], {"verdict": "already computed"}))
        connection.close()

    monkeypatch.setattr(
        "soft_prompt_distillation.code_verifier_worker._run_livecodebench_candidate",
        send_then_linger,
    )
    started = time.monotonic()
    results, metadata = _safe_livecodebench_check(
        {"input_output": json.dumps({"inputs": ["1"], "outputs": ["1"]})},
        "unused", 1,
    )
    assert time.monotonic() - started < 5
    assert results == [False]
    assert metadata == {"verdict": "already computed"}


