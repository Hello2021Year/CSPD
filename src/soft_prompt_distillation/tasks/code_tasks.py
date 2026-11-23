"""Executable-code task adapters with official LiveCodeBench prompt semantics."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..code_verifier import (
    LCB_GRADER_ID,
    LCB_UPSTREAM_COMMIT,
    PYTHON_ASSERTIONS_GRADER_ID,
    run_external_code_verifier,
)
from .base import CompletionParse, DatasetSource, GradeResult, TaskAdapter
from .klear_contamination import KLEAR_LCB_V1_V6_NEAR_DUPLICATE_IDS


LCB_SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
LCB_DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_RELEASE_V5_REVISION = "0687ab61843a90a0cc864a2b67db729861cd0ae5"
OPEN_INSTRUCT_UPSTREAM_COMMIT = "e6c3044aa6201c8bfa4573c1b03142af599e730d"
OLMES_UPSTREAM_COMMIT = "7bebf0eb148cd89d1d8a8096505f99d4fc48c209"
AREAL_UPSTREAM_COMMIT = "e7eda163ba63270dbce99a919d70bf3454c626a8"
AREAL_DATASET_REVISION = "1abdc2c4884eb6c4411c0d5c5e42da5e2be817de"
LCB_STARTER_FORMAT = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
LCB_STDIN_FORMAT = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads "
    "the inputs, runs the algorithm and writes output to STDOUT."
)


def livecodebench_user_prompt(question: str, starter_code: str = "") -> str:
    """Match upstream ``get_generic_question_template_answer`` byte-for-byte."""

    if not isinstance(question, str) or not question:
        raise ValueError("LiveCodeBench question_content must be non-empty")
    prompt = f"### Question:\n{question}\n\n"
    if starter_code:
        prompt += f"### Format: {LCB_STARTER_FORMAT}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_STDIN_FORMAT}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def olmo3_livecodebench_user_prompt(question: str, starter_code: str = "") -> str:
    """Match OLMES' ``livecodebench_codegeneration::olmo3:midtrain`` prompt."""

    if not isinstance(question, str) or not question:
        raise ValueError("LiveCodeBench question_content must be non-empty")
    if starter_code:
        format_instruction = (
            f"{LCB_STARTER_FORMAT}\n```python\n{starter_code}\n```"
        )
    else:
        format_instruction = (
            f"{LCB_STDIN_FORMAT}\n```python\n# YOUR CODE HERE\n```"
        )
    return (
        f"### Question:\n{question}\n\n"
        "### Format:\n"
        "Provide CONCISE reasoning on how to arrive at the answer in the "
        "<think> </think> tag.\n"
        f"{format_instruction}\n\n"
        "### Answer: (use the provided format with backticks)\n\n"
    )


def extract_livecodebench_code(model_output: str) -> str:
    """Match LiveCodeBench's non-base-model fenced-code extraction."""

    output_lines = model_output.split("\n")
    delimiters = [
        index for index, line in enumerate(output_lines) if "```" in line
    ]
    if len(delimiters) < 2:
        return ""
    return "\n".join(output_lines[delimiters[-2] + 1 : delimiters[-1]])


def extract_open_instruct_python_code(model_output: str) -> str:
    """Match Open Instruct ``CodeVerifier.extract_python_code`` exactly."""

    matches = re.findall(r"```(?:python)?(.*?)```", model_output, re.DOTALL)
    return matches[-1].strip() if matches else model_output


def extract_areal_python_code(model_output: str) -> str:
    """Match AReaL v0.3 code evaluation extraction exactly.

    AReaL accepts Python, Py, C++, or unlabelled fences, discards blocks under
    20 stripped characters, and evaluates the last remaining block.  Its
    published evaluation leaves syntax checking disabled.
    """

    pattern = r"(?i)```(?:python|py|cpp|CPP)?\s*\n?(.*?)\n?```"
    blocks = re.findall(pattern, model_output, re.DOTALL)
    valid = [block.strip() for block in blocks if len(block.strip()) >= 20]
    return valid[-1] if valid else ""


def areal_qwen3_user_content(prompt: str) -> str:
    """Recover the exact user content from AReaL's published raw prompt."""

    user_marker = "<｜User｜>"
    assistant_marker = "<｜Assistant｜><think>\n"
    if not prompt.startswith(user_marker) or not prompt.endswith(assistant_marker):
        raise ValueError("AReaL row has an unknown raw prompt template")
    if prompt.count(user_marker) != 1 or prompt.count("<｜Assistant｜>") != 1:
        raise ValueError("AReaL row must contain exactly one user/assistant turn")
    content = prompt[len(user_marker) : -len(assistant_marker)]
    if not content:
        raise ValueError("AReaL row has empty user content")
    return content


def _lcb_verifier_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        key: row.get(key)
        for key in ("public_test_cases", "private_test_cases", "metadata")
    }
    if not all(isinstance(value, str) and value for value in values.values()):
        raise ValueError("LiveCodeBench row has invalid raw test fields")
    return {
        "livecodebench_raw": values,
    }


def _io_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("code test payload is not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("code test payload must be an object")
    inputs = value.get("inputs")
    outputs = value.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("code test payload must contain input/output lists")
    if len(inputs) != len(outputs) or not inputs:
        raise ValueError("code test input/output lists must be non-empty and aligned")
    if not all(isinstance(item, (str, list)) for item in inputs):
        raise ValueError("code test inputs must be strings or call-argument lists")
    return {
        "input_output": {
            "inputs": inputs,
            "outputs": outputs,
            "fn_name": value.get("fn_name"),
        }
    }


def _normalized_code_row(
    *,
    dataset: str | None,
    problem: str,
    question_id: str,
    protocol: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not problem or not question_id:
        raise ValueError("normalized code row requires problem and question_id")
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    normalized = {
        "problem": problem,
        "answer": serialized,
        "question_id": question_id,
        "verifier_protocol": protocol,
        "verifier_payload": dict(payload),
    }
    if dataset is not None:
        normalized["dataset"] = dataset
    return normalized


LCB_V5_SOURCES = tuple(
    DatasetSource(
        repo="livecodebench/code_generation_lite",
        filename=("test.jsonl" if version == 1 else f"test{version}.jsonl"),
        split="test",
        domain="code",
        revision=LCB_DATASET_REVISION,
    )
    for version in range(1, 6)
)
LCB_RELEASE_V5_SOURCES = tuple(
    DatasetSource(
        repo="livecodebench/code_generation_lite",
        filename=("test.jsonl" if version == 1 else f"test{version}.jsonl"),
        split="test",
        domain="code",
        revision=LCB_RELEASE_V5_REVISION,
    )
    for version in range(1, 6)
)
LCB_RELEASE_V5_DELTA_SOURCES = LCB_RELEASE_V5_SOURCES[-1:]
LCB_V3_SOURCES = LCB_V5_SOURCES[:3]
LCB_V6_SOURCES = LCB_V5_SOURCES + (
    DatasetSource(
        repo="livecodebench/code_generation_lite",
        filename="test6.jsonl",
        split="test",
        domain="code",
        revision=LCB_DATASET_REVISION,
    ),
)
LCB_V6_DELTA_SOURCES = LCB_V6_SOURCES[-1:]


class ExecutableCodeTask(TaskAdapter):
    """Base adapter that only grades through the external verifier bridge."""

    evaluation_datasets = {
        "livecodebench_v5": LCB_V5_SOURCES,
        "livecodebench_v6": LCB_V6_SOURCES,
    }
    evaluation_windows: Mapping[str, tuple[str | None, str | None]] = {}
    stream_sampling_selection = True
    requires_external_verifier = True

    def include_evaluation_row(self, row: dict[str, Any], dataset: str) -> bool:
        bounds = self.evaluation_windows.get(dataset)
        if bounds is None:
            return True
        value = row.get("contest_date")
        if not isinstance(value, str):
            raise ValueError("dated LiveCodeBench slice has no contest_date")
        date = datetime.fromisoformat(value).date().isoformat()
        start, end = bounds
        return (start is None or date >= start) and (end is None or date <= end)

    def normalize_evaluation_row(
        self, row: dict[str, Any], dataset: str
    ) -> dict[str, Any]:
        question = row.get("question_content")
        starter_code = row.get("starter_code", "")
        question_id = row.get("question_id")
        if not isinstance(question, str) or not isinstance(starter_code, str):
            raise ValueError("invalid LiveCodeBench question fields")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("invalid LiveCodeBench question_id")
        return _normalized_code_row(
            dataset=dataset,
            problem=livecodebench_user_prompt(question, starter_code),
            question_id=question_id,
            protocol="livecodebench",
            payload=_lcb_verifier_payload(row),
        )

    def parse_completion(self, completion: str) -> CompletionParse:
        code = extract_livecodebench_code(completion)
        return CompletionParse(
            extractor=f"livecodebench@{LCB_UPSTREAM_COMMIT}:last_fenced_block",
            reasoning_text=None,
            final_text=code,
            extracted_answer=code or None,
        )

    def grade(self, completion: str, answer: str) -> bool:
        del completion, answer
        raise RuntimeError(
            "executable code must be graded through the isolated batch verifier"
        )

    def _grade_batch(
        self,
        completions: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
        *,
        verifier_command: Path | None,
        verifier_timeout_seconds: int | None,
    ) -> list[GradeResult]:
        if len(completions) != len(rows):
            raise ValueError("code completions and rows have different lengths")
        requests = []
        for completion, row in zip(completions, rows, strict=True):
            protocol = row.get("verifier_protocol")
            payload = row.get("verifier_payload")
            if not isinstance(protocol, str) or not isinstance(payload, Mapping):
                raise ValueError("normalized code row has no verifier contract")
            if protocol == "livecodebench":
                code = self.extract_livecodebench_completion(completion)
                grader_requested = LCB_GRADER_ID
            elif protocol == "python_assertions":
                code = extract_open_instruct_python_code(completion)
                grader_requested = PYTHON_ASSERTIONS_GRADER_ID
            else:
                raise ValueError(f"unsupported executable-code protocol: {protocol!r}")
            requests.append(
                {
                    "protocol": protocol,
                    "code": code,
                    "payload": dict(payload),
                    "payload_identity": json.dumps(
                        [protocol, row.get("dataset"), row.get("question_id")],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "grader_requested": grader_requested,
                }
            )
        return run_external_code_verifier(
            requests,
            command=verifier_command,
            timeout_seconds=verifier_timeout_seconds,
        )

    def extract_livecodebench_completion(self, completion: str) -> str:
        return extract_livecodebench_code(completion)

    def grade_sampling_batch(
        self,
        completions: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
        *,
        verifier_command: Path | None = None,
        verifier_timeout_seconds: int | None = None,
    ) -> list[GradeResult]:
        return self._grade_batch(
            completions,
            rows,
            verifier_command=verifier_command,
            verifier_timeout_seconds=verifier_timeout_seconds,
        )

    def grade_evaluation_batch(
        self,
        completions: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
        *,
        verifier_command: Path | None = None,
        verifier_timeout_seconds: int | None = None,
    ) -> list[GradeResult]:
        return self._grade_batch(
            completions,
            rows,
            verifier_command=verifier_command,
            verifier_timeout_seconds=verifier_timeout_seconds,
        )

    def evaluation_grader(self, row: Mapping[str, Any]) -> str:
        del row
        return LCB_GRADER_ID


class KlearCodeTask(ExecutableCodeTask):
    # The Klear card reports disjoint release windows for its v5/v6 table.
    evaluation_windows = {
        "livecodebench_v5": ("2024-08-01", "2025-01-31"),
        "livecodebench_v6": ("2025-02-01", "2025-05-01"),
    }
    sampling_dataset = DatasetSource(
        repo="Kwai-Klear/KlearReasoner-CodeSub-15K",
        filename="train_code_rllm_clean.json",
        split="train",
        domain="code",
        revision="92b67015a1e84cb596dadcdc09462006c83c8a9a",
    )

    def normalize_sampling_row(self, row: dict[str, Any]) -> dict[str, Any]:
        prompt = row.get("prompt")
        reward = row.get("reward_model")
        question_id = row.get("hash")
        if (
            not isinstance(prompt, list)
            or len(prompt) != 1
            or not isinstance(prompt[0], dict)
            or prompt[0].get("role") != "user"
            or not isinstance(prompt[0].get("content"), str)
        ):
            raise ValueError("Klear code row must contain one user prompt")
        if not isinstance(reward, dict):
            raise ValueError("Klear code row has no reward_model")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Klear code row has no hash")
        ground_truth = reward.get("ground_truth")
        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except json.JSONDecodeError as exc:
                raise ValueError("Klear code ground truth is not JSON") from exc
        if not isinstance(ground_truth, dict):
            raise ValueError("Klear code ground truth must be an object")
        functional = ground_truth.get("functional")
        if functional is not None:
            if set(ground_truth) != {"functional"}:
                raise ValueError("Klear functional ground truth has mixed schemas")
            if (
                not isinstance(functional, list)
                or not functional
                or not all(isinstance(item, str) for item in functional)
            ):
                raise ValueError("Klear functional ground truth has no assertions")
            protocol = "python_assertions"
            payload: Mapping[str, Any] = {"assertions": functional}
        else:
            protocol = "livecodebench"
            payload = _io_payload(ground_truth)
        return _normalized_code_row(
            dataset=None,
            problem=prompt[0]["content"],
            question_id=question_id,
            protocol=protocol,
            payload=payload,
        )


class _KlearProxySamplingMixin:
    """Reuse the pinned Klear source while preserving the target eval contract."""

    sampling_dataset = KlearCodeTask.sampling_dataset

    def include_sampling_row(self, row: dict[str, Any]) -> bool:
        question_id = row.get("hash")
        return (
            isinstance(question_id, str)
            and question_id not in KLEAR_LCB_V1_V6_NEAR_DUPLICATE_IDS
        )

    def get_sampling_datasets(self) -> tuple[DatasetSource, ...]:
        return (self.sampling_dataset,)

    def normalize_sampling_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return KlearCodeTask.normalize_sampling_row(self, row)


class MimoCodeTask(ExecutableCodeTask):
    """MiMo code evaluation without inventing an unpublished RL prompt pool."""

    evaluation_windows = {
        "livecodebench_v5": ("2024-08-01", "2025-01-31"),
        "livecodebench_v6": ("2025-02-01", "2025-05-01"),
    }

    def get_sampling_datasets(self) -> tuple[DatasetSource, ...]:
        raise RuntimeError(
            "MiMo's fixed 130K mathematics-and-code RL prompt set is "
            "unpublished; sampling is blocked until an explicit replacement "
            "dataset is selected"
        )

    def normalize_sampling_row(self, row: dict[str, Any]) -> dict[str, Any]:
        del row
        raise RuntimeError(
            "MiMo code sampling has no published source row contract"
        )


class MellumCodeTask(ExecutableCodeTask):
    """Mellum2 SFT-to-RLVR evaluation without inventing private RL data."""

    evaluation_datasets = {
        "livecodebench_v6": LCB_V6_SOURCES,
    }

    def get_sampling_datasets(self) -> tuple[DatasetSource, ...]:
        raise RuntimeError(
            "Mellum2's 57.5K-code / 260.5K-total Instruct RLVR mix includes "
            "an unpublished JetBrains code collection and has no released "
            "fixed training artifact; sampling is blocked until an explicit "
            "replacement dataset is selected"
        )

    def normalize_sampling_row(self, row: dict[str, Any]) -> dict[str, Any]:
        del row
        raise RuntimeError("Mellum2 code sampling has no published row contract")


class OlmoCodeTask(ExecutableCodeTask):
    evaluation_datasets = {
        "livecodebench_v3": LCB_V3_SOURCES,
    }
    sampling_datasets = tuple(
        DatasetSource(
            repo="allenai/Dolci-Instruct-RL",
            filename=f"data/train-0000{index}-of-00003.parquet",
            split="train",
            domain="code",
            revision="1a8d79b1012d77e4bb1fd7f699c5c7eb33ef931e",
        )
        for index in range(3)
    )
    sampling_dataset = sampling_datasets[0]

    def normalize_evaluation_row(
        self, row: dict[str, Any], dataset: str
    ) -> dict[str, Any]:
        question = row.get("question_content")
        starter_code = row.get("starter_code", "")
        question_id = row.get("question_id")
        if not isinstance(question, str) or not isinstance(starter_code, str):
            raise ValueError("invalid LiveCodeBench question fields")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("invalid LiveCodeBench question_id")
        return _normalized_code_row(
            dataset=dataset,
            problem=olmo3_livecodebench_user_prompt(question, starter_code),
            question_id=question_id,
            protocol="livecodebench",
            payload=_lcb_verifier_payload(row),
        )

    def include_sampling_row(self, row: dict[str, Any]) -> bool:
        datasets = row.get("dataset")
        return isinstance(datasets, list) and "code" in datasets

    def normalize_sampling_row(self, row: dict[str, Any]) -> dict[str, Any]:
        problem = row.get("prompt")
        question_id = row.get("id")
        ground_truth = row.get("ground_truth")
        source_prompt_ids = row.get("input_ids_prompt")
        if isinstance(problem, str) and problem.startswith("user: "):
            problem = problem.removeprefix("user: ")
        if not isinstance(problem, str) or not problem:
            raise ValueError("Dolci code row has no prompt")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Dolci code row has no id")
        if (
            not isinstance(source_prompt_ids, list)
            or not source_prompt_ids
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int)
                for token_id in source_prompt_ids
            )
        ):
            raise ValueError("Dolci code row has no source prompt token IDs")
        if (
            not isinstance(ground_truth, list)
            or not ground_truth
            or not all(isinstance(value, str) for value in ground_truth)
        ):
            raise ValueError("Dolci code row has no assertion tests")
        assertions: list[str] = []
        for value in ground_truth:
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("Dolci code ground truth is not JSON") from exc
            if not isinstance(decoded, list) or not all(
                isinstance(item, str) for item in decoded
            ):
                raise ValueError("Dolci code ground truth is not an assertion list")
            assertions.extend(decoded)
        normalized = _normalized_code_row(
            dataset=None,
            problem=problem,
            question_id=question_id,
            protocol="python_assertions",
            payload={"assertions": assertions},
        )
        normalized["source_prompt_ids"] = source_prompt_ids
        return normalized


class NemotronCodeTask(ExecutableCodeTask):
    """Nemotron CodeRL evaluation without substituting unpublished data.

    LiveCodeBench v5 is pinned to its release snapshot.  Reusing the later v6
    repository state for the older slice backfills 44 questions and changes
    the published 279-question v5 contract into 323 questions.
    """

    evaluation_datasets = {
        "livecodebench_v5": LCB_RELEASE_V5_SOURCES,
        "livecodebench_v6": LCB_V6_SOURCES,
    }
    evaluation_windows = {
        "livecodebench_v5": ("2024-08-01", "2025-01-31"),
        "livecodebench_v6": ("2024-08-01", "2025-05-01"),
    }

    def get_sampling_datasets(self) -> tuple[DatasetSource, ...]:
        raise RuntimeError(
            "Nemotron-Cascade's approximately 9.8K CodeRL prompt set is "
            "unpublished; sampling is blocked until an explicit replacement "
            "dataset is selected"
        )

    def normalize_sampling_row(self, row: dict[str, Any]) -> dict[str, Any]:
        del row
        raise RuntimeError(
            "Nemotron-Cascade CodeRL sampling has no published source row contract"
        )


class MellumKlearProxyCodeTask(_KlearProxySamplingMixin, MellumCodeTask):
    """Mellum2 checkpoints trained on an explicit Klear data proxy."""


class NemotronKlearProxyCodeTask(_KlearProxySamplingMixin, NemotronCodeTask):
    """Nemotron CodeRL checkpoints trained on an explicit Klear data proxy."""


class ArealCodeTask(ExecutableCodeTask):
    evaluation_datasets = {
        "livecodebench_v5_2410_2502": DatasetSource(
            repo="inclusionAI/AReaL-boba-2-RL-Code",
            filename="code_benchmark/lcb_v5_2410_2502/test.jsonl",
            split="test",
            domain="code",
            revision=AREAL_DATASET_REVISION,
        ),
        "codecontests": DatasetSource(
            repo="inclusionAI/AReaL-boba-2-RL-Code",
            filename="code_benchmark/code_contest_all/test.jsonl",
            split="test",
            domain="code",
            revision=AREAL_DATASET_REVISION,
        ),
    }
    sampling_dataset = DatasetSource(
        repo="inclusionAI/AReaL-boba-2-RL-Code",
        filename="train/train.jsonl",
        split="train",
        domain="code",
        revision=AREAL_DATASET_REVISION,
    )

    def normalize_sampling_row(self, row: dict[str, Any]) -> dict[str, Any]:
        prompt = row.get("prompt")
        question_id = row.get("query_id")
        if not isinstance(prompt, str):
            raise ValueError("AReaL code row has no raw prompt")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("AReaL code row has no query_id")
        return _normalized_code_row(
            dataset=None,
            problem=areal_qwen3_user_content(prompt),
            question_id=question_id,
            protocol="livecodebench",
            payload=_io_payload(row.get("input_output")),
        )

    def normalize_evaluation_row(
        self, row: dict[str, Any], dataset: str
    ) -> dict[str, Any]:
        question = row.get("question")
        if not isinstance(question, str) or not question:
            raise ValueError("AReaL evaluation row has no question")
        question_id = hashlib.sha256(question.encode("utf-8")).hexdigest()
        return _normalized_code_row(
            dataset=dataset,
            problem=question,
            question_id=question_id,
            protocol="livecodebench",
            payload=_io_payload(row.get("input_output")),
        )

    def extract_livecodebench_completion(self, completion: str) -> str:
        return extract_areal_python_code(completion)

    def parse_completion(self, completion: str) -> CompletionParse:
        code = self.extract_livecodebench_completion(completion)
        return CompletionParse(
            extractor=f"areal@{AREAL_UPSTREAM_COMMIT}:last-valid-fenced-block",
            reasoning_text=None,
            final_text=code,
            extracted_answer=code or None,
        )


class MimoArealProxyCodeTask(MimoCodeTask):
    """AReaL training questions with unchanged MiMo evaluation and extraction."""

    sampling_dataset = ArealCodeTask.sampling_dataset

    def get_sampling_datasets(self) -> tuple[DatasetSource, ...]:
        return (self.sampling_dataset,)

    def normalize_sampling_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return ArealCodeTask.normalize_sampling_row(self, row)


__all__ = [
    "ArealCodeTask",
    "AREAL_DATASET_REVISION",
    "AREAL_UPSTREAM_COMMIT",
    "ExecutableCodeTask",
    "KlearCodeTask",
    "LCB_GRADER_ID",
    "LCB_DATASET_REVISION",
    "LCB_RELEASE_V5_REVISION",
    "LCB_RELEASE_V5_SOURCES",
    "LCB_V3_SOURCES",
    "LCB_V6_DELTA_SOURCES",
    "LCB_SYSTEM_PROMPT",
    "LCB_UPSTREAM_COMMIT",
    "MellumCodeTask",
    "MellumKlearProxyCodeTask",
    "MimoCodeTask",
    "NemotronCodeTask",
    "NemotronKlearProxyCodeTask",
    "OLMES_UPSTREAM_COMMIT",
    "OPEN_INSTRUCT_UPSTREAM_COMMIT",
    "OlmoCodeTask",
    "extract_livecodebench_code",
    "extract_areal_python_code",
    "extract_open_instruct_python_code",
    "livecodebench_user_prompt",
    "olmo3_livecodebench_user_prompt",
    "areal_qwen3_user_content",
]
