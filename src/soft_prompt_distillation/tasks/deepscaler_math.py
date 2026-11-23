"""DeepScaleR math dataset normalization and grading."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .base import CompletionParse, DatasetSource, TaskAdapter
from .math_grader import extract_answer, grade_answer_mathd, grade_answer_sympy
from .official_math_grader import SWALLOW_MATH_GRADER, grade_official_math


_DEEPSCALER_INSTRUCTION = (
    "Let's think step by step and output the final answer within \\boxed{}."
)


def _identity(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _answer_text(value: Any) -> str:
    """Keep integer-valued Parquet labels equivalent to JSON integer labels."""

    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("evaluation answer list must contain exactly one value")
        value = value[0]
    if isinstance(value, dict):
        if "ground_truth" not in value:
            raise ValueError("evaluation answer mapping has no ground_truth")
        value = value["ground_truth"]
    if value is None or isinstance(value, bool) or not isinstance(
        value, (str, int, float)
    ):
        raise ValueError("evaluation answer must be scalar text or numeric")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _problem_from_messages(value: Any, dataset: str) -> str:
    if not isinstance(value, list):
        raise ValueError(f"cannot extract problem from {dataset} row")
    user_contents = [
        message["content"]
        for message in value
        if isinstance(message, dict)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"]
    ]
    if len(user_contents) != 1:
        raise ValueError(
            f"{dataset} prompt must contain exactly one nonempty user message"
        )
    return user_contents[0]


def _strip_embedded_instruction(problem: str) -> str:
    """Remove only the exact instruction added by the AMC23 HF conversion."""

    suffix = f" {_DEEPSCALER_INSTRUCTION}"
    return problem[: -len(suffix)] if problem.endswith(suffix) else problem


def _parse_deepscaler_completion(completion: str) -> CompletionParse:
    # Only grade the final response, never a boxed expression from reasoning.
    if "</think>" in completion:
        reasoning, _separator, model_solution = completion.rpartition("</think>")
        extractor = "deepscaler_last_boxed_after_think"
    elif "###Response" in completion:
        reasoning, _separator, model_solution = completion.partition("###Response")
        extractor = "deepscaler_last_boxed_after_response_marker"
    else:
        return CompletionParse(
            extractor="deepscaler_missing_response_boundary",
            reasoning_text=None,
            final_text=completion,
            extracted_answer=None,
        )
    return CompletionParse(
        extractor=extractor,
        reasoning_text=reasoning,
        final_text=model_solution,
        extracted_answer=extract_answer(model_solution),
    )


def _grade_deepscaler_answer(completion: str, answer: str) -> bool:
    """Slime's DeepScaleR response wrapper, kept local for reproducibility."""

    # The official reward requires an explicit boxed final answer.
    model_answer = _parse_deepscaler_completion(completion).extracted_answer
    if model_answer is None or answer == "":
        return False

    # Labels may already contain \boxed{...}; normalize both dataset schemas to
    # the answer inside the box before applying the two official checkers.
    ground_truth = str(answer)
    if "\\boxed" in ground_truth:
        ground_truth = extract_answer(ground_truth)
        if ground_truth is None:
            return False

    # mathd covers strict normalized matches; SymPy covers the additional
    # expression-equivalence cases allowed by the official DeepScaleR reward.
    return grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(
        model_answer, ground_truth
    )


class DeepScalerMathTask(TaskAdapter):
    sampling_dataset = DatasetSource(
        repo="agentica-org/DeepScaleR-Preview-Dataset",
        filename="deepscaler.json",
        split="train",
        domain="math",
    )
    evaluation_datasets: Mapping[str, DatasetSource] = {
        "aime_2024": DatasetSource(
            repo="zhuzilin/aime-2024",
            filename="aime-2024.jsonl",
            split="test",
            domain="math",
        ),
        "aime_2025": DatasetSource(
            repo="zhuzilin/aime-2025",
            filename="aime-2025.jsonl",
            split="test",
            domain="math",
        ),
        "aime_2025_i": DatasetSource(
            repo="opencompass/AIME2025",
            filename="aime2025-I.jsonl",
            split="test",
            domain="math",
        ),
        "aime_2026": DatasetSource(
            repo="math-ai/aime26",
            filename="aime2026.jsonl",
            split="test",
            domain="math",
        ),
        "math_500": DatasetSource(
            repo="math-ai/math500",
            filename="test.jsonl",
            split="test",
            domain="math",
        ),
        "gsm8k": DatasetSource(
            repo="openai/gsm8k",
            filename="main/test-00000-of-00001.parquet",
            split="test",
            domain="math",
        ),
        "amc_2022_2023": DatasetSource(
            repo="rawsh/amc23",
            filename="amc2023.parquet",
            split="test",
            domain="math",
        ),
        "minerva_math": DatasetSource(
            repo="knoveleng/Minerva-Math",
            filename="data/train-00000-of-00001.parquet",
            split="test",
            domain="math",
        ),
        "olympiad_bench": DatasetSource(
            repo="knoveleng/OlympiadBench",
            filename="data/train-00000-of-00001.parquet",
            split="test",
            domain="math",
        ),
        "hmmt_feb_2025": DatasetSource(
            repo="MathArena/hmmt_feb_2025",
            filename="data/train-00000-of-00001.parquet",
            split="train",
            domain="math",
        ),
    }

    def normalize_sampling_row(self, row: dict[str, Any]) -> dict[str, Any]:
        problem = row.get("problem")
        if not isinstance(problem, str) or not problem:
            raise ValueError("DeepScaleR sampling row has no problem")
        return {
            "problem": problem,
            "answer": str(row.get("answer", "")),
            "question_id": str(row.get("id", _identity(problem))),
        }

    def normalize_evaluation_row(
        self, row: dict[str, Any], dataset: str
    ) -> dict[str, Any]:
        if isinstance(row.get("problem"), str):
            problem = row["problem"]
        elif isinstance(row.get("question"), str):
            problem = row["question"]
        elif isinstance(row.get("prompt"), list):
            problem = _problem_from_messages(row["prompt"], dataset)
        else:
            raise ValueError(f"cannot extract problem from {dataset} row")
        if dataset == "amc_2022_2023":
            problem = _strip_embedded_instruction(problem)
        answer = row.get("answer")
        if answer is None:
            answer = row.get("label")
        if answer is None:
            answer = row.get("solution")
        if answer is None:
            answer = row.get("gold_answer")
        if answer is None:
            answer = row.get("final_answer")
        if answer is None:
            answer = row.get("clean_answer")
        reward_model = row.get("reward_model")
        if answer is None and isinstance(reward_model, dict):
            answer = reward_model.get("ground_truth")
        if answer is None:
            raise ValueError(f"cannot extract answer from {dataset} row")
        if dataset == "gsm8k":
            if not isinstance(answer, str) or "####" not in answer:
                raise ValueError("GSM8K answer has no published #### final marker")
            _reasoning, _marker, answer = answer.rpartition("####")
            answer = answer.strip().replace(",", "")
            if not answer:
                raise ValueError("GSM8K answer has an empty published final value")
        extra_info = row.get("extra_info")
        question_id = row.get("unique_id")
        if question_id is None:
            question_id = row.get("id")
        if question_id is None:
            question_id = row.get("idx")
        if question_id is None:
            question_id = row.get("problem_idx")
        if question_id is None and isinstance(extra_info, dict):
            question_id = extra_info.get("index")
        if question_id is None:
            question_id = _identity(problem)
        answer_text = _answer_text(answer)
        reference = answer_text
        reference_format = "expression"
        # Swallow's MATH-500 task grades against the published worked solution
        # with its LaTeX-gold extractor.  Retaining that source field is
        # necessary: parsing only the already-extracted answer is not equivalent
        # for sets, equations, matrices, units, and textual answers.
        if dataset == "math_500" and isinstance(row.get("solution"), str):
            reference = row["solution"]
            reference_format = "latex"
        elif "\\boxed" in reference:
            reference_format = "latex"
        return {
            "dataset": dataset,
            "problem": problem,
            "answer": answer_text,
            "question_id": str(question_id),
            "evaluation_reference": reference,
            "evaluation_reference_format": reference_format,
        }

    def grade(self, completion: str, answer: str) -> bool:
        return _grade_deepscaler_answer(completion, answer)

    def grade_evaluation(
        self, completion: str, row: Mapping[str, Any]
    ) -> bool:
        reference = row.get("evaluation_reference")
        reference_format = row.get("evaluation_reference_format")
        if not isinstance(reference, str) or not isinstance(reference_format, str):
            raise ValueError("normalized math evaluation row lacks verifier reference")
        return grade_official_math(
            self.parse_completion(completion).final_text,
            reference,
            reference_format,
        )

    def evaluation_grader(self, row: Mapping[str, Any]) -> str:
        del row
        return SWALLOW_MATH_GRADER

    def parse_completion(self, completion: str) -> CompletionParse:
        return _parse_deepscaler_completion(completion)


__all__ = ["DeepScalerMathTask"]
