"""OLMo math task with boxed-answer grading that does not require think tags."""

from __future__ import annotations

import hashlib
from typing import Any

from .base import CompletionParse, DatasetSource
from .deepscaler_math import DeepScalerMathTask
from .math_grader import extract_answer, grade_answer_mathd, grade_answer_sympy


def _parse_boxed_math_completion(completion: str) -> CompletionParse:
    """Separate common reasoning/final wrappers, then extract the last box."""

    reasoning = None
    if "</think>" in completion:
        reasoning, _separator, solution = completion.rpartition("</think>")
        boundary = "after_think"
    elif "###Response" in completion:
        reasoning, _separator, solution = completion.partition("###Response")
        boundary = "after_response_marker"
    else:
        solution = completion
        boundary = "full_response"
    if "<final>" in solution and "</final>" in solution:
        _before, _open, after_open = solution.rpartition("<final>")
        final_text, _close, _after = after_open.partition("</final>")
        boundary += "_final_tag"
    else:
        final_text = solution
    return CompletionParse(
        extractor=f"last_boxed_{boundary}",
        reasoning_text=reasoning,
        final_text=final_text,
        extracted_answer=extract_answer(final_text),
    )


def _grade_olmo_answer(completion: str, answer: str) -> bool:
    """Grade boxed math responses, including answers without ``</think>``.

    OLMo Instruct often emits a boxed answer without a think delimiter.  The
    DeepScaleR adapter intentionally rejects that format, so this pair-specific
    adapter keeps the same extraction/equivalence logic but grades the complete
    response when no response delimiter is present.
    """

    model_answer = _parse_boxed_math_completion(completion).extracted_answer
    if model_answer is None or answer == "":
        return False
    ground_truth = str(answer)
    if "\\boxed" in ground_truth:
        ground_truth = extract_answer(ground_truth)
        if ground_truth is None:
            return False
    return grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(
        model_answer, ground_truth
    )


class OlmoMathTask(DeepScalerMathTask):
    """Official Dolci-Instruct-RL math prompts with OLMo response grading."""

    sampling_datasets = tuple(
        DatasetSource(
            repo="allenai/Dolci-Instruct-RL",
            filename=f"data/train-{index:05d}-of-00003.parquet",
            split="train",
            domain="math",
        )
        for index in range(3)
    )
    sampling_dataset = sampling_datasets[0]

    def include_sampling_row(self, row: dict[str, Any]) -> bool:
        datasets = row.get("dataset")
        return (
            (isinstance(datasets, list) and "math" in datasets)
            or row.get("ability") == "math"
        )

    def normalize_sampling_row(self, row: dict[str, Any]) -> dict[str, Any]:
        prompt = row.get("prompt")
        if isinstance(prompt, str) and prompt.startswith("user: "):
            problem = prompt.removeprefix("user: ")
        else:
            problem = None
            source_prompt = row.get("source_prompt")
            if isinstance(source_prompt, list):
                for message in reversed(source_prompt):
                    if (
                        isinstance(message, dict)
                        and message.get("role") == "user"
                        and isinstance(message.get("content"), str)
                    ):
                        problem = message["content"]
                        break

        answers = row.get("ground_truth")
        answer = (
            next((value for value in answers if value not in {None, ""}), None)
            if isinstance(answers, list)
            else None
        )
        if answer is None:
            reward_model = row.get("reward_model")
            if isinstance(reward_model, dict):
                answer = reward_model.get("ground_truth")

        question_id = row.get("id")
        if question_id is None:
            extra_info = row.get("extra_info")
            if isinstance(extra_info, dict):
                question_id = extra_info.get("index")
        if question_id is None and isinstance(problem, str):
            question_id = hashlib.sha256(problem.encode()).hexdigest()

        if not isinstance(problem, str) or not problem:
            raise ValueError("OLMo sampling row has no supported math prompt")
        if answer is None:
            raise ValueError("OLMo sampling row has no ground truth")
        return {
            "problem": problem,
            "answer": str(answer),
            "question_id": str(question_id),
        }

    def grade(self, completion: str, answer: str) -> bool:
        return _grade_olmo_answer(completion, answer)

    def parse_completion(self, completion: str) -> CompletionParse:
        return _parse_boxed_math_completion(completion)


__all__ = ["OlmoMathTask"]
