"""Math adapters for the Swallow MoE and Nemotron Cascade pairs."""

from __future__ import annotations

from typing import Any

from .base import CompletionParse, DatasetSource
from .math_grader import extract_answer, grade_answer_mathd, grade_answer_sympy
from .olmo_math import OlmoMathTask


class DolciThinkMathTask(OlmoMathTask):
    """The math subset used for both Qwen3- and GPT-OSS-Swallow RLVR."""

    sampling_datasets = tuple(
        DatasetSource(
            repo="allenai/Dolci-Think-RL-7B",
            filename=f"data/train-{index:05d}-of-00009.parquet",
            split="train",
            domain="math",
        )
        for index in range(9)
    )
    sampling_dataset = sampling_datasets[0]

    # Dolci-Think-RL-7B uses the same prompt, ground_truth, and dataset columns
    # handled by OlmoMathTask. Keep this named adapter separate so provenance
    # cannot silently fall back to Dolci-Instruct-RL.


def _parse_harmony_math_completion(completion: str) -> CompletionParse:
    """Extract GPT-OSS analysis/final channels without dropping raw evidence."""

    analysis_marker = "<|channel|>analysis<|message|>"
    final_marker = "<|channel|>final<|message|>"
    reasoning = None
    if analysis_marker in completion:
        analysis = completion.rsplit(analysis_marker, 1)[1]
        for boundary in ("<|end|>", "<|start|>assistant", final_marker):
            if boundary in analysis:
                analysis = analysis.split(boundary, 1)[0]
        reasoning = analysis
    if final_marker in completion:
        final_text = completion.rsplit(final_marker, 1)[1]
        boundary = "harmony_final_channel"
    else:
        final_text = completion
        boundary = "full_response_no_final_channel"
    for stop in ("<|return|>", "<|end|>"):
        if stop in final_text:
            final_text = final_text.split(stop, 1)[0]
    return CompletionParse(
        extractor=f"last_boxed_{boundary}",
        reasoning_text=reasoning,
        final_text=final_text,
        extracted_answer=extract_answer(final_text),
    )


class GptOssSwallowMathTask(DolciThinkMathTask):
    """Dolci math prompts plus GPT-OSS Harmony channel-aware grading."""

    # TaskAdapter deliberately does not inherit multi-artifact declarations:
    # every concrete adapter must opt into its exact published shards.
    sampling_datasets = DolciThinkMathTask.sampling_datasets
    sampling_dataset = sampling_datasets[0]

    def parse_completion(self, completion: str) -> CompletionParse:
        return _parse_harmony_math_completion(completion)

    def grade(self, completion: str, answer: str) -> bool:
        model_answer = self.parse_completion(completion).extracted_answer
        if model_answer is None or answer == "":
            return False
        ground_truth = str(answer)
        if "\\boxed" in ground_truth:
            extracted = extract_answer(ground_truth)
            if extracted is None:
                return False
            ground_truth = extracted
        return grade_answer_mathd(
            model_answer, ground_truth
        ) or grade_answer_sympy(model_answer, ground_truth)


class NemotronCascadeMathTask(OlmoMathTask):
    """The exact 14,476-problem public math RL artifact for Cascade."""

    sampling_dataset = DatasetSource(
        repo="nvidia/Nemotron-Cascade-RL-Math",
        filename="math.json",
        split="train",
        domain="math",
    )

    def include_sampling_row(self, row: dict[str, Any]) -> bool:
        """Keep every row from the already math-only Cascade artifact.

        ``OlmoMathTask`` filters a mixed dataset through its ``dataset`` and
        ``ability`` columns.  Cascade's dedicated math artifact has neither
        column, so inheriting that filter would silently drop all 14,476 rows.
        """

        return True

    def normalize_sampling_row(self, row: dict[str, Any]) -> dict[str, Any]:
        problem = row.get("problem")
        answer = row.get("answer")
        if not isinstance(problem, str) or not problem:
            raise ValueError("Nemotron Cascade sampling row has no problem")
        if answer is None:
            raise ValueError("Nemotron Cascade sampling row has no answer")
        return {
            "problem": problem,
            "answer": str(answer),
            # The artifact has no ID column; the stable question identity is
            # therefore derived by the shared text-identity helper downstream.
            "question_id": self._question_identity(problem),
        }

    @staticmethod
    def _question_identity(problem: str) -> str:
        import hashlib

        return hashlib.sha256(problem.encode()).hexdigest()


__all__ = [
    "DolciThinkMathTask",
    "GptOssSwallowMathTask",
    "NemotronCascadeMathTask",
]
