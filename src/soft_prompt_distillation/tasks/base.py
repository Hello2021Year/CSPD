"""Minimal interface implemented by each task adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..grading import GradeResult


@dataclass(frozen=True, slots=True)
class DatasetSource:
    repo: str
    filename: str
    split: str
    domain: str
    revision: str | None = None


@dataclass(frozen=True, slots=True)
class CompletionParse:
    """Auditable, model/task-specific view of one decoded completion."""

    extractor: str
    reasoning_text: str | None
    final_text: str
    extracted_answer: str | None


class TaskAdapter(ABC):
    """Translate task-specific datasets and grading into the shared pipeline."""

    sampling_dataset: DatasetSource
    evaluation_datasets: Mapping[
        str, DatasetSource | tuple[DatasetSource, ...]
    ]
    stream_sampling_selection: bool = False
    requires_external_verifier: bool = False

    def get_sampling_datasets(self) -> tuple[DatasetSource, ...]:
        """Return every published RL prompt artifact used by the checkpoint."""

        # A multi-artifact declaration belongs to one concrete adapter.  Using
        # getattr() here lets a child that overrides only ``sampling_dataset``
        # accidentally inherit its parent's artifact list (for example, every
        # OlmoMathTask child would otherwise load Dolci-Instruct-RL).
        sources = vars(type(self)).get("sampling_datasets")
        return tuple(sources) if sources is not None else (self.sampling_dataset,)

    def include_sampling_row(self, row: dict[str, Any]) -> bool:
        """Return whether a source row belongs to this task's RL prompt pool."""

        return True

    def include_evaluation_row(self, row: dict[str, Any], dataset: str) -> bool:
        """Return whether a source row belongs to one reported eval slice."""

        del row, dataset
        return True

    def get_evaluation_datasets(self, label: str) -> tuple[DatasetSource, ...]:
        """Return every artifact pooled into one reported benchmark."""

        source_or_sources = self.evaluation_datasets[label]
        if isinstance(source_or_sources, DatasetSource):
            return (source_or_sources,)
        sources = tuple(source_or_sources)
        if not sources:
            raise RuntimeError(f"evaluation dataset group {label!r} is empty")
        return sources

    @abstractmethod
    def normalize_sampling_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return problem, answer, and question_id for a sampling row."""

    @abstractmethod
    def normalize_evaluation_row(
        self, row: dict[str, Any], dataset: str
    ) -> dict[str, Any]:
        """Return dataset, problem, answer, and question_id for an eval row."""

    def parse_completion(self, completion: str) -> CompletionParse:
        """Return structured output fields without silently guessing an answer."""

        return CompletionParse(
            extractor="unparsed",
            reasoning_text=None,
            final_text=completion,
            extracted_answer=None,
        )

    @abstractmethod
    def grade(self, completion: str, answer: str) -> bool:
        """Return whether a sampling completion is correct.

        This is the reward/filter contract used to select teacher trajectories.
        Evaluation may deliberately use a different published benchmark
        verifier through :meth:`grade_evaluation`.
        """

    def grade_evaluation(
        self, completion: str, row: Mapping[str, Any]
    ) -> bool:
        """Grade one evaluation row without changing the sampling reward."""

        answer = row.get("answer")
        if not isinstance(answer, str):
            raise ValueError("normalized evaluation row has no string answer")
        return self.grade(completion, answer)

    def evaluation_grader(self, row: Mapping[str, Any]) -> str:
        """Return the auditable grader identity written beside each result."""

        del row
        return f"{type(self).__name__}.grade"

    def grade_sampling_batch(
        self,
        completions: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
        *,
        verifier_command: Path | None = None,
        verifier_timeout_seconds: int | None = None,
    ) -> list[GradeResult]:
        """Grade one sampling batch.

        The default preserves scalar task behavior.  Code tasks override this
        hook so generated programs are sent to an isolated external verifier
        instead of being executed in the generation process.
        """

        del verifier_command, verifier_timeout_seconds
        if len(completions) != len(rows):
            raise ValueError("sampling completions and rows have different lengths")
        results = []
        for completion, row in zip(completions, rows, strict=True):
            answer = row.get("answer")
            if not isinstance(answer, str):
                raise ValueError("normalized sampling row has no string answer")
            results.append(
                GradeResult(
                    correct=self.grade(completion, answer),
                    grader=f"{type(self).__name__}.grade",
                    metadata={},
                )
            )
        return results

    def grade_evaluation_batch(
        self,
        completions: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
        *,
        verifier_command: Path | None = None,
        verifier_timeout_seconds: int | None = None,
    ) -> list[GradeResult]:
        """Grade one evaluation batch, optionally through an external verifier."""

        del verifier_command, verifier_timeout_seconds
        if len(completions) != len(rows):
            raise ValueError("evaluation completions and rows have different lengths")
        return [
            GradeResult(
                correct=self.grade_evaluation(completion, row),
                grader=self.evaluation_grader(row),
                metadata={},
            )
            for completion, row in zip(completions, rows, strict=True)
        ]


__all__ = ["CompletionParse", "DatasetSource", "GradeResult", "TaskAdapter"]
