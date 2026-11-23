"""Task-independent grading value objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class GradeResult:
    """One auditable reward/evaluation decision."""

    correct: bool
    grader: str
    metadata: Mapping[str, Any]


__all__ = ["GradeResult"]
