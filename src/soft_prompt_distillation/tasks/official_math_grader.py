"""Swallow/Lighteval-compatible evaluation grading for free-form math.

The Swallow post-training leaderboard uses Lighteval's expression/LaTeX
extractors and symbolic comparison.  ``math-verify==0.5.2`` contains the same
parser/comparator implementation and pins the same
``latex2sympy2_extended==1.0.6`` dependency as the v202510 Swallow evaluator.
Keep this evaluation contract separate from the stricter reward grader used
to filter teacher trajectories.
"""

from __future__ import annotations

from functools import lru_cache

from math_verify import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    parse,
    verify,
)


SWALLOW_MATH_GRADER = "swallow_lighteval_v202510_math_verify_0.5.2"
_EXPRESSION_GOLD = (ExprExtractionConfig(),)
_LATEX_GOLD = (LatexExtractionConfig(),)
_PREDICTION = (
    ExprExtractionConfig(),
    LatexExtractionConfig(boxed_match_priority=0),
)


@lru_cache(maxsize=16_384)
def _parse_gold(reference: str, reference_format: str):
    if reference_format == "expression":
        extraction = _EXPRESSION_GOLD
    elif reference_format == "latex":
        extraction = _LATEX_GOLD
    else:
        raise ValueError(
            "official math reference_format must be 'expression' or 'latex'"
        )
    return parse(
        reference,
        extraction_config=extraction,
        fallback_mode="first_match",
        extraction_mode="any_match",
        parsing_timeout=5,
    )


def grade_official_math(
    completion: str, reference: str, reference_format: str
) -> bool:
    """Apply the exact extraction/comparison settings used by Swallow math."""

    if not completion or not reference:
        return False
    gold = _parse_gold(reference, reference_format)
    prediction = parse(
        completion,
        extraction_config=_PREDICTION,
        fallback_mode="first_match",
        extraction_mode="any_match",
        parsing_timeout=5,
    )
    if not gold or not prediction:
        return False
    return bool(verify(gold, prediction, precision=5, timeout_seconds=5))


def prime_official_math_reference(reference: str, reference_format: str) -> None:
    """Populate the gold parse cache before forking regrade workers."""

    _parse_gold(reference, reference_format)


__all__ = [
    "SWALLOW_MATH_GRADER",
    "grade_official_math",
    "prime_official_math_reference",
]
