"""Natural-language prompt variants placed where the soft prompt sits.

Text variants are tokenized on their own and inserted at the soft-prompt
insertion index (start of the system message body), so they occupy the same
position as the learned vectors.  ``R_high`` instead switches the template's
built-in ``Reasoning: medium`` line to ``Reasoning: high``.
"""

from __future__ import annotations

from typing import Any, Sequence

TEXT_VARIANTS = {
    "H_check": (
        "Do not conclude too quickly. After each intermediate result, question it "
        "and check it before moving on.\n"
    ),
    "H_verify": (
        "Before concluding anything, pause and ask whether it is really correct. "
        "Prefer to double-check each intermediate result, verify it with an "
        "alternative method, and consider edge cases. Do not stop reasoning early; "
        "give the final answer only after careful verification.\n"
    ),
    "H_neutral": "Please read the problem carefully and answer the user's question.\n",
}
VARIANTS = (*TEXT_VARIANTS, "R_high")


def _replace_once(ids: list[int], old: list[int], new: list[int]) -> list[int]:
    hits = [i for i in range(len(ids) - len(old) + 1) if ids[i:i + len(old)] == old]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one occurrence of {old}, found {len(hits)}")
    i = hits[0]
    return ids[:i] + new + ids[i + len(old):]


def apply_variant(
    prompt_ids: Sequence[int], insert_at: int, variant: str, tokenizer: Any
) -> list[int]:
    ids = [int(x) for x in prompt_ids]
    if variant in TEXT_VARIANTS:
        text_ids = tokenizer.encode(TEXT_VARIANTS[variant], add_special_tokens=False)
        return ids[:insert_at] + text_ids + ids[insert_at:]
    if variant == "R_high":
        old = tokenizer.encode("Reasoning: medium", add_special_tokens=False)
        new = tokenizer.encode("Reasoning: high", add_special_tokens=False)
        return _replace_once(ids, old, new)
    raise ValueError(f"unknown prompt variant {variant!r}")
