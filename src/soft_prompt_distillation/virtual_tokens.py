"""Export and validate a soft prompt as ordinary model token embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


METADATA_FILENAME = "soft_prompt_tokens.json"
LOGIT_BIAS = -100.0


@dataclass(frozen=True, slots=True)
class VirtualPromptSpec:
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]
    logit_bias: float


def virtual_token_names(prompt_length: int) -> list[str]:
    if prompt_length <= 0:
        raise ValueError("prompt length must be positive")
    if prompt_length > 9999:
        raise ValueError("prompt length must be <= 9999 to fit in token name")
    return [f"<|soft_prompt_{index:04d}|>" for index in range(prompt_length)]


def install_virtual_prompt(
    model: Any,
    tokenizer: Any,
    prompt_embeddings: torch.Tensor,
) -> VirtualPromptSpec:
    """Append virtual tokens and copy learned vectors into their embedding rows.

    Untied models keep the virtual rows input-only by zeroing the corresponding
    output rows.  Tied models necessarily expose the same learned rows through
    the output head; the evaluation runtime's mandatory negative logit bias
    prevents those IDs from being generated.
    """

    if prompt_embeddings.ndim != 2 or prompt_embeddings.size(0) <= 0:
        raise RuntimeError("prompt_embeddings must have shape [prompt_length, hidden_size]")
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise RuntimeError("model must expose input and output embeddings")
    if input_embeddings.weight.size(1) != prompt_embeddings.size(1):
        raise RuntimeError("soft prompt hidden size differs from student embedding size")

    old_tokenizer_size = len(tokenizer)
    tokens = virtual_token_names(prompt_embeddings.size(0))
    added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    if added != len(tokens):
        raise RuntimeError("virtual soft-prompt token names already exist in tokenizer")
    token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in tokens]
    expected_ids = list(range(old_tokenizer_size, old_tokenizer_size + len(tokens)))
    if token_ids != expected_ids or len(tokenizer) != old_tokenizer_size + len(tokens):
        raise RuntimeError("virtual soft-prompt tokens were not appended contiguously")

    # Many open models pad the embedding table beyond len(tokenizer). Reuse
    # those unassigned rows when possible; resize only if the new IDs exceed it.
    if len(tokenizer) > input_embeddings.weight.size(0):
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    assert input_embeddings is not None and output_embeddings is not None
    with torch.no_grad():
        row_indices = torch.tensor(token_ids, device=input_embeddings.weight.device)
        input_embeddings.weight.index_copy_(
            0,
            row_indices,
            prompt_embeddings.to(
                device=input_embeddings.weight.device,
                dtype=input_embeddings.weight.dtype,
            ),
        )
        if input_embeddings.weight.data_ptr() != output_embeddings.weight.data_ptr():
            # These rows are input-only on untied models. A negative runtime
            # logit bias remains the generation guard; zeros also avoid
            # arbitrary random output logits.
            output_embeddings.weight.index_fill_(
                0,
                row_indices.to(output_embeddings.weight.device),
                0.0,
            )
    return VirtualPromptSpec(tuple(token_ids), tuple(tokens), LOGIT_BIAS)


def insert_virtual_token_ids(
    prompt_ids: Sequence[int], insert_at: int, virtual_ids: Sequence[int]
) -> list[int]:
    if not 0 <= insert_at <= len(prompt_ids):
        raise ValueError("insert_at is outside the prompt")
    if not virtual_ids:
        raise ValueError("virtual token IDs must not be empty")
    return [
        *[int(value) for value in prompt_ids[:insert_at]],
        *[int(value) for value in virtual_ids],
        *[int(value) for value in prompt_ids[insert_at:]],
    ]


def write_virtual_prompt_metadata(
    directory: Path,
    spec: VirtualPromptSpec,
    *,
    base_model: str,
    soft_prompt_sha256: str,
) -> Path:
    path = directory / METADATA_FILENAME
    body = {
        "schema_version": 1,
        "base_model": base_model,
        "soft_prompt_sha256": soft_prompt_sha256,
        "prompt_length": len(spec.token_ids),
        "tokens": list(spec.tokens),
        "token_ids": list(spec.token_ids),
        "logit_bias": spec.logit_bias,
    }
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_virtual_prompt_metadata(directory: Path) -> VirtualPromptSpec:
    path = directory / METADATA_FILENAME
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read derived soft-prompt metadata {path}: {exc}") from exc
    if body.get("schema_version") != 1:
        raise RuntimeError("unsupported soft-prompt token metadata schema")
    token_ids = body.get("token_ids")
    tokens = body.get("tokens")
    logit_bias = body.get("logit_bias")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or not all(isinstance(value, int) and value >= 0 for value in token_ids)
        or not isinstance(tokens, list)
        or not all(isinstance(value, str) and value for value in tokens)
        or len(tokens) != len(token_ids)
        or not isinstance(logit_bias, (int, float))
    ):
        raise RuntimeError("invalid soft-prompt token metadata")
    return VirtualPromptSpec(tuple(token_ids), tuple(tokens), float(logit_bias))


__all__ = [
    "VirtualPromptSpec",
    "insert_virtual_token_ids",
    "install_virtual_prompt",
    "load_virtual_prompt_metadata",
    "virtual_token_names",
    "write_virtual_prompt_metadata",
]
