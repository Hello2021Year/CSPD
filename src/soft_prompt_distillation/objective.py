"""Reference response-only causal-SFT alignment for a soft prompt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


IGNORE_INDEX = -100


class AlignmentError(ValueError):
    """Raised when token regions or causal targets cannot be aligned exactly."""


@dataclass(frozen=True, slots=True)
class ResponseSFTExample:
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_loss_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise AlignmentError("prompt_ids cannot be empty")
        if not self.response_ids:
            raise AlignmentError("response_ids cannot be empty")
        if len(self.response_ids) != len(self.response_loss_mask):
            raise AlignmentError("response_loss_mask length must equal response_ids length")
        if not any(self.response_loss_mask):
            raise AlignmentError("response_loss_mask selects no response tokens")

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.response_ids

    @property
    def labels_without_soft_prompt(self) -> tuple[int, ...]:
        return (IGNORE_INDEX,) * len(self.prompt_ids) + tuple(
            token_id if selected else IGNORE_INDEX
            for token_id, selected in zip(
                self.response_ids, self.response_loss_mask, strict=True
            )
        )


def make_response_sft_example(
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    response_loss_mask: Sequence[bool | int] | None = None,
) -> ResponseSFTExample:
    response = tuple(int(value) for value in response_ids)
    if response_loss_mask is None:
        mask = (True,) * len(response)
    else:
        mask = tuple(bool(value) for value in response_loss_mask)
    return ResponseSFTExample(
        prompt_ids=tuple(int(value) for value in prompt_ids),
        response_ids=response,
        response_loss_mask=mask,
    )


def labels_with_soft_prompt(
    labels_without_soft_prompt: Sequence[int],
    *,
    prompt_length: int,
    insert_at: int,
) -> tuple[int, ...]:
    labels = tuple(int(value) for value in labels_without_soft_prompt)
    if prompt_length <= 0:
        raise AlignmentError("prompt_length must be positive")
    if not 0 <= insert_at <= len(labels):
        raise AlignmentError("insert_at is outside the text-token sequence")
    return (
        labels[:insert_at]
        + (IGNORE_INDEX,) * prompt_length
        + labels[insert_at:]
    )


def prediction_targets(labels: Sequence[int]) -> tuple[int, ...]:
    """Target predicted by logits at each non-final causal position."""

    values = tuple(int(value) for value in labels)
    if len(values) < 2:
        raise AlignmentError("a causal sequence needs at least two positions")
    return values[1:]


def prediction_mask(labels: Sequence[int]) -> tuple[bool, ...]:
    return tuple(target != IGNORE_INDEX for target in prediction_targets(labels))


@dataclass(frozen=True, slots=True)
class TraceRow:
    position: int
    region: str
    token_id: int | None
    label: int
    predicts_position: int | None
    prediction_target: int | None
    loss: bool


def alignment_trace(
    example: ResponseSFTExample,
    *,
    prompt_length: int,
    insert_at: int,
) -> tuple[TraceRow, ...]:
    """Human-reviewable positions after virtual soft-token insertion."""

    text_ids = example.input_ids
    labels = labels_with_soft_prompt(
        example.labels_without_soft_prompt,
        prompt_length=prompt_length,
        insert_at=insert_at,
    )
    tokens: tuple[int | None, ...] = (
        text_ids[:insert_at] + (None,) * prompt_length + text_ids[insert_at:]
    )
    prompt_end = len(example.prompt_ids) + prompt_length
    soft_start = insert_at
    soft_end = insert_at + prompt_length
    rows = []
    for position, (token_id, label) in enumerate(zip(tokens, labels, strict=True)):
        if soft_start <= position < soft_end:
            region = "soft_prompt"
        elif position < prompt_end:
            region = "prompt"
        else:
            response_position = position - prompt_end
            region = (
                "response_loss"
                if example.response_loss_mask[response_position]
                else "response_ignored"
            )
        target = labels[position + 1] if position + 1 < len(labels) else None
        rows.append(
            TraceRow(
                position=position,
                region=region,
                token_id=token_id,
                label=label,
                predicts_position=position + 1 if target is not None else None,
                prediction_target=target,
                loss=target is not None and target != IGNORE_INDEX,
            )
        )
    return tuple(rows)


def response_cross_entropy(
    logits: Any,
    labels_without_soft_prompt: Any,
    *,
    prompt_length: int,
    insert_at: Any,
    normalization: str = "token",
) -> tuple[Any, int]:
    """Mean response-token NLL with explicit causal shift and shape checks.

    Torch is imported only inside this function so config and alignment tests
    remain runnable on the login node without the training environment.
    """

    import torch
    import torch.nn.functional as functional

    if logits.ndim != 3 or labels_without_soft_prompt.ndim != 2:
        raise AlignmentError("expected logits [B,L,V] and labels [B,text_L]")
    if logits.size(0) != labels_without_soft_prompt.size(0):
        raise AlignmentError("logits and labels batch sizes differ")
    positions = torch.as_tensor(
        insert_at, dtype=torch.long, device=labels_without_soft_prompt.device
    )
    if positions.ndim == 0:
        positions = positions.expand(labels_without_soft_prompt.size(0))
    if positions.ndim != 1 or positions.size(0) != labels_without_soft_prompt.size(0):
        raise AlignmentError("insert_at must be a scalar or have shape [B]")
    if bool((positions < 0).any()) or bool(
        (positions > labels_without_soft_prompt.size(1)).any()
    ):
        raise AlignmentError("insert_at is outside the text-token sequence")
    soft = labels_without_soft_prompt.new_full((prompt_length,), IGNORE_INDEX)
    labels = torch.stack(
        [
            torch.cat(
                [
                    labels_without_soft_prompt[row, :position],
                    soft,
                    labels_without_soft_prompt[row, position:],
                ],
                dim=0,
            )
            for row, position in enumerate(positions.tolist())
        ],
        dim=0,
    )
    if logits.size(1) != labels.size(1):
        raise AlignmentError(
            f"logits length {logits.size(1)} does not match aligned labels {labels.size(1)}"
        )
    # A model loaded with ``device_map="balanced"`` produces logits on the
    # output head's device, while the input batch (and therefore labels) may
    # remain on the input-embedding device. Keep the comparatively small
    # aligned target tensor with the logits before selecting response tokens.
    labels = labels.to(logits.device)
    if normalization not in {"token", "sequence"}:
        raise AlignmentError("normalization must be token or sequence")
    targets = labels[:, 1:]
    selected_mask = targets.ne(IGNORE_INDEX)
    token_count = int(selected_mask.sum().item())
    if token_count == 0:
        raise AlignmentError("batch contains no response loss tokens")
    predictions = logits[:, :-1, :]
    if normalization == "token":
        loss = functional.cross_entropy(
            predictions[selected_mask].float(),
            targets[selected_mask],
            reduction="mean",
        )
        return loss, token_count
    sequence_losses = []
    for row in range(targets.size(0)):
        row_mask = selected_mask[row]
        if not bool(row_mask.any()):
            raise AlignmentError(
                f"batch row {row} contains no response loss tokens"
            )
        sequence_losses.append(
            functional.cross_entropy(
                predictions[row, row_mask].float(),
                targets[row, row_mask],
                reduction="mean",
            )
        )
    return torch.stack(sequence_losses).mean(), token_count


__all__ = [
    "AlignmentError",
    "IGNORE_INDEX",
    "ResponseSFTExample",
    "TraceRow",
    "alignment_trace",
    "labels_with_soft_prompt",
    "make_response_sft_example",
    "prediction_mask",
    "prediction_targets",
    "response_cross_entropy",
]
