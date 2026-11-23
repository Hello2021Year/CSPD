from __future__ import annotations

import math

import pytest

from soft_prompt_distillation.objective import (
    IGNORE_INDEX,
    alignment_trace,
    labels_with_soft_prompt,
    make_response_sft_example,
    prediction_mask,
    prediction_targets,
    response_cross_entropy,
)


def test_response_labels_exclude_prompt_and_selected_response_tokens() -> None:
    example = make_response_sft_example(
        prompt_ids=[10, 11, 12],
        response_ids=[20, 21, 22],
        response_loss_mask=[1, 0, 1],
    )
    assert example.input_ids == (10, 11, 12, 20, 21, 22)
    assert example.labels_without_soft_prompt == (
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        20,
        IGNORE_INDEX,
        22,
    )


@pytest.mark.parametrize("insert_at", [0, 1, 3])
def test_soft_insertion_preserves_exact_causal_response_targets(insert_at: int) -> None:
    example = make_response_sft_example([10, 11, 12], [20, 21])
    labels = labels_with_soft_prompt(
        example.labels_without_soft_prompt,
        prompt_length=2,
        insert_at=insert_at,
    )
    targets = prediction_targets(labels)
    selected = [target for target in targets if target != IGNORE_INDEX]
    assert selected == [20, 21]
    assert sum(prediction_mask(labels)) == 2


def test_trace_makes_first_response_prediction_position_obvious() -> None:
    example = make_response_sft_example([10, 11], [20, 21])
    trace = alignment_trace(example, prompt_length=2, insert_at=1)
    assert [row.region for row in trace] == [
        "prompt",
        "soft_prompt",
        "soft_prompt",
        "prompt",
        "response_loss",
        "response_loss",
    ]
    loss_rows = [row for row in trace if row.loss]
    assert [(row.position, row.prediction_target) for row in loss_rows] == [
        (3, 20),
        (4, 21),
    ]
    assert trace[-1].loss is False


def test_nonuniform_logits_catch_one_token_alignment_error() -> None:
    torch = pytest.importorskip("torch")
    example = make_response_sft_example([1, 2], [3, 4])
    labels = torch.tensor([example.labels_without_soft_prompt])
    vocab_size = 8
    aligned = torch.full((1, 6, vocab_size), -10.0)
    # Text positions are [1, soft, soft, 2, 3, 4]. Positions 3 and 4 must
    # predict response tokens 3 and 4 respectively.
    aligned[0, 3, 3] = 10.0
    aligned[0, 4, 4] = 10.0
    correct, count = response_cross_entropy(
        aligned, labels, prompt_length=2, insert_at=1
    )
    shifted = aligned.roll(shifts=1, dims=1)
    wrong, _ = response_cross_entropy(shifted, labels, prompt_length=2, insert_at=1)
    assert count == 2
    assert float(correct) < 1e-5
    assert float(wrong) > 5.0


def test_cross_entropy_rejects_shape_that_omits_soft_positions() -> None:
    torch = pytest.importorskip("torch")
    example = make_response_sft_example([1, 2], [3])
    logits = torch.zeros((1, 3, 8))
    labels = torch.tensor([example.labels_without_soft_prompt])
    with pytest.raises(ValueError, match="does not match aligned labels"):
        response_cross_entropy(logits, labels, prompt_length=2, insert_at=0)


def test_cross_entropy_accepts_per_example_insertion_positions() -> None:
    torch = pytest.importorskip("torch")
    first = make_response_sft_example([1, 2], [3])
    second = make_response_sft_example([4, 5], [6])
    labels = torch.tensor(
        [first.labels_without_soft_prompt, second.labels_without_soft_prompt]
    )
    logits = torch.full((2, 4, 8), -10.0)
    logits[0, 2, 3] = 10.0
    logits[1, 2, 6] = 10.0

    loss, count = response_cross_entropy(
        logits,
        labels,
        prompt_length=1,
        insert_at=torch.tensor([0, 2]),
    )

    assert count == 2
    assert float(loss) < 1e-5


def test_sequence_normalization_weights_each_response_equally() -> None:
    torch = pytest.importorskip("torch")
    labels = torch.tensor(
        [
            [-100, 2, -100, -100],
            [-100, 4, 5, 6],
        ]
    )
    logits = torch.zeros((2, 4, 8))
    logits[0, 0, 2] = 10.0

    token_loss, token_count = response_cross_entropy(
        logits,
        labels,
        prompt_length=0,
        insert_at=torch.tensor([0, 0]),
        normalization="token",
    )
    sequence_loss, sequence_count = response_cross_entropy(
        logits,
        labels,
        prompt_length=0,
        insert_at=torch.tensor([0, 0]),
        normalization="sequence",
    )

    assert token_count == sequence_count == 4
    assert float(sequence_loss) < float(token_loss)


def test_cross_entropy_moves_targets_to_sharded_lm_head_device() -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices to exercise balanced model sharding")

    example = make_response_sft_example([1, 2], [3])
    labels = torch.tensor(
        [example.labels_without_soft_prompt], device=torch.device("cuda:0")
    )
    logits = torch.full(
        (1, 4, 8),
        -10.0,
        device=torch.device("cuda:1"),
        requires_grad=True,
    )
    with torch.no_grad():
        logits[0, 2, 3] = 10.0

    loss, count = response_cross_entropy(
        logits, labels, prompt_length=1, insert_at=1
    )
    assert loss.device == logits.device
    assert count == 1
    loss.backward()
    assert logits.grad is not None
