from __future__ import annotations

import inspect
import io
import json
from types import SimpleNamespace

import pytest

from soft_prompt_distillation.distillation import (
    cap_trajectories_per_question,
    filter_trajectories_by_sequence_length,
    load_training_trajectories,
    log_training_sample,
    record_training_metric,
    scheduled_checkpoint_updates,
    training_sample_payload,
    validate_snapshot_prompt_reuse,
)
import soft_prompt_distillation.distillation as distillation_module
from soft_prompt_distillation.objective import make_response_sft_example
from soft_prompt_distillation.training import (
    SoftPromptModel,
    accumulate_response_loss,
    batch_indices_for_update,
    collate_examples,
    finish_prompt_update,
    initialize_prompt_embeddings,
    response_token_count,
    train_update,
)


def test_checkpoint_schedule_always_includes_final_update() -> None:
    assert scheduled_checkpoint_updates(10, None) == (10,)
    assert scheduled_checkpoint_updates(10, 3) == (3, 6, 9, 10)
    assert scheduled_checkpoint_updates(10, 5) == (5, 10)


def test_training_recomputes_insertion_position_from_current_config() -> None:
    source = inspect.getsource(distillation_module.train)

    assert 'trajectory["insert_at"]' not in source
    assert "resolve_insertion_index(config, current)" in source
    assert "make_response_sft_example(\n                        current.token_ids" in source
    assert "validate_snapshot_prompt_reuse(" in source
    assert "tokenizer=snapshot_template_tokenizer" in source


def test_snapshot_prompt_reuse_allows_marker_boundary_tokenization_change() -> None:
    snapshot = SimpleNamespace(token_ids=(1, 2, 3), text_sha256="same-text")
    current = SimpleNamespace(token_ids=(1, 9, 3), text_sha256="same-text")

    validate_snapshot_prompt_reuse(snapshot, current, (1, 2, 3))


def test_snapshot_prompt_reuse_rejects_changed_marker_free_text() -> None:
    snapshot = SimpleNamespace(token_ids=(1, 2, 3), text_sha256="old-text")
    current = SimpleNamespace(token_ids=(1, 9, 3), text_sha256="new-text")

    with pytest.raises(RuntimeError, match="marker-free prompt text"):
        validate_snapshot_prompt_reuse(snapshot, current, (1, 2, 3))


def test_training_metric_is_flushed_and_checkpoint_progress_is_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FlushTrackingBuffer(io.StringIO):
        flush_count = 0

        def flush(self) -> None:
            self.flush_count += 1
            super().flush()

    progress = []
    monkeypatch.setattr(
        "soft_prompt_distillation.distillation.log_progress", progress.append
    )
    handle = FlushTrackingBuffer()
    metric = {
        "update": 10,
        "epoch": 1,
        "trajectories": 128,
        "loss": 1.25,
        "response_tokens": 2048,
        "gradient_norm": 0.5,
        "learning_rate": 0.003,
    }

    record_training_metric(handle, metric, total_updates=150, checkpoint=True)

    assert handle.flush_count == 1
    assert json.loads(handle.getvalue()) == metric
    assert progress == [
        "stage=train event=checkpoint update=10/150 loss=1.25 "
        "gradient_norm=0.5 learning_rate=0.003"
    ]


def test_non_checkpoint_training_metric_does_not_log_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = []
    monkeypatch.setattr(
        "soft_prompt_distillation.distillation.log_progress", progress.append
    )
    handle = io.StringIO()
    metric = {
        "update": 1,
        "epoch": 1,
        "trajectories": 128,
        "loss": 2.0,
        "response_tokens": 1024,
        "gradient_norm": 1.0,
        "learning_rate": 0.0003,
    }

    record_training_metric(handle, metric, total_updates=150, checkpoint=False)

    assert json.loads(handle.getvalue()) == metric
    assert progress == []


def test_training_sample_log_contains_rendered_text_and_exact_token_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = []
    monkeypatch.setattr(
        "soft_prompt_distillation.distillation.log_progress", progress.append
    )
    rendered = SimpleNamespace(
        messages=(
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "2+2?"},
        ),
        text="<system>be helpful</system><user>2+2?</user>",
        token_ids=(10, 11, 12),
        control_token_ids=(10, -200, 11, 12),
        tokenization_strategy="marker_segmented",
        token_ids_sha256="prompt-sha",
        tokenization_matches_canonical=False,
        canonical_token_count=2,
        canonical_token_ids_sha256="canonical-sha",
    )
    trajectory = {
        "question_id": "q1",
        "source_row_index": 7,
        "sample_index": 3,
        "completion": "four\nlines stay escaped",
        "response_ids": [20, 21],
    }

    payload = training_sample_payload(
        trajectory,
        rendered,
        trajectory_index=5,
        insert_at=1,
        soft_prompt_length=128,
    )
    log_training_sample(payload)

    assert len(progress) == 1
    prefix = "stage=train event=sample payload="
    assert progress[0].startswith(prefix)
    logged = json.loads(progress[0][len(prefix) :])
    assert logged["selection"] == "first_global_batch_update_1"
    assert logged["trajectory_index"] == 5
    assert logged["rendered_prompt_text"] == rendered.text
    assert logged["prompt_token_ids"] == [10, 11, 12]
    assert logged["control_prompt_token_ids"] == [10, -200, 11, 12]
    assert logged["response_token_ids"] == [20, 21]
    assert logged["soft_prompt_insert_at"] == 1
    assert logged["soft_prompt_length"] == 128
    assert logged["training_sequence_length_with_soft_prompt"] == 133
    assert "\n" not in progress[0]


@pytest.mark.parametrize("interval", [0, -1])
def test_checkpoint_schedule_rejects_invalid_interval(interval: int) -> None:
    with pytest.raises(ValueError, match="checkpoint_every_updates"):
        scheduled_checkpoint_updates(10, interval)


def test_sequence_length_filter_counts_soft_tokens_and_preserves_order() -> None:
    trajectories = [
        {"prompt_ids": [1, 2], "response_ids": [3]},
        {"prompt_ids": [1, 2], "response_ids": [3, 4, 5]},
        {"prompt_ids": [1], "response_ids": [2]},
    ]
    kept, stats = filter_trajectories_by_sequence_length(
        trajectories,
        prompt_length=2,
        max_sequence_length=5,
        overlength_policy="filter",
    )
    assert kept == [trajectories[0], trajectories[2]]
    assert stats == {
        "input_trajectories": 3,
        "kept_trajectories": 2,
        "filtered_trajectories": 1,
        "max_observed_sequence_length": 7,
        "max_kept_sequence_length": 5,
        "max_sequence_length": 5,
        "overlength_policy": "filter",
    }


def test_sequence_length_error_reports_first_overlength_trajectory() -> None:
    with pytest.raises(RuntimeError, match=r"trajectory 1 sequence length 6"):
        filter_trajectories_by_sequence_length(
            [
                {"prompt_ids": [1], "response_ids": [2]},
                {"prompt_ids": [1, 2], "response_ids": [3, 4]},
            ],
            prompt_length=2,
            max_sequence_length=5,
            overlength_policy="error",
        )


def test_per_question_trajectory_cap_preserves_question_and_snapshot_order() -> None:
    trajectories = [
        {"question_id": "q0", "response_ids": [1]},
        {"question_id": "q1", "response_ids": [2]},
        {"question_id": "q0", "response_ids": [3]},
        {"question_id": "q0", "response_ids": [4]},
        {"question_id": "q1", "response_ids": [5]},
    ]
    kept, stats = cap_trajectories_per_question(
        trajectories, max_trajectories_per_question=2
    )
    assert kept == [
        trajectories[0],
        trajectories[1],
        trajectories[2],
        trajectories[4],
    ]
    assert stats == {
        "input_trajectories": 5,
        "kept_trajectories": 4,
        "capped_trajectories": 1,
        "question_count": 2,
        "max_observed_trajectories_per_question": 3,
        "max_trajectories_per_question": 2,
    }


def test_streaming_training_loader_matches_filter_then_cap(tmp_path) -> None:
    trajectories = [
        {"question_id": "q0", "prompt_ids": [1], "response_ids": [10]},
        {"question_id": "q0", "prompt_ids": [1], "response_ids": [11, 12, 13]},
        {"question_id": "q1", "prompt_ids": [1], "response_ids": [20]},
        {"question_id": "q0", "prompt_ids": [1], "response_ids": [14]},
        {"question_id": "q1", "prompt_ids": [1], "response_ids": [21]},
    ]
    snapshot = tmp_path / "teacher_trajectories.jsonl"
    snapshot.write_text(
        "".join(json.dumps(row) + "\n" for row in trajectories),
        encoding="utf-8",
    )

    expected_after_length, expected_length_stats = (
        filter_trajectories_by_sequence_length(
            trajectories,
            prompt_length=2,
            max_sequence_length=5,
            overlength_policy="filter",
        )
    )
    expected, expected_cap_stats = cap_trajectories_per_question(
        expected_after_length,
        max_trajectories_per_question=2,
    )
    actual, actual_length_stats, actual_cap_stats = load_training_trajectories(
        snapshot,
        prompt_length=2,
        max_sequence_length=5,
        overlength_policy="filter",
        max_trajectories_per_question=2,
    )

    assert actual == expected
    assert actual_length_stats == expected_length_stats
    assert actual_cap_stats == expected_cap_stats


def test_one_update_changes_only_prompt_and_keeps_backbone_frozen() -> None:
    torch = pytest.importorskip("torch")

    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.head = torch.nn.Linear(4, 8, bias=False)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, inputs_embeds, attention_mask, use_cache, return_dict):
            assert use_cache is False and return_dict is True
            # Depend on all prior embeddings so the inserted prompt gets gradient.
            hidden = inputs_embeds.cumsum(dim=1)
            return SimpleNamespace(logits=self.head(hidden))

    torch.manual_seed(7)
    base = TinyCausalLM()
    before = {name: value.detach().clone() for name, value in base.state_dict().items()}
    wrapper = SoftPromptModel(base, torch.zeros((2, 4)))
    optimizer = torch.optim.AdamW([wrapper.prompt_parameter], lr=0.1, weight_decay=0.0)
    batch = collate_examples(
        [make_response_sft_example([1, 2], [3, 4])],
        insert_at=[0],
        pad_token_id=0,
    )
    prompt_before = wrapper.prompt_parameter.detach().clone()

    result = train_update(
        wrapper,
        batch,
        optimizer,
        learning_rate=0.1,
        max_grad_norm=None,
    )

    assert result.response_tokens == 2
    assert result.gradient_norm > 0
    assert not torch.equal(wrapper.prompt_parameter.detach(), prompt_before)
    for name, value in base.state_dict().items():
        assert torch.equal(value, before[name])


def test_activation_checkpointing_keeps_backbone_frozen_and_enables_recompute() -> None:
    torch = pytest.importorskip("torch")
    captured = {}

    class CheckpointableTiny(torch.nn.Module):
        supports_gradient_checkpointing = True

        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.config = SimpleNamespace(use_cache=True)

        def get_input_embeddings(self):
            return self.embedding

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
            captured.update(gradient_checkpointing_kwargs)

    base = CheckpointableTiny()
    wrapper = SoftPromptModel(
        base,
        torch.zeros((2, 4)),
        activation_checkpointing=True,
    )

    assert captured == {"use_reentrant": False}
    assert base.training is True
    assert base.config.use_cache is False
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert wrapper.prompt_parameter.requires_grad is True


def test_qwen_activation_checkpointing_preserves_prompt_gradient() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    base = transformers.Qwen2ForCausalLM(config)
    wrapper = SoftPromptModel(
        base,
        torch.zeros((2, 8)),
        activation_checkpointing=True,
    )
    optimizer = torch.optim.AdamW([wrapper.prompt_parameter], lr=0.1, weight_decay=0.0)
    batch = collate_examples(
        [make_response_sft_example([1, 2], [3, 4])],
        insert_at=[0],
        pad_token_id=0,
    )

    result = train_update(
        wrapper, batch, optimizer, learning_rate=0.1, max_grad_norm=None
    )

    assert result.gradient_norm > 0
    assert base.is_gradient_checkpointing is True


def test_optimizer_scope_must_be_prompt_only() -> None:
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)

        def get_input_embeddings(self):
            return self.embedding

    wrapper = SoftPromptModel(Tiny(), torch.zeros((1, 4)))
    extra = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([wrapper.prompt_parameter, extra], lr=0.1)
    batch = collate_examples(
        [make_response_sft_example([1], [2])], insert_at=[0], pad_token_id=0
    )
    with pytest.raises(RuntimeError, match="exactly the soft-prompt"):
        train_update(
            wrapper,
            batch,
            optimizer,
            learning_rate=0.1,
            max_grad_norm=None,
        )


def test_missing_soft_prompt_gradient_fails_closed() -> None:
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)

        def get_input_embeddings(self):
            return self.embedding

    wrapper = SoftPromptModel(Tiny(), torch.zeros((1, 4)))
    optimizer = torch.optim.AdamW([wrapper.prompt_parameter], lr=0.1, weight_decay=0.0)

    with pytest.raises(RuntimeError, match="gradient is missing after backward"):
        finish_prompt_update(
            wrapper,
            optimizer,
            local_loss_sum=1.0,
            local_response_tokens=1,
            local_loss_denominator=1,
            learning_rate=0.1,
            max_grad_norm=None,
        )


def test_batch_supports_a_different_insertion_position_per_example() -> None:
    torch = pytest.importorskip("torch")

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 1)
            self.embedding.weight.data.copy_(torch.arange(8).reshape(8, 1))
            self.seen_embeddings = None

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, inputs_embeds, attention_mask, use_cache, return_dict):
            self.seen_embeddings = inputs_embeds.detach().clone()
            logits = inputs_embeds.new_zeros((*inputs_embeds.shape[:2], 8))
            return SimpleNamespace(logits=logits)

    model = RecordingModel()
    wrapper = SoftPromptModel(model, torch.tensor([[99.0]]))
    examples = [
        make_response_sft_example([1, 2], [3]),
        make_response_sft_example([4, 5], [6]),
    ]
    batch = collate_examples(examples, insert_at=[0, 2], pad_token_id=0)

    wrapper(batch.input_ids, batch.attention_mask, batch.insert_at)

    assert model.seen_embeddings is not None
    assert model.seen_embeddings.squeeze(-1).tolist() == [
        [99.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 99.0, 6.0],
    ]


def test_collation_rejects_insertion_inside_the_response() -> None:
    pytest.importorskip("torch")
    example = make_response_sft_example([1, 2], [3])
    with pytest.raises(ValueError, match="inside the example prompt"):
        collate_examples([example], insert_at=[3], pad_token_id=0)


def test_global_batch_schedule_is_deterministic_and_drops_epoch_tail() -> None:
    batches = [
        batch_indices_for_update(
            dataset_size=7,
            effective_batch_size=3,
            batch_seed=42,
            update=update,
        )
        for update in range(1, 3)
    ]
    repeated = [
        batch_indices_for_update(
            dataset_size=7,
            effective_batch_size=3,
            batch_seed=42,
            update=update,
        )
        for update in range(1, 3)
    ]
    assert batches == repeated
    assert all(len(batch) == 3 for batch in batches)
    assert len(set(index for batch in batches for index in batch)) == 6


def test_drop_last_rejects_dataset_smaller_than_global_batch() -> None:
    with pytest.raises(ValueError, match="drop_last"):
        batch_indices_for_update(
            dataset_size=2,
            effective_batch_size=3,
            batch_seed=42,
            update=1,
        )


@pytest.mark.parametrize("loss_normalization", ["token", "sequence"])
def test_micro_batch_accumulation_matches_one_full_batch(
    loss_normalization: str,
) -> None:
    torch = pytest.importorskip("torch")

    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 4)
            self.head = torch.nn.Linear(4, 10, bias=False)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, inputs_embeds, attention_mask, use_cache, return_dict):
            return SimpleNamespace(logits=self.head(inputs_embeds.cumsum(dim=1)))

    torch.manual_seed(11)
    full_model = TinyCausalLM()
    micro_model = TinyCausalLM()
    micro_model.load_state_dict(full_model.state_dict())
    initial_prompt = torch.randn(2, 4)
    full_wrapper = SoftPromptModel(
        full_model, initial_prompt, loss_normalization=loss_normalization
    )
    micro_wrapper = SoftPromptModel(
        micro_model, initial_prompt, loss_normalization=loss_normalization
    )
    full_optimizer = torch.optim.AdamW(
        [full_wrapper.prompt_parameter], lr=0.05, weight_decay=0.0
    )
    micro_optimizer = torch.optim.AdamW(
        [micro_wrapper.prompt_parameter], lr=0.05, weight_decay=0.0
    )
    examples = [
        make_response_sft_example([1, 2], [3]),
        make_response_sft_example([4], [5, 6]),
    ]
    full_batch = collate_examples(examples, insert_at=[0, 1], pad_token_id=0)
    train_update(
        full_wrapper,
        full_batch,
        full_optimizer,
        learning_rate=0.05,
        max_grad_norm=None,
    )

    micro_optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    response_tokens = 0
    micro_batches = [
        collate_examples([example], insert_at=[position], pad_token_id=0)
        for example, position in zip(examples, [0, 1], strict=True)
    ]
    global_response_tokens = sum(response_token_count(batch) for batch in micro_batches)
    global_loss_denominator = (
        global_response_tokens
        if loss_normalization == "token"
        else sum(int(batch.input_ids.size(0)) for batch in micro_batches)
    )
    for micro_batch in micro_batches:
        current_loss_sum, current_tokens = accumulate_response_loss(
            micro_wrapper,
            micro_batch,
            global_loss_denominator=global_loss_denominator,
        )
        loss_sum += current_loss_sum
        response_tokens += current_tokens
    finish_prompt_update(
        micro_wrapper,
        micro_optimizer,
        local_loss_sum=loss_sum,
        local_response_tokens=response_tokens,
        local_loss_denominator=global_loss_denominator,
        learning_rate=0.05,
        max_grad_norm=None,
    )

    assert torch.allclose(
        full_wrapper.prompt_parameter,
        micro_wrapper.prompt_parameter,
        atol=1e-6,
        rtol=1e-6,
    )


def test_sampled_vocabulary_init_uses_non_special_embedding_rows() -> None:
    torch = pytest.importorskip("torch")

    embedding = torch.nn.Embedding(5, 2)
    embedding.weight.data.copy_(torch.arange(10).reshape(5, 2))
    model = SimpleNamespace(get_input_embeddings=lambda: embedding)

    class Tokenizer:
        all_special_ids = [0, 4]

        def __len__(self):
            return 5

    spec = SimpleNamespace(method="sampled_vocabulary", seed=42, std=None)
    first = initialize_prompt_embeddings(model, Tokenizer(), spec, 4)
    second = initialize_prompt_embeddings(model, Tokenizer(), spec, 4)
    longer = initialize_prompt_embeddings(model, Tokenizer(), spec, 8)
    allowed_rows = {tuple(row.tolist()) for row in embedding.weight[1:4]}
    assert torch.equal(first, second)
    assert torch.equal(first, longer[:4])
    assert all(tuple(row.tolist()) in allowed_rows for row in first)




def test_liger_backend_matches_torch_without_calling_full_lm_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import ModuleType

    torch = pytest.importorskip("torch")

    class FakeFusedLoss:
        def __init__(self, *, reduction):
            assert reduction == "mean"

        def __call__(self, weight, hidden, targets, *, bias=None):
            return torch.nn.functional.cross_entropy(
                torch.nn.functional.linear(hidden, weight, bias).float(), targets
            )

    package = ModuleType("liger_kernel")
    transformers_module = ModuleType("liger_kernel.transformers")
    transformers_module.LigerFusedLinearCrossEntropyLoss = FakeFusedLoss
    monkeypatch.setitem(sys.modules, "liger_kernel", package)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", transformers_module)

    class Decoder(torch.nn.Module):
        def forward(self, *, inputs_embeds, attention_mask, use_cache, return_dict):
            assert use_cache is False and return_dict is True
            return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(dim=1))

    class TinyCausalLM(torch.nn.Module):
        base_model_prefix = "model"

        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 4)
            self.model = Decoder()
            self.head = torch.nn.Linear(4, 10, bias=False)
            self.full_forward_calls = 0

        def get_input_embeddings(self):
            return self.embedding

        def get_output_embeddings(self):
            return self.head

        def forward(self, *, inputs_embeds, attention_mask, use_cache, return_dict):
            self.full_forward_calls += 1
            hidden = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=use_cache,
                return_dict=return_dict,
            ).last_hidden_state
            return SimpleNamespace(logits=self.head(hidden))

    torch.manual_seed(19)
    torch_model = TinyCausalLM()
    liger_model = TinyCausalLM()
    liger_model.load_state_dict(torch_model.state_dict())
    initial_prompt = torch.randn(2, 4)
    torch_wrapper = SoftPromptModel(torch_model, initial_prompt)
    liger_wrapper = SoftPromptModel(
        liger_model, initial_prompt, cross_entropy_backend="liger_fused"
    )
    batch = collate_examples(
        [make_response_sft_example([1, 2], [3, 4])],
        insert_at=[1],
        pad_token_id=0,
    )
    torch_optimizer = torch.optim.SGD([torch_wrapper.prompt_parameter], lr=0.1)
    liger_optimizer = torch.optim.SGD([liger_wrapper.prompt_parameter], lr=0.1)

    torch_result = train_update(
        torch_wrapper, batch, torch_optimizer, learning_rate=0.1, max_grad_norm=None
    )
    liger_result = train_update(
        liger_wrapper, batch, liger_optimizer, learning_rate=0.1, max_grad_norm=None
    )

    assert liger_model.full_forward_calls == 0
    assert torch_result.loss == pytest.approx(liger_result.loss, abs=1e-6)
    assert torch.allclose(
        torch_wrapper.prompt_parameter,
        liger_wrapper.prompt_parameter,
        atol=1e-6,
        rtol=1e-6,
    )


def test_training_state_round_trips_optimizer_and_update(tmp_path) -> None:
    """Resume must restore AdamW's moments, not just the prompt vectors.

    Reloading only the checkpoint would silently reset the first and second
    moments, changing the trajectory of every subsequent update.
    """

    torch = pytest.importorskip("torch")
    from soft_prompt_distillation.distillation import (
        _load_training_state,
        _save_training_state,
    )

    parameter = torch.nn.Parameter(torch.zeros(2, 3))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    _save_training_state(tmp_path, 7, optimizer, 5)
    restored = _load_training_state(tmp_path)
    assert restored is not None
    assert restored["update"] == 7
    assert restored["metrics_count"] == 5

    fresh = torch.nn.Parameter(torch.zeros(2, 3))
    fresh_optimizer = torch.optim.AdamW([fresh], lr=0.01)
    assert fresh_optimizer.state_dict()["state"] == {}
    fresh_optimizer.load_state_dict(restored["optimizer"])
    original = optimizer.state_dict()["state"][0]
    recovered = fresh_optimizer.state_dict()["state"][0]
    assert torch.equal(original["exp_avg"], recovered["exp_avg"])
    assert torch.equal(original["exp_avg_sq"], recovered["exp_avg_sq"])
    assert int(original["step"]) == int(recovered["step"])


def test_load_training_state_returns_none_without_a_prior_run(tmp_path) -> None:
    pytest.importorskip("torch")
    from soft_prompt_distillation.distillation import _load_training_state

    assert _load_training_state(tmp_path) is None


def test_truncate_metrics_drops_rows_after_the_saved_state(tmp_path) -> None:
    """A crash can leave metric rows for updates the saved state does not cover;
    resuming must not leave those stale rows in the file."""

    from soft_prompt_distillation.distillation import _truncate_metrics

    path = tmp_path / "training_metrics.jsonl"
    path.write_text("".join(f'{{"update": {i}}}\n' for i in range(1, 6)), encoding="utf-8")
    _truncate_metrics(path, 3)
    rows = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 3
    assert rows[-1] == '{"update": 3}'
    _truncate_metrics(path, 10)          # keep > rows is a no-op
    assert len([l for l in path.read_text().splitlines() if l.strip()]) == 3


def test_resume_step_accepts_matching_saved_metrics_and_stale_tail(tmp_path) -> None:
    from soft_prompt_distillation.distillation import _validate_resume_step

    path = tmp_path / "training_metrics.jsonl"
    path.write_text(
        "".join(f'{{"update": {i}}}\n' for i in range(1, 5)),
        encoding="utf-8",
    )

    assert _validate_resume_step(
        {"update": 3, "metrics_count": 3}, path, 10
    ) == (3, 3)


@pytest.mark.parametrize(
    ("state", "updates", "match"),
    [
        ({"update": 3, "metrics_count": 2}, [1, 2, 3], "disagrees"),
        ({"update": 3, "metrics_count": 3}, [1, 3, 2], "row 2"),
        ({"update": 3, "metrics_count": 3}, [1, 2], "only 2 rows"),
    ],
)
def test_resume_step_rejects_inconsistent_metrics(
    tmp_path, state, updates, match
) -> None:
    from soft_prompt_distillation.distillation import _validate_resume_step

    path = tmp_path / "training_metrics.jsonl"
    path.write_text(
        "".join(f'{{"update": {i}}}\n' for i in updates),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=match):
        _validate_resume_step(state, path, 10)
