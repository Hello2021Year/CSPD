"""Minimal prompt-only model wrapper and one-update training primitive."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .objective import ResponseSFTExample, response_cross_entropy


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    input_ids: Any
    attention_mask: Any
    labels: Any
    insert_at: Any


def initialize_prompt_embeddings(
    base_model: Any, tokenizer: Any, prompt_init: Any, prompt_length: int
) -> Any:
    """Initialize a standalone prompt with the declared deterministic recipe."""

    import torch

    embedding = base_model.get_input_embeddings()
    weight = embedding.weight
    generator = torch.Generator(device=weight.device)
    generator.manual_seed(int(prompt_init.seed))
    if prompt_init.method == "gaussian":
        return torch.randn(
            prompt_length,
            weight.size(1),
            device=weight.device,
            dtype=torch.float32,
            generator=generator,
        ) * float(prompt_init.std)
    if prompt_init.method == "sampled_vocabulary":
        upper = min(len(tokenizer), weight.size(0))
        special_ids = {int(value) for value in tokenizer.all_special_ids}
        allowed = [token_id for token_id in range(upper) if token_id not in special_ids]
        if not allowed:
            raise ValueError("tokenizer has no non-special vocabulary rows to sample")
        allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=weight.device)
        positions = torch.randint(
            len(allowed),
            (prompt_length,),
            device=weight.device,
            generator=generator,
        )
        return (
            weight.detach().index_select(0, allowed_tensor[positions]).float().clone()
        )
    raise ValueError(f"unsupported prompt initialization {prompt_init.method!r}")


def collate_examples(
    examples: Sequence[ResponseSFTExample],
    *,
    insert_at: Sequence[int],
    pad_token_id: int,
) -> TrainingBatch:
    import torch

    if not examples:
        raise ValueError("cannot collate an empty batch")
    if len(insert_at) != len(examples):
        raise ValueError("insert_at must contain one position per example")
    width = max(len(example.input_ids) for example in examples)
    inputs = []
    masks = []
    labels = []
    positions = []
    for example, position in zip(examples, insert_at, strict=True):
        position = int(position)
        if not 0 <= position <= len(example.prompt_ids):
            raise ValueError("insert_at must be inside the example prompt")
        positions.append(position)
        padding = width - len(example.input_ids)
        inputs.append(list(example.input_ids) + [int(pad_token_id)] * padding)
        masks.append([1] * len(example.input_ids) + [0] * padding)
        labels.append(list(example.labels_without_soft_prompt) + [-100] * padding)
    return TrainingBatch(
        input_ids=torch.tensor(inputs, dtype=torch.long),
        attention_mask=torch.tensor(masks, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        insert_at=torch.tensor(positions, dtype=torch.long),
    )


def configure_activation_checkpointing(base_model: Any, enabled: bool) -> None:
    """Configure recomputation without changing the frozen parameter set."""

    if not enabled:
        base_model.eval()
        return
    if not getattr(base_model, "supports_gradient_checkpointing", False):
        raise RuntimeError(
            f"{type(base_model).__name__} does not support activation checkpointing"
        )
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(base_model, "config"):
        base_model.config.use_cache = False
    # Transformers checkpointing layers only recompute while training=True.
    # Freezing parameters below remains authoritative; train mode does not make
    # the backbone trainable. Qwen2 reasoning checkpoints use zero dropout.
    base_model.train()


class SoftPromptModel:
    """Frozen causal LM plus one trainable block of input embeddings.

    This intentionally is not an ``nn.Module``: callers cannot accidentally pass
    the frozen backbone to an optimizer by using ``wrapper.parameters()``.  The
    sole optimizer parameter is exposed as ``prompt_parameter``.
    """

    def __init__(
        self,
        base_model: Any,
        prompt_embeddings: Any,
        *,
        disallowed_output_ids: Iterable[int] = (),
        activation_checkpointing: bool = False,
        cross_entropy_backend: str = "torch",
        loss_normalization: str = "token",
    ) -> None:
        import torch

        if prompt_embeddings.ndim != 2 or prompt_embeddings.size(0) <= 0:
            raise ValueError(
                "prompt_embeddings must have shape [prompt_length, hidden]"
            )
        self.base_model = base_model
        self.prompt_parameter = torch.nn.Parameter(prompt_embeddings.detach().clone())
        self.disallowed_output_ids = tuple(
            sorted(set(int(value) for value in disallowed_output_ids))
        )
        if cross_entropy_backend not in {"torch", "liger_fused"}:
            raise ValueError("cross_entropy_backend must be torch or liger_fused")
        if loss_normalization not in {"token", "sequence"}:
            raise ValueError("loss_normalization must be token or sequence")
        if cross_entropy_backend == "liger_fused" and self.disallowed_output_ids:
            raise ValueError(
                "liger_fused does not support masking disallowed output IDs"
            )
        self.cross_entropy_backend = cross_entropy_backend
        self.loss_normalization = loss_normalization
        self._liger_loss = None
        if cross_entropy_backend == "liger_fused":
            try:
                from liger_kernel.transformers import (
                    LigerFusedLinearCrossEntropyLoss,
                )
            except ImportError as error:
                raise RuntimeError(
                    "liger_fused requires liger-kernel in the training runtime"
                ) from error
            self._liger_loss = LigerFusedLinearCrossEntropyLoss(
                reduction="mean"
            )
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        configure_activation_checkpointing(self.base_model, activation_checkpointing)

    @property
    def prompt_length(self) -> int:
        return int(self.prompt_parameter.size(0))

    def _inputs_with_prompt(
        self, input_ids: Any, attention_mask: Any, insert_at: Any
    ) -> tuple[Any, Any]:
        import torch

        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [B,L]")
        if insert_at.ndim != 1 or insert_at.size(0) != input_ids.size(0):
            raise ValueError("insert_at must have shape [B]")
        if bool((insert_at < 0).any()) or bool(
            (insert_at > attention_mask.sum(dim=1)).any()
        ):
            raise ValueError("insert_at is outside an unpadded text sequence")
        text = self.base_model.get_input_embeddings()(input_ids)
        prompt = self.prompt_parameter.to(dtype=text.dtype, device=text.device)
        prompt_mask = attention_mask.new_ones((self.prompt_length,))
        embeddings = torch.stack(
            [
                torch.cat([text[row, :position], prompt, text[row, position:]], dim=0)
                for row, position in enumerate(insert_at.tolist())
            ],
            dim=0,
        )
        full_mask = torch.stack(
            [
                torch.cat(
                    [
                        attention_mask[row, :position],
                        prompt_mask,
                        attention_mask[row, position:],
                    ],
                    dim=0,
                )
                for row, position in enumerate(insert_at.tolist())
            ],
            dim=0,
        )
        return embeddings, full_mask

    def forward(self, input_ids: Any, attention_mask: Any, insert_at: Any) -> Any:
        import torch

        embeddings, full_mask = self._inputs_with_prompt(
            input_ids, attention_mask, insert_at
        )
        output = self.base_model(
            inputs_embeds=embeddings,
            attention_mask=full_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = output.logits
        if self.disallowed_output_ids:
            ids = torch.tensor(
                self.disallowed_output_ids, dtype=torch.long, device=logits.device
            )
            if int(ids.min()) < 0 or int(ids.max()) >= logits.size(-1):
                raise ValueError("disallowed output ID is outside the model vocabulary")
            logits = logits.index_fill(-1, ids, -float("inf"))
        return logits

    def response_loss(
        self,
        input_ids: Any,
        attention_mask: Any,
        insert_at: Any,
        labels: Any,
    ) -> tuple[Any, int]:
        """Compute exact response-only CE without materializing full logits."""

        if self.cross_entropy_backend == "torch":
            return response_cross_entropy(
                self(input_ids, attention_mask, insert_at),
                labels,
                prompt_length=self.prompt_length,
                insert_at=insert_at,
                normalization=self.loss_normalization,
            )

        import torch

        embeddings, full_mask = self._inputs_with_prompt(
            input_ids, attention_mask, insert_at
        )
        decoder_name = getattr(self.base_model, "base_model_prefix", "model")
        decoder = getattr(self.base_model, decoder_name, None)
        output_head = self.base_model.get_output_embeddings()
        if decoder is None or output_head is None or not hasattr(output_head, "weight"):
            raise RuntimeError(
                "liger_fused requires a causal LM exposing decoder and output embeddings"
            )
        hidden = decoder(
            inputs_embeds=embeddings,
            attention_mask=full_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        soft = labels.new_full((self.prompt_length,), -100)
        aligned_labels = torch.stack(
            [
                torch.cat([labels[row, :position], soft, labels[row, position:]])
                for row, position in enumerate(insert_at.tolist())
            ]
        )
        targets = aligned_labels[:, 1:]
        selected_mask = targets.ne(-100)
        token_count = int(selected_mask.sum().item())
        if token_count == 0:
            raise ValueError("training batch contains no response loss tokens")
        head_device = output_head.weight.device
        assert self._liger_loss is not None
        shifted_hidden = hidden[:, :-1, :]
        if self.loss_normalization == "token":
            loss = self._liger_loss(
                output_head.weight,
                shifted_hidden[selected_mask].to(head_device),
                targets[selected_mask].to(head_device),
                bias=getattr(output_head, "bias", None),
            )
            return loss, token_count
        sequence_losses = []
        for row in range(targets.size(0)):
            row_mask = selected_mask[row]
            if not bool(row_mask.any()):
                raise ValueError(
                    f"training batch row {row} contains no response loss tokens"
                )
            sequence_losses.append(
                self._liger_loss(
                    output_head.weight,
                    shifted_hidden[row, row_mask].to(head_device),
                    targets[row, row_mask].to(head_device),
                    bias=getattr(output_head, "bias", None),
                )
            )
        return torch.stack(sequence_losses).mean(), token_count

    __call__ = forward


@dataclass(frozen=True, slots=True)
class UpdateResult:
    loss: float
    response_tokens: int
    gradient_norm: float
    learning_rate: float


def batch_indices_for_update(
    *,
    dataset_size: int,
    effective_batch_size: int,
    batch_seed: int,
    update: int,
) -> list[int]:
    """Return one deterministic full global batch, dropping each epoch's tail."""

    if dataset_size <= 0 or effective_batch_size <= 0 or update <= 0:
        raise ValueError(
            "dataset_size, effective_batch_size, and update must be positive"
        )
    steps_per_epoch = dataset_size // effective_batch_size
    if steps_per_epoch == 0:
        raise ValueError(
            "drop_last training requires dataset_size >= effective_batch_size"
        )
    epoch = (update - 1) // steps_per_epoch
    batch_in_epoch = (update - 1) % steps_per_epoch
    indices = list(range(dataset_size))
    random.Random(int(batch_seed) + epoch).shuffle(indices)
    start = batch_in_epoch * effective_batch_size
    batch = indices[start : start + effective_batch_size]
    assert len(batch) == effective_batch_size
    return batch



def accumulate_response_loss(
    wrapper: SoftPromptModel,
    batch: TrainingBatch,
    *,
    global_loss_denominator: int,
) -> tuple[float, int]:
    """Backpropagate this micro-batch's share of the declared global mean."""

    if global_loss_denominator <= 0:
        raise ValueError("global_loss_denominator must be positive")

    mean_loss, token_count = wrapper.response_loss(
        batch.input_ids,
        batch.attention_mask,
        batch.insert_at,
        batch.labels,
    )
    local_denominator = (
        token_count
        if wrapper.loss_normalization == "token"
        else int(batch.input_ids.size(0))
    )
    loss_sum = mean_loss * local_denominator
    (loss_sum / global_loss_denominator).backward()
    return float(loss_sum.detach()), token_count


def response_token_count(batch: TrainingBatch) -> int:
    """Count supervised response targets without running the model."""

    count = int(batch.labels.ne(-100).sum().item())
    if count <= 0:
        raise ValueError("training batch contains no response loss tokens")
    return count


def finish_prompt_update(
    wrapper: SoftPromptModel,
    optimizer: Any,
    *,
    local_loss_sum: float,
    local_response_tokens: int,
    local_loss_denominator: int,
    learning_rate: float,
    max_grad_norm: float | None,
) -> UpdateResult:
    """Synchronize already-normalized gradients, clip, and apply one update."""

    import torch
    import torch.distributed as distributed

    owned = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    if len(owned) != 1 or owned[0] is not wrapper.prompt_parameter:
        raise RuntimeError("optimizer must own exactly the soft-prompt parameter")
    if any(parameter.requires_grad for parameter in wrapper.base_model.parameters()):
        raise RuntimeError("base model is not fully frozen")
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
    gradient = wrapper.prompt_parameter.grad
    if gradient is None:
        raise RuntimeError("soft-prompt gradient is missing after backward")
    totals = torch.tensor(
        [
            float(local_response_tokens),
            float(local_loss_denominator),
            float(local_loss_sum),
        ],
        dtype=torch.float64,
        device=gradient.device,
    )
    if distributed.is_available() and distributed.is_initialized():
        distributed.all_reduce(gradient, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(totals, op=distributed.ReduceOp.SUM)
    response_tokens = int(totals[0].item())
    loss_denominator = int(totals[1].item())
    loss_sum = float(totals[2].item())
    if response_tokens <= 0:
        raise RuntimeError("global batch contains no response loss tokens")
    if loss_denominator <= 0:
        raise RuntimeError("global loss denominator must be positive")
    if not torch.isfinite(gradient).all():
        raise RuntimeError("soft-prompt gradient is missing or non-finite")
    if max_grad_norm is None:
        gradient_norm = float(torch.linalg.vector_norm(gradient.detach().float()))
    else:
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_([wrapper.prompt_parameter], max_grad_norm)
        )
    if not math.isfinite(gradient_norm):
        raise RuntimeError("soft-prompt gradient norm is non-finite")
    optimizer.step()
    return UpdateResult(
        loss=loss_sum / loss_denominator,
        response_tokens=response_tokens,
        gradient_norm=gradient_norm,
        learning_rate=float(learning_rate),
    )


def train_update(
    wrapper: SoftPromptModel,
    batch: TrainingBatch,
    optimizer: Any,
    *,
    learning_rate: float,
    max_grad_norm: float | None,
) -> UpdateResult:
    """Perform one prompt-only update with the wrapper's declared normalization."""

    owned = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    if len(owned) != 1 or owned[0] is not wrapper.prompt_parameter:
        raise RuntimeError("optimizer must own exactly the soft-prompt parameter")
    if any(parameter.requires_grad for parameter in wrapper.base_model.parameters()):
        raise RuntimeError("base model is not fully frozen")
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
    optimizer.zero_grad(set_to_none=True)
    token_count = response_token_count(batch)
    loss_denominator = (
        token_count
        if wrapper.loss_normalization == "token"
        else int(batch.input_ids.size(0))
    )
    loss_sum, observed_tokens = accumulate_response_loss(
        wrapper,
        batch,
        global_loss_denominator=loss_denominator,
    )
    if observed_tokens != token_count:
        raise RuntimeError("response-token count changed during forward alignment")
    return finish_prompt_update(
        wrapper,
        optimizer,
        local_loss_sum=loss_sum,
        local_response_tokens=token_count,
        local_loss_denominator=loss_denominator,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
    )


__all__ = [
    "SoftPromptModel",
    "configure_activation_checkpointing",
    "TrainingBatch",
    "UpdateResult",
    "accumulate_response_loss",
    "batch_indices_for_update",
    "collate_examples",
    "finish_prompt_update",
    "initialize_prompt_embeddings",
    "response_token_count",
    "train_update",
]
