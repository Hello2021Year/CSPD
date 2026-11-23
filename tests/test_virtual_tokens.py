from __future__ import annotations

from types import SimpleNamespace

import torch

from soft_prompt_distillation.virtual_tokens import (
    insert_virtual_token_ids,
    install_virtual_prompt,
    load_virtual_prompt_metadata,
    write_virtual_prompt_metadata,
)


class _Tokenizer:
    def __init__(self, size: int) -> None:
        self.tokens = {f"base-{index}": index for index in range(size)}

    def __len__(self) -> int:
        return len(self.tokens)

    def add_special_tokens(self, values):
        added = 0
        for token in values["additional_special_tokens"]:
            if token not in self.tokens:
                self.tokens[token] = len(self.tokens)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self.tokens[token]


class _Model:
    def __init__(self, tied: bool = False, vocab_size: int = 3) -> None:
        self.config = SimpleNamespace(tie_word_embeddings=tied)
        self.input = torch.nn.Embedding(vocab_size, 2)
        self.output = (
            self.input if tied else torch.nn.Linear(2, vocab_size, bias=False)
        )

    def get_input_embeddings(self):
        return self.input

    def get_output_embeddings(self):
        return self.output

    def resize_token_embeddings(self, size, mean_resizing=False):
        assert mean_resizing is False
        old_input = self.input.weight.detach().clone()
        old_output = self.output.weight.detach().clone()
        self.input = torch.nn.Embedding(size, 2)
        self.output = (
            self.input
            if self.config.tie_word_embeddings
            else torch.nn.Linear(2, size, bias=False)
        )
        with torch.no_grad():
            self.input.weight[: len(old_input)].copy_(old_input)
            if not self.config.tie_word_embeddings:
                self.output.weight[: len(old_output)].copy_(old_output)


def test_virtual_prompt_becomes_contiguous_input_only_tokens(tmp_path) -> None:
    model = _Model()
    tokenizer = _Tokenizer(3)
    prompt = torch.tensor([[9.0, 8.0], [7.0, 6.0]])
    spec = install_virtual_prompt(model, tokenizer, prompt)

    assert spec.token_ids == (3, 4)
    assert torch.equal(model.input.weight[3:].detach(), prompt)
    assert torch.equal(model.output.weight[3:].detach(), torch.zeros(2, 2))
    assert insert_virtual_token_ids([10, 11], 1, spec.token_ids) == [10, 3, 4, 11]

    write_virtual_prompt_metadata(
        tmp_path, spec, base_model="org/student", soft_prompt_sha256="abc"
    )
    assert load_virtual_prompt_metadata(tmp_path) == spec


def test_virtual_prompt_supports_tied_embeddings_with_runtime_logit_bias() -> None:
    model = _Model(tied=True)
    prompt = torch.tensor([[9.0, 8.0], [7.0, 6.0]])
    spec = install_virtual_prompt(model, _Tokenizer(3), prompt)

    assert spec.logit_bias < 0
    assert model.input.weight.data_ptr() == model.output.weight.data_ptr()
    assert torch.equal(model.input.weight[3:].detach(), prompt)
    assert torch.equal(model.output.weight[3:].detach(), prompt)


def test_virtual_prompt_reuses_model_vocab_padding_rows() -> None:
    model = _Model(vocab_size=8)
    spec = install_virtual_prompt(model, _Tokenizer(3), torch.ones(2, 2))
    assert spec.token_ids == (3, 4)
    assert model.input.num_embeddings == 8
    assert torch.equal(model.input.weight[3:5].detach(), torch.ones(2, 2))
