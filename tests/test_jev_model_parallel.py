from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from soft_prompt_distillation.training import SoftPromptModel


M2RL = Path(__file__).resolve().parents[1] / "experiments" / "m2rl"


@pytest.mark.parametrize("device_kind", ("cpu", "cuda"))
def test_cspd_prompt_backpropagates_through_four_stages(device_kind, monkeypatch):
    if device_kind == "cuda" and torch.cuda.device_count() < 4:
        pytest.skip("requires four CUDA GPUs to check cross-device forward and backward")
    monkeypatch.syspath_prepend(str(M2RL))
    from train_jev import place_student

    class Layer(torch.nn.Linear):
        def forward(self, hidden, *, static):
            mask = static["mask"][0]
            assert hidden.device == mask.device == self.weight.device
            if device_kind == "cuda":
                assert torch.cuda.current_device() == self.weight.device.index
            return super().forward(hidden.cumsum(dim=1) * mask)

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = torch.nn.Identity()
            self.layers = torch.nn.ModuleList(Layer(8, 8) for _ in range(8))
            self.norm = torch.nn.LayerNorm(8)

        def forward(self, *, inputs_embeds, attention_mask, **kwargs):
            hidden = inputs_embeds
            for layer in self.layers:
                hidden = layer(hidden, static={"mask": (attention_mask.unsqueeze(-1),)})
            return SimpleNamespace(last_hidden_state=self.norm(hidden))

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Decoder()
            self.embedding = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16, bias=False)

        def get_input_embeddings(self):
            return self.embedding

        def get_output_embeddings(self):
            return self.head

    devices = tuple(torch.device("cuda", index) if device_kind == "cuda" else torch.device("cpu")
                    for index in range(4))
    torch.manual_seed(13)
    student = place_student(Student(), devices[0], devices=devices)
    assert [layer.weight.device for layer in student.model.layers] == [
        device for device in devices for _ in range(2)
    ]
    wrapper = SoftPromptModel(student, torch.zeros(1, 8, device=devices[0]), loss_normalization="sequence")

    def reference_loss(weight, hidden, targets, *, bias=None):
        assert weight.device == hidden.device == targets.device == devices[-1]
        if device_kind == "cuda":
            assert torch.cuda.current_device() == devices[-1].index
        return torch.nn.functional.cross_entropy(torch.nn.functional.linear(hidden, weight, bias), targets)

    wrapper.cross_entropy_backend = "liger_fused"
    wrapper._liger_loss = reference_loss
    ids = torch.tensor([[1, 2, 3, 4]], device=devices[0])
    labels = torch.tensor([[-100, -100, -100, 4]], device=devices[0])
    loss, count = wrapper.response_loss(ids, torch.ones_like(ids), torch.tensor([2], device=devices[0]), labels)
    loss.backward()
    prompt_gradient = wrapper.prompt_parameter.grad
    assert count == 1 and torch.isfinite(loss)
    assert prompt_gradient is not None and prompt_gradient.abs().sum() > 0
    assert all(parameter.grad is None for parameter in student.parameters())
