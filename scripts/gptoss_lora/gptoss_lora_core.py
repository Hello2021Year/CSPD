"""K/V LoRA with FP32 master adapters and BF16 computation; no virtual tokens."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


class KVLoRA(nn.Module):
    def __init__(self, base, rank=5, alpha=5):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        delta = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return self.base(x) + delta * self.scale


def install_lora(model, rank=5, alpha=5, seed=42):
    model.requires_grad_(False)
    torch.manual_seed(seed)
    targets = [(name, module) for name, module in model.named_modules()
               if name.endswith(('.k_proj', '.v_proj')) and isinstance(module, nn.Linear)]
    assert targets
    for name, module in targets:
        parent, leaf = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), leaf, KVLoRA(module, rank, alpha))
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def adapter_state(model):
    return {name: p.detach().cpu().contiguous() for name, p in model.named_parameters() if p.requires_grad}


def restore_adapter(model, state):
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert params.keys() == state.keys()
    with torch.no_grad():
        for name, p in params.items():
            assert p.shape == state[name].shape
            p.copy_(state[name])


def merge_lora(model):
    for name, layer in list(model.named_modules()):
        if isinstance(layer, KVLoRA):
            with torch.no_grad():
                layer.base.weight.copy_((layer.base.weight.float() + layer.scale * (layer.lora_B @ layer.lora_A)).to(layer.base.weight.dtype))
            parent, leaf = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), leaf, layer.base)


def response_loss(model, batch, loss_fn):
    ids, mask, labels = batch
    hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
    targets = labels[:, 1:]
    selected = targets.ne(-100)
    count = int(selected.sum())
    assert count > 0
    head = model.get_output_embeddings()
    loss = loss_fn(head.weight, hidden[:, :-1][selected], targets[selected], bias=getattr(head, 'bias', None))
    return loss, count


def collate(rows, pad_id, device):
    longest = max(len(prompt) + len(response) for prompt, response in rows)
    ids = torch.full((len(rows), longest), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, (prompt, response) in enumerate(rows):
        tokens = torch.as_tensor(__import__('numpy').concatenate((prompt, response)).astype('int64'))
        n, p = len(tokens), len(prompt)
        ids[i, :n] = tokens
        labels[i, p:n] = tokens[p:]
        mask[i, :n] = 1
    return tuple(x.to(device) for x in (ids, mask, labels))


class TokenData:
    def __init__(self, root):
        import numpy as np
        self.root = Path(root)
        self.index = np.load(self.root / 'index.npy', mmap_mode='r')
        self.tokens = np.memmap(self.root / 'tokens.i32', dtype='int32', mode='r')
        self.identity = json.loads((self.root / 'identity.json').read_text())

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        offset, prompt, response = (int(x) for x in self.index[i])
        return self.tokens[offset:offset+prompt], self.tokens[offset+prompt:offset+prompt+response]

    def length(self, i):
        return int(self.index[i, 1] + self.index[i, 2])


def microbatches(indices, data, micro):
    # Reorder only inside each rank's fixed global update; no resampling/drop.
    ordered = sorted(indices, key=data.length)
    return [ordered[i:i+micro] for i in range(0, len(ordered), micro)]
