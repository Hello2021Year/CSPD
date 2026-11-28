"""Train one M2RL domain soft prompt with the paper recipe (Section 4.2, Appendix A).

The backbone (M2RL-SFT) stays frozen in BF16 and only a 256 x hidden FP32 prompt is
optimized. Defaults reproduce the released MCSI modules:

  * prefix-robust training: on half of the examples (chosen by a content hash) 128-256
    frozen tokens taken from other examples of the same domain are inserted before the
    prompt; omitted at inference
  * Muon (momentum 0.95, Nesterov, 5 Newton-Schulz steps, update RMS matched to AdamW)
  * cosine learning rate 0.03 -> 0.003 over 200 updates, no warmup, clip 1.0
  * global batch 64 responses, sequence-mean response-only loss, seed 42

Launch with one process per GPU, for example

  torchrun --nproc-per-node 4 experiments/m2rl/train_sp.py \
      --model Jackwang111/M2RL-SFT --data data/m2rl/math --output runs/m2rl/math

``--frozen-prompts`` trains a progressive prompt in front of earlier frozen prompts
(Figure 4), ``--no-prefix-robust`` and ``--optimizer adamw`` give the ablations in
Table 5. The packed training data format is produced by ``pack_data.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from soft_prompt_distillation.training import (
    SoftPromptModel,
    TrainingBatch,
    accumulate_response_loss,
    batch_indices_for_update,
    finish_prompt_update,
    initialize_prompt_embeddings,
)

MODEL_REVISION = "aa49d49a86b15ac920cf356c359f299edd0c1eb3"
PREFIX_POOL_SIZE, PREFIX_POOL_SEED = 2048, 7


# --------------------------------------------------------------------------- data

def load_packed(directory: Path):
    entries = json.loads((directory / "index.json").read_text())
    tokens = np.memmap(directory / "tokens.uint32", dtype="<u4", mode="r")
    return entries, tokens


def balanced_ranks(indices, entries, world):
    """Split one global batch across ranks, greedily balancing token counts."""
    groups = [[] for _ in range(world)]
    loads = [0] * world
    cap = len(indices) // world
    if len(indices) % world:
        raise ValueError("global batch must divide the number of ranks")
    for index in sorted(indices, key=lambda i: (-entries[i]["length"], i)):
        rank = min((r for r in range(world) if len(groups[r]) < cap), key=lambda r: (loads[r], r))
        groups[rank].append(index)
        loads[rank] += entries[index]["length"]
    return groups


def prefix_pool(entries, tokens):
    """Prompt-side token spans from up to 2,048 records."""
    rng = np.random.default_rng(PREFIX_POOL_SEED)
    pick = rng.choice(len(entries), size=min(PREFIX_POOL_SIZE, len(entries)), replace=False)
    pool = [
        np.array(tokens[entries[i]["offset"] + entries[i]["insert_at"]: entries[i]["offset"] + entries[i]["prefix_tokens"]], dtype=np.int64)
        for i in pick
    ]
    return [x for x in pool if len(x) >= 32]


# ------------------------------------------------------------------ prompt layout

class PromptLayout:
    """Builds the inserted block: [optional prefix F][frozen earlier prompts?][trainable prompt]."""

    def __init__(self, pool, *, prefix_robust: bool, frozen=None, frozen_after: bool = True):
        self.pool = pool
        self.prefix_robust = prefix_robust
        self.frozen = frozen
        self.frozen_after = frozen_after

    def install(self, wrapper: SoftPromptModel) -> None:
        layout = self
        wrapper.__class__ = type("LayoutSoftPromptModel", (SoftPromptModel,), {
            "prompt_length": property(lambda self: self._cur_block),
            "_inputs_with_prompt": lambda self, ids, mask, at: layout.inputs(self, ids, mask, at),
        })
        wrapper._cur_block = int(wrapper.prompt_parameter.size(0))

    def inputs(self, wrapper, input_ids, attention_mask, insert_at):
        embed = wrapper.base_model.get_input_embeddings()
        text = embed(input_ids)
        prompt = wrapper.prompt_parameter.to(dtype=text.dtype, device=text.device)
        if self.frozen is not None:
            fz = self.frozen.to(dtype=text.dtype, device=text.device)
            # Progressive training prepends the new prompt: [new][earlier][question].
            prompt = torch.cat([prompt, fz], dim=0) if self.frozen_after else torch.cat([fz, prompt], dim=0)
        assert input_ids.size(0) == 1, "micro batch 1"
        n = int(attention_mask[0].sum())
        h = int(hashlib.sha256(input_ids[0, :n].detach().cpu().numpy().tobytes()).hexdigest(), 16)
        if self.prefix_robust and h % 2:
            length = 128 + (h // 2) % 129
            ids = []
            j = (h // 512) % len(self.pool)
            while len(ids) < length:
                ids.extend(self.pool[j].tolist())
                j = (j + 1) % len(self.pool)
            pre = embed(torch.tensor(ids[:length], device=input_ids.device)).detach()
            block = torch.cat([pre, prompt], dim=0)
        else:
            block = prompt
        wrapper._cur_block = block.size(0)
        position = int(insert_at[0])
        x = torch.cat([text[0, :position], block, text[0, position:]], dim=0).unsqueeze(0)
        m = torch.cat([attention_mask[0, :position], attention_mask.new_ones((block.size(0),)), attention_mask[0, position:]]).unsqueeze(0)
        return x, m


# ---------------------------------------------------------------------- optimizer

def newton_schulz5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposed else X


class Muon(torch.optim.AdamW):
    """Muon on the single prompt matrix.

    Subclasses AdamW so checkpoints keep the ``step``/``exp_avg``/``exp_avg_sq`` layout;
    ``exp_avg`` holds the momentum and ``exp_avg_sq`` the last update.
    """

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                g = p.grad.float()
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = torch.tensor(0.0, dtype=torch.float32)
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1
                m = st["exp_avg"]
                m.mul_(0.95).add_(g)
                u = newton_schulz5(g.add(m, alpha=0.95)) * (0.2 * math.sqrt(max(p.shape)))
                st["exp_avg_sq"].copy_(u)
                p.add_(u, alpha=-lr)
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(p.data, src=0)
        return None


def make_optimizer(name, parameter):
    if name == "muon":
        return Muon([parameter], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    return torch.optim.AdamW([parameter], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


def cosine_lr(completed_updates, lr_max, lr_min, updates):
    """Update 1 uses lr_max and update ``updates`` uses lr_min."""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * completed_updates / (updates - 1)))


# --------------------------------------------------------------------------- main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Jackwang111/M2RL-SFT")
    p.add_argument("--revision", default=MODEL_REVISION)
    p.add_argument("--data", type=Path, required=True, help="packed domain directory (index.json, tokens.uint32)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt-length", type=int, default=256)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    p.add_argument("--lr-max", type=float, help="default 0.03 (Muon) or 0.01 (AdamW)")
    p.add_argument("--lr-min", type=float, help="default 0.003 (Muon) or 0.001 (AdamW)")
    p.add_argument("--no-prefix-robust", action="store_true")
    p.add_argument("--frozen-prompts", nargs="*", type=Path, default=[],
                   help="earlier state.pt files kept frozen behind the new prompt (progressive training)")
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--stop-after", type=int, help="stop after this many updates (smoke tests); schedule unchanged")
    args = p.parse_args(argv)
    scale = 1.0 if args.optimizer == "adamw" else 3.0
    args.lr_max = args.lr_max if args.lr_max is not None else 0.01 * scale
    args.lr_min = args.lr_min if args.lr_min is not None else 0.001 * scale
    return args


def main(argv=None, *, student_loader=None, student_placement=None):
    args = parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if distributed:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    entries, tokens = load_packed(args.data)
    pool = [] if args.no_prefix_robust else prefix_pool(entries, tokens)
    if not args.no_prefix_robust and not pool:
        raise ValueError("prefix-robust training needs at least one 32-token prefix span")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if student_loader is None:
        student = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=dtype,
                                                       attn_implementation="sdpa")
    else:
        student = student_loader(args.model, dtype)
    student = student.to(device) if student_placement is None else student_placement(student, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    init = SimpleNamespace(method="sampled_vocabulary", seed=42)
    prompt = initialize_prompt_embeddings(student, tokenizer, init, args.prompt_length)
    use_liger = device.type == "cuda"
    wrapper = SoftPromptModel(student, prompt, activation_checkpointing=device.type == "cuda",
                              cross_entropy_backend="liger_fused" if use_liger else "torch",
                              loss_normalization="sequence")
    frozen = None
    if args.frozen_prompts:
        frozen = torch.cat([torch.load(p, map_location="cpu", weights_only=True)["prompt"].float()
                            for p in args.frozen_prompts]).to(device).requires_grad_(False)
    PromptLayout(pool, prefix_robust=not args.no_prefix_robust, frozen=frozen).install(wrapper)
    optimizer = make_optimizer(args.optimizer, wrapper.prompt_parameter)

    ckpt_dir = args.output / "checkpoints"
    start = 1
    existing = sorted(ckpt_dir.glob("update-*")) if ckpt_dir.exists() else []
    if existing:
        state = torch.load(existing[-1] / "state.pt", map_location="cpu", weights_only=False)
        with torch.no_grad():
            wrapper.prompt_parameter.copy_(state["prompt"].to(device))
        optimizer.load_state_dict(state["optimizer"])
        start = state["update"] + 1
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    last = args.updates if args.stop_after is None else min(args.updates, start + args.stop_after - 1)
    for update in range(start, last + 1):
        global_indices = batch_indices_for_update(dataset_size=len(entries), effective_batch_size=args.batch_size,
                                                  batch_seed=args.seed, update=update)
        local = balanced_ranks(global_indices, entries, world)[rank]
        optimizer.zero_grad(set_to_none=True)
        loss_sum, response_tokens = 0.0, 0
        for index in local:
            e = entries[index]
            ids = torch.tensor(np.asarray(tokens[e["offset"]:e["offset"] + e["length"]], dtype=np.int64), device=device).unsqueeze(0)
            labels = ids.clone()
            labels[:, :e["prefix_tokens"]] = -100
            batch = TrainingBatch(ids, torch.ones_like(ids), labels, torch.tensor([e["insert_at"]], device=device))
            value, count = accumulate_response_loss(wrapper, batch, global_loss_denominator=len(global_indices))
            loss_sum += value
            response_tokens += count
        completed = int(optimizer.state.get(wrapper.prompt_parameter, {}).get("step", 0))
        lr = cosine_lr(completed, args.lr_max, args.lr_min, args.updates)
        result = finish_prompt_update(wrapper, optimizer, local_loss_sum=loss_sum, local_response_tokens=response_tokens,
                                      local_loss_denominator=len(local), learning_rate=lr, max_grad_norm=1.0)
        if rank == 0:
            event = dict(update=update, lr=lr, loss=result.loss, response_tokens=result.response_tokens)
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            if update % args.save_every == 0 or update == last:
                path = ckpt_dir / f"update-{update:04d}"
                path.mkdir(parents=True, exist_ok=True)
                torch.save({"prompt": wrapper.prompt_parameter.detach().cpu(), "optimizer": optimizer.state_dict(),
                            "update": update}, path / "state.pt")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
