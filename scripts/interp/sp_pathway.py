#!/usr/bin/env python3
"""Attention-knockout analysis of how the soft prompt reaches the response.

Appendix G attention pathways.  Every condition is one teacher-forced
student forward with the soft prompt spliced into ``inputs_embeds`` as in
``training.py``.  A registered attention function hides chosen
(layer, query group, head) -> SP entries from the softmax while keeping
GPT-OSS's sinks and sliding windows, and can split each head's output into the
part read from SP and the rest (to record attention mass on SP, or to replace
the SP part by its mean / a shuffle over response positions).  The kv set swaps
the keys and values that chosen heads read at SP positions between the main and
a donor checkpoint, with the key's effect split into total SP mass and the
pattern within SP.

Per response token and condition we keep KL(T||cond), the log-prob of the
realised token and a few decision-token log-probs; ``analyze_sp_pathway.py``
turns them into recovery fractions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import flex_attention as hf_flex

# Every condition builds a new block mask (a fresh mask_mod closure), and each one
# recompiles the compiled flex kernel.  Past dynamo's default limit of 8, flex
# silently falls back to the unfused implementation with different numerics.
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.accumulated_recompile_limit = 4096

GROUPS = ("PRE", "SP", "SYS", "Q", "H", "R")
PRE, SP, SYS, Q, H, R = range(len(GROUPS))
ROLE_GROUPS = {"system": SYS, "developer": SYS, "user": Q, "assistant": H}
PROMPT_SIDE = (SYS, Q, H)
PATHS = {"direct": (R,), "indirect": PROMPT_SIDE}
# who reads the swapped SP keys/values: R alone with prompt->SP blocked, or every non-SP query
KV_PATHS = {"direct": (R,), "open": PROMPT_SIDE + (R,)}
# (mass key, pattern key, value) sources: t main, 0 donor, n donor direction at main row norms,
# r random direction at main row norms, p main values permuted over SP positions
KV_COMBOS = ("000", "tt0", "00t", "ttt", "t00", "0t0", "t0t", "0tt",
             "nn0", "00n", "rrt", "ttr", "ttp", "rrr")
CHUNK = 2048
WATCH_TOKENS = ("<|end|>", " But", " So", " Thus", "But", "Thus", "Check", "?")
ATTN_NAME = "sp_knockout"
CONDITION_SETS = ("core", "layers", "heads", "readout", "kv")


# --------------------------------------------------------------------------- conditions


@dataclass(frozen=True)
class Condition:
    name: str
    soft: str | None
    # (layers, query groups, query heads or None for all) whose attention to SP keys is removed
    blocks: tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...] | None], ...] = ()
    # (layer, mode) with mode in mean | shuffle | identity, applied to R rows' SP read-out
    readout: tuple[tuple[int, str], ...] = ()
    # (layers, kv heads or None for all, reader groups, combo): SP keys/values those heads'
    # reader rows see, per KV_COMBOS
    kv: tuple[tuple[tuple[int, ...], tuple[int, ...] | None, tuple[int, ...], str], ...] = ()


def parse_kv_target(target, layer_types):
    """'full' (all full layers), 'all', 'l5' or 'l5g0' -> (layers, kv heads or None)."""
    if target == "full":
        return tuple(i for i, t in enumerate(layer_types) if t == "full_attention"), None
    if target == "all":
        return tuple(range(len(layer_types))), None
    m = re.fullmatch(r"l(\d+)(?:g(\d+))?", target)
    if not m:
        raise ValueError(f"bad kv target {target}")
    return (int(m[1]),), None if m[2] is None else (int(m[2]),)


def build_conditions(sets, layer_types, n_heads, n_kv, main, control, key_layers, paths,
                     kv_targets=(), kv_paths=("direct",), kv_combos=KV_COMBOS):
    L = len(layer_types)
    every = tuple(range(L))
    full = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    sliding = tuple(i for i, t in enumerate(layer_types) if t != "full_attention")
    out = [Condition("S", None)]

    def core(sp):
        return [
            Condition(f"{sp}.C0", sp),
            Condition(f"{sp}.C_all", sp, ((every, PROMPT_SIDE + (R,), None),)),
            Condition(f"{sp}.C_direct", sp, ((every, PROMPT_SIDE, None),)),
            Condition(f"{sp}.C_indirect", sp, ((every, (R,), None),)),
            Condition(f"{sp}.C_ind_sys", sp, ((every, (R, Q, H), None),)),
            Condition(f"{sp}.C_ind_q", sp, ((every, (R, SYS, H), None),)),
            Condition(f"{sp}.C_win", sp, ((sliding, PROMPT_SIDE + (R,), None),)),
            Condition(f"{sp}.C_full", sp, ((full, PROMPT_SIDE + (R,), None),)),
        ]

    out += core(main)
    if "core" in sets and control:
        out += core(control)
    if "core" not in sets:
        out = out[:2]  # S and main C0 are always needed as reference points

    group = n_heads // n_kv
    for path in paths:
        qp = PATHS[path]
        other = PATHS["indirect" if path == "direct" else "direct"]
        base = ((every, other, None),)  # isolate the scanned path
        # R reads SP directly only where it is inside the window, i.e. in full layers
        scan = full if path == "direct" else every
        if "layers" in sets:
            for l in scan:
                out.append(Condition(f"{main}.{path}.drop_l{l}", main, base + (((l,), qp, None),)))
                rest = tuple(i for i in every if i != l)
                out.append(Condition(f"{main}.{path}.only_l{l}", main, base + ((rest, qp, None),)))
            for l in range(1, L):
                out.append(Condition(f"{main}.{path}.upto_l{l}", main,
                                     base + ((tuple(range(l)), qp, None),)))
                out.append(Condition(f"{main}.{path}.from_l{l}", main,
                                     base + ((tuple(range(l, L)), qp, None),)))
        if "heads" in sets:
            for l in key_layers:
                for g in range(n_kv):
                    heads = tuple(range(g * group, (g + 1) * group))
                    out.append(Condition(f"{main}.{path}.head_l{l}_g{g}", main,
                                         base + (((l,), qp, heads),)))
    if "readout" in sets:
        base = ((every, PATHS["indirect"], None),)
        for l in key_layers:
            for mode in ("mean", "shuffle"):
                out.append(Condition(f"{main}.direct.read_{mode}_l{l}", main, base, ((l, mode),)))
    if "kv" in sets:
        for path in kv_paths:
            readers = KV_PATHS[path]
            base = ((every, PROMPT_SIDE, None),) if path == "direct" else ()
            for target in kv_targets:
                layers, kv_heads = parse_kv_target(target, layer_types)
                heads = None if kv_heads is None else tuple(
                    h for g in kv_heads for h in range(g * group, (g + 1) * group))
                out.append(Condition(f"{main}.{path}.kv_{target}_blk", main,
                                     base + ((layers, readers, heads),)))
                for combo in kv_combos:
                    out.append(Condition(f"{main}.{path}.kv_{target}_{combo}", main, base,
                                         kv=((layers, kv_heads, readers, combo),)))
    return out


def blocked_tensor(cond: Condition, n_layers: int, n_heads: int, device) -> torch.Tensor | None:
    """[L, heads, groups] bool: query group of that head may not attend to SP keys."""
    if not cond.blocks:
        return None
    blk = torch.zeros(n_layers, n_heads, len(GROUPS), dtype=torch.bool)
    for layers, qgroups, heads in cond.blocks:
        hs = list(heads) if heads is not None else list(range(n_heads))
        for l in layers:
            for g in qgroups:
                blk[l, hs, g] = True
    return blk.to(device)


# --------------------------------------------------------------------------- attention


class KnockoutState:
    """What the registered attention function reads during one forward."""

    def __init__(self, backend: str):
        self.backend = backend
        self.groups: torch.Tensor | None = None
        self.r_rows: torch.Tensor | None = None
        self.blocked: torch.Tensor | None = None
        self.readout: dict[int, str] = {}
        self.mass_layers: set[int] = set()
        self.mass: dict[int, torch.Tensor] = {}
        self.perm: torch.Tensor | None = None
        self.record_router = False
        self.router: dict[int, torch.Tensor] = {}
        self.sp_cols: torch.Tensor | None = None
        self.kv: dict[int, tuple] = {}
        self.kv_seed = 0
        self.kv_mass: dict[int, torch.Tensor] = {}
        self.capture_layers: set[int] = set()
        self.captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._masks: dict = {}

    def set_sequence(self, groups: torch.Tensor, seed: int) -> None:
        self.groups = groups
        self.r_rows = (groups == R).nonzero()[:, 0]
        self.sp_cols = (groups == SP).nonzero()[:, 0]
        gen = torch.Generator().manual_seed(seed)
        self.perm = torch.randperm(len(self.r_rows), generator=gen).to(groups.device)
        self._masks = {}

    def set_condition(self, cond: Condition, blocked, mass_layers=(), record_router=False,
                      kv_seed=0, capture_layers=()):
        self.blocked = blocked
        self.readout = dict(cond.readout)
        self.kv = {}
        for layers, kv_heads, readers, combo in cond.kv:
            for l in layers:
                if l in self.kv or l in self.readout:
                    raise ValueError(f"layer {l} has more than one kv swap or readout")
                self.kv[l] = (kv_heads, readers, combo)
        self.kv_seed = kv_seed
        self.kv_mass = {}
        self.capture_layers = set(capture_layers)
        self.mass_layers = set(mass_layers)
        self.mass = {}
        self.record_router = record_router
        self.router = {}


STATE = KnockoutState("flex")


def _block_mask(groups, blk, window, sp_only, n_heads, T, device):
    key = (window, sp_only, None if blk is None else blk.cpu().numpy().tobytes())
    if key in STATE._masks:
        return STATE._masks[key]
    uniform = blk is None or bool((blk == blk[:1]).all())
    is_sp = groups == SP

    def mask_mod(b, h, q, k):
        ok = k <= q
        if window is not None:
            ok = ok & (q - k < window)
        if blk is not None:
            ok = ok & ~(blk[h, groups[q]] & is_sp[k])
        if sp_only:
            ok = ok & is_sp[k]
        return ok

    mask = create_block_mask(mask_mod, B=None, H=None if uniform else n_heads,
                             Q_LEN=T, KV_LEN=T, device=device)
    STATE._masks[key] = mask
    return mask


def _flex(query, key, value, block_mask, scaling):
    out = hf_flex.compile_friendly_flex_attention(
        query, key, value, score_mod=None, block_mask=block_mask, enable_gqa=True,
        scale=scaling, training=False, **hf_flex.get_flex_attention_lse_kwargs(True))
    if hf_flex._TORCH_FLEX_USE_AUX:
        out, aux = out
        return out, aux.lse
    return out


def _attend_flex(module, query, key, value, scaling, window, sinks, blk, split):
    T, Hq = query.shape[2], query.shape[1]
    groups = STATE.groups
    out, lse = _flex(query, key, value,
                     _block_mask(groups, blk, window, False, Hq, T, query.device), scaling)
    # sinks exactly as transformers.integrations.flex_attention.flex_attention_forward
    lse_v = lse.to(value.dtype).unsqueeze(-1)
    sink_v = sinks.view(1, -1, 1, 1).expand(1, Hq, T, 1)
    combined = torch.logsumexp(torch.cat([lse_v, sink_v], dim=-1), dim=-1, keepdim=True)
    out = (out * torch.exp(lse_v - combined)).to(query.dtype)
    z = combined[..., 0]  # log of the softmax normaliser, sink included
    if not split:
        return out, None, None, z
    out_sp, lse_sp = _flex(query, key, value,
                           _block_mask(groups, blk, window, True, Hq, T, query.device), scaling)
    total = torch.logsumexp(torch.stack([lse.float(), sinks.float().view(1, -1, 1).expand_as(lse)]), 0)
    share = torch.exp(lse_sp.float() - total)
    share = torch.where(torch.isfinite(lse_sp), share, torch.zeros_like(share))
    sp_part = torch.nan_to_num(out_sp.float()) * share.unsqueeze(-1)
    return out, sp_part, share, z


def _attend_eager(module, query, key, value, scaling, window, sinks, blk, split, chunk=512):
    """Dense fp32 reference with the same masking; also a fallback where flex is unavailable."""
    T, Hq = query.shape[2], query.shape[1]
    groups = STATE.groups
    rep = Hq // key.shape[1]
    k = key[0].repeat_interleave(rep, 0).float()
    v = value[0].repeat_interleave(rep, 0).float()
    kpos = torch.arange(T, device=query.device)
    is_sp = groups == SP
    outs, parts, shares, zs = [], [], [], []
    for s in range(0, T, chunk):
        qpos = kpos[s:s + chunk]
        allowed = kpos[None] <= qpos[:, None]
        if window is not None:
            allowed = allowed & (qpos[:, None] - kpos[None] < window)
        allowed = allowed[None].expand(Hq, -1, -1)
        if blk is not None:
            allowed = allowed & ~(blk[:, groups[qpos]][:, :, None] & is_sp[None, None])
        logits = (query[0, :, qpos].float() @ k.transpose(-1, -2)) * scaling
        logits = logits.masked_fill(~allowed, float("-inf"))
        sink = sinks.float().view(-1, 1, 1).expand(Hq, len(qpos), 1)
        full = torch.cat([logits, sink], -1)
        zs.append(torch.logsumexp(full, -1))
        probs = torch.softmax(full, -1)[..., :-1]
        outs.append(probs @ v)
        if split:
            p_sp = probs * is_sp
            parts.append(p_sp @ v)
            shares.append(p_sp.sum(-1))
    out = torch.cat(outs, 1)[None].to(query.dtype)
    z = torch.cat(zs, 1)[None]
    if not split:
        return out, None, None, z
    return out, torch.cat(parts, 1)[None], torch.cat(shares, 1)[None], z


def kv_sources(k_t, v_t, k_0, v_0, combo, gen):
    """Mass keys, pattern keys and values, each [kv heads, n_sp, head_dim], per KV_COMBOS."""

    def pick(c, t, d):
        if c == "t":
            return t
        if c == "0":
            return d
        if c == "n":
            return d * (t.norm(dim=-1, keepdim=True) / d.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        if c == "r":
            x = torch.randn(t.shape, generator=gen).to(t.device)
            return x * (t.norm(dim=-1, keepdim=True) / x.norm(dim=-1, keepdim=True))
        if c == "p":
            return t[:, torch.randperm(t.shape[1], generator=gen).to(t.device)]
        raise ValueError(f"unknown kv source {c}")

    keys = {c: pick(c, k_t, k_0) for c in sorted(set(combo[:2]))}  # same draw for mass and pattern
    return keys[combo[0]], keys[combo[1]], pick(combo[2], v_t, v_0)


def swap_kv(attend, module, query, key, value, scaling, window, sinks, blk, out, layer):
    """Recompute the swapped heads' reader rows with substituted SP keys / values.

    The non-SP part (sink included) comes from the same attention with SP hidden from those
    rows; its normaliser z_rest and the SP keys' normaliser z_sp give the SP mass
    m = sigmoid(z_sp - z_rest), so a row's output is (1 - m) * rest + m * (pi @ V).  m is
    computed from the mass keys and pi from the pattern keys, which may differ.
    """
    kv_heads, readers, combo = STATE.kv[layer]
    Hq, n_kv = query.shape[1], key.shape[1]
    rep = Hq // n_kv
    dev = query.device
    groups, sp = STATE.groups, STATE.sp_cols
    kv_heads = range(n_kv) if kv_heads is None else kv_heads
    heads = torch.tensor([h for g in kv_heads for h in range(g * rep, (g + 1) * rep)], device=dev)
    rows = torch.isin(groups, torch.tensor(readers, device=dev)).nonzero()[:, 0]
    blk_rest = (torch.zeros(Hq, len(GROUPS), dtype=torch.bool, device=dev) if blk is None
                else blk.clone())
    blk_rest[heads[:, None], torch.tensor(readers, device=dev)[None]] = True
    rest, _, _, z_rest = attend(module, query, key, value, scaling, window, sinks, blk_rest, False)

    gen = torch.Generator().manual_seed(STATE.kv_seed * 1009 + layer)
    k_0, v_0 = STATE.captured[layer]
    k_m, k_p, v = kv_sources(key[0][:, sp].float(), value[0][:, sp].float(),
                             k_0.float(), v_0.float(), combo, gen)
    g_of = heads // rep
    visible = (rows[:, None] - sp[None] < window) if window is not None else \
        torch.ones(len(rows), len(sp), dtype=torch.bool, device=dev)
    visible = visible[None].expand(len(heads), -1, -1)
    if blk is not None:
        visible = visible & ~blk[heads][:, groups[rows]][:, :, None]
    q = query[0][heads][:, rows].float()
    logit_m = (q @ k_m[g_of].transpose(-1, -2) * scaling).masked_fill(~visible, float("-inf"))
    logit_p = (q @ k_p[g_of].transpose(-1, -2) * scaling).masked_fill(~visible, float("-inf"))
    m = torch.sigmoid(torch.logsumexp(logit_m, -1) - z_rest[0][heads][:, rows].float())
    read = torch.nan_to_num(torch.softmax(logit_p, -1)) @ v[g_of]
    new = (1 - m)[..., None] * rest[0][heads][:, rows].float() + m[..., None] * read
    out = out.clone()
    out[0, heads[:, None], rows[None]] = new.to(out.dtype)
    STATE.kv_mass[layer] = m[:, groups[rows] == R].mean(0).cpu()
    return out


def knockout_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0,
                       sliding_window=None, s_aux=None, **kwargs):
    layer = module.layer_idx
    blk = None if STATE.blocked is None else STATE.blocked[layer]
    if blk is not None and not bool(blk.any()):
        blk = None
    split = layer in STATE.readout or layer in STATE.mass_layers
    attend = _attend_eager if STATE.backend == "eager" else _attend_flex
    out, sp_part, share, _ = attend(module, query, key, value, scaling, sliding_window,
                                    s_aux, blk, split)
    if layer in STATE.capture_layers:
        STATE.captured[layer] = (key[0][:, STATE.sp_cols].clone(),
                                 value[0][:, STATE.sp_cols].clone())
    if layer in STATE.kv:
        out = swap_kv(attend, module, query, key, value, scaling, sliding_window, s_aux, blk,
                      out, layer)
    rows = STATE.r_rows
    if layer in STATE.mass_layers:
        STATE.mass[layer] = share[0][:, rows].half().cpu()
    if layer in STATE.readout:
        mode = STATE.readout[layer]
        part = sp_part[:, :, rows]
        if mode == "mean":
            new = part.mean(dim=2, keepdim=True).expand_as(part)
        elif mode == "shuffle":
            new = part[:, :, STATE.perm]
        elif mode == "identity":
            new = part
        else:
            raise ValueError(f"unknown readout mode {mode}")
        out = out.clone()
        out[:, :, rows] = (out[:, :, rows].float() - part + new).to(out.dtype)
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register(ATTN_NAME, knockout_attention)


def attach_router_hooks(model):
    def hook(layer):
        def fn(module, inputs, output):
            if STATE.record_router:
                STATE.router[layer] = output[2][STATE.r_rows].clone()
        return fn

    for i, block in enumerate(model.model.layers):
        block.mlp.router.register_forward_hook(hook(i))


def router_overlap(a: dict, b: dict, n_layers: int) -> np.ndarray:
    out = np.full(n_layers, np.nan, dtype=np.float32)
    for l in range(n_layers):
        if l in a and l in b:
            same = (a[l][:, :, None] == b[l][:, None, :]).any(-1).float().mean()
            out[l] = float(same)
    return out


# --------------------------------------------------------------------------- data


def label_groups(prompt_ids, insert_at, n_soft, n_resp, start_id, role_ids):
    """Group id of every position of the spliced sequence (text order, SP inserted)."""
    text = []
    group = None
    for i, tok in enumerate(prompt_ids):
        if tok == start_id:
            role = prompt_ids[i + 1] if i + 1 < len(prompt_ids) else None
            if role not in role_ids:
                raise ValueError(f"unknown role token {role} after <|start|> at {i}")
            group = role_ids[role]
        if group is None:
            raise ValueError("prompt does not start with <|start|>")
        text.append(PRE if i < insert_at else group)
    return torch.tensor(text[:insert_at] + [SP] * n_soft + text[insert_at:] + [R] * n_resp)


def final_channel_start(response_ids, final_seq) -> int:
    """Index of the first response token after `<|channel|>final<|message|>` (len if absent)."""
    n = len(final_seq)
    for i in range(len(response_ids) - n + 1):
        if list(response_ids[i:i + n]) == final_seq:
            return i + n
    return len(response_ids)


def select_traces(paths, quotas, seed):
    rows = []
    for path in paths:
        rows += [json.loads(line) for line in path.open()]
    rng = random.Random(seed)
    picked = []
    for source, n in quotas.items():
        pool = [r for r in rows if r["source"] == source]
        if not pool:
            raise ValueError(f"no traces with source {source}")
        by_set: dict[str, list] = {}
        for r in pool:
            by_set.setdefault(r["dataset"], []).append(r)
        for v in by_set.values():
            rng.shuffle(v)
        order = []
        for i in range(max(len(v) for v in by_set.values())):  # round-robin over datasets
            order += [v[i] for _, v in sorted(by_set.items()) if i < len(v)]
        picked += order[:n]
    return picked


def trace_key(tr: dict) -> str:
    return f"{tr['source']}__{tr['question_id'].replace('/', '_')}__{tr['sample']}"


def load_prompt(path: Path) -> torch.Tensor:
    with safe_open(str(path), framework="pt") as handle:
        return handle.get_tensor("prompt_embeddings")


def load_teacher_head(teacher_dir: Path, device) -> torch.Tensor:
    index_path = teacher_dir / "model.safetensors.index.json"
    if index_path.exists():
        shard = json.loads(index_path.read_text())["weight_map"]["lm_head.weight"]
    else:
        shard = "model.safetensors"
    with safe_open(str(teacher_dir / shard), framework="pt") as h:
        return h.get_tensor("lm_head.weight").to(device, torch.bfloat16)


# --------------------------------------------------------------------------- scoring


@torch.no_grad()
def response_hidden(model, embed_text, insert_at, soft, start, n_resp):
    if soft is not None:
        embed = torch.cat([embed_text[:, :insert_at], soft[None], embed_text[:, insert_at:]], dim=1)
    else:
        embed = embed_text
    out = model.model(inputs_embeds=embed, use_cache=False)
    return out.last_hidden_state[0, start:start + n_resp]


@torch.no_grad()
def token_stats(hidden, head, target, teacher_lp, watch_ids):
    kl, logp, watch = [], [], []
    for n, s in enumerate(range(0, len(target), CHUNK)):
        lp = F.log_softmax((hidden[s:s + CHUNK] @ head.T).float(), dim=-1)
        lp_t = teacher_lp[n]
        kl.append((lp_t.exp() * (lp_t - lp)).sum(-1))
        logp.append(lp.gather(1, target[s:s + CHUNK, None])[:, 0])
        watch.append(lp[:, watch_ids])
    return {"kl": torch.cat(kl), "logp": torch.cat(logp), "watch": torch.cat(watch)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--select", nargs="+", default=["teacher=40", "student_sp=20"],
                        help="source=count quotas, round-robin over datasets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--student-dir", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True,
                        help="teacher snapshot; its lm_head is read, and the full model is "
                             "loaded only when --teacher-hidden-cache lacks a trace")
    parser.add_argument("--teacher-hidden-cache", type=Path, default=None)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path pairs")
    parser.add_argument("--main", default="u180")
    parser.add_argument("--control", default="u0", help="checkpoint run on the core set ('' = none)")
    parser.add_argument("--sets", nargs="+", default=["core"], choices=CONDITION_SETS)
    parser.add_argument("--paths", nargs="+", default=["direct", "indirect"], choices=list(PATHS))
    parser.add_argument("--key-layers", type=int, nargs="*", default=[],
                        help="layers for the heads / readout sets")
    parser.add_argument("--kv-targets", nargs="*", default=[],
                        help="kv set: where to swap SP keys/values, e.g. l5g0 l5 full")
    parser.add_argument("--kv-paths", nargs="+", default=["direct"], choices=list(KV_PATHS))
    parser.add_argument("--kv-combos", nargs="+", default=list(KV_COMBOS), choices=KV_COMBOS)
    parser.add_argument("--kv-donor", default="u0", help="checkpoint the 0 / n sources come from")
    parser.add_argument("--max-response", type=int, default=4096)
    parser.add_argument("--backend", default="flex", choices=["flex", "eager"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--list-conditions", action="store_true")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    STATE.backend = args.backend

    tokenizer = AutoTokenizer.from_pretrained(str(args.student_dir))
    student = AutoModelForCausalLM.from_pretrained(
        str(args.student_dir), dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=ATTN_NAME).to(args.device).eval()
    cfg = student.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    full_layers = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    conditions = build_conditions(set(args.sets), cfg.layer_types, n_heads,
                                  cfg.num_key_value_heads, args.main, args.control or None,
                                  args.key_layers, args.paths, args.kv_targets, args.kv_paths,
                                  args.kv_combos)
    if ({"heads", "readout"} & set(args.sets)) and not args.key_layers:
        parser.error("--key-layers is required for the heads and readout sets")
    if "kv" in args.sets and not args.kv_targets:
        parser.error("--kv-targets is required for the kv set")
    kv_layers = sorted({l for c in conditions for spec in c.kv for l in spec[0]})
    print(f"{len(conditions)} conditions", flush=True)
    if args.list_conditions:
        for c in conditions:
            print(c.name)
        return
    attach_router_hooks(student)

    soft = {k: load_prompt(Path(v)).to(args.device, torch.bfloat16)
            for k, v in (item.split("=", 1) for item in args.checkpoints)}
    n_soft = soft[args.main].shape[0]
    start_id = tokenizer.convert_tokens_to_ids("<|start|>")
    role_ids = {}
    for role, g in ROLE_GROUPS.items():
        ids = tokenizer.encode(role, add_special_tokens=False)
        assert len(ids) == 1, role
        role_ids[ids[0]] = g
    final_seq = tokenizer.encode("<|channel|>final<|message|>", add_special_tokens=False)
    watch = {t: tokenizer.encode(t, add_special_tokens=False) for t in WATCH_TOKENS}
    watch = {t: ids[0] for t, ids in watch.items() if len(ids) == 1}
    watch_ids = list(watch.values())

    quotas = dict((s.split("=")[0], int(s.split("=")[1])) for s in args.select)
    traces = select_traces(args.traces, quotas, args.seed)[args.shard::args.num_shards]
    head_s = student.lm_head.weight
    head_t = load_teacher_head(args.teacher_dir, args.device)
    teacher = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "conditions.json").write_text(json.dumps(
        [{"name": c.name, "soft": c.soft, "blocks": c.blocks, "readout": c.readout, "kv": c.kv}
         for c in conditions]))
    if kv_layers and args.kv_donor not in soft:
        parser.error(f"--kv-donor {args.kv_donor} is not among --checkpoints")
    window = cfg.sliding_window
    t0 = time.time()
    for n, tr in enumerate(traces):
        out_path = args.output_dir / f"{trace_key(tr)}.npz"
        if out_path.exists():
            continue
        p_ids = list(tr["prompt_token_ids"])
        r_ids = list(tr["response_token_ids"])[: args.max_response]
        ins = int(tr["soft_prompt_insert_at"])
        n_resp = len(r_ids)
        dev = args.device

        # teacher log-probs at the positions predicting each response token
        cached = None
        if args.teacher_hidden_cache is not None:
            path = args.teacher_hidden_cache / f"{trace_key(tr)}.pt"
            if path.exists():
                cached = torch.load(path)[:n_resp]
        if cached is None:
            if teacher is None:
                teacher = AutoModelForCausalLM.from_pretrained(
                    str(args.teacher_dir), dtype=torch.bfloat16, low_cpu_mem_usage=True,
                    attn_implementation="flex_attention" if args.backend == "flex" else "eager",
                ).to(args.teacher_device).eval()
            ids = torch.tensor([p_ids + r_ids], device=args.teacher_device)
            hid = teacher.model(input_ids=ids, use_cache=False).last_hidden_state[0]
            cached = hid[len(p_ids) - 1:len(p_ids) - 1 + n_resp]
        h_t = cached.to(dev, torch.bfloat16)
        teacher_lp = [F.log_softmax((h_t[s:s + CHUNK] @ head_t.T).float(), dim=-1)
                      for s in range(0, n_resp, CHUNK)]
        target = torch.tensor(r_ids, device=dev)
        embed_text = student.get_input_embeddings()(torch.tensor([p_ids + r_ids], device=dev))

        groups_sp = label_groups(p_ids, ins, n_soft, n_resp, start_id, role_ids).to(dev)
        groups_s = torch.cat([groups_sp[:ins], groups_sp[ins + n_soft:]])
        seed = args.seed * 1_000_003 + n
        result: dict[str, np.ndarray] = {}
        routers = {}
        if kv_layers:
            # donor SP keys/values, from an unblocked forward of the same sequence
            STATE.set_sequence(groups_sp, seed)
            STATE.set_condition(Condition("capture", args.kv_donor), None,
                                capture_layers=kv_layers)
            response_hidden(student, embed_text, ins, soft[args.kv_donor], len(p_ids) + n_soft - 1, 1)
            STATE.capture_layers = set()
        for i, cond in enumerate(conditions):
            with_sp = cond.soft is not None
            STATE.set_sequence(groups_sp if with_sp else groups_s, seed)
            is_c0 = cond.name.endswith(".C0")
            STATE.set_condition(cond, blocked_tensor(cond, n_layers, n_heads, dev),
                                mass_layers=full_layers if is_c0 else (), record_router=True,
                                kv_seed=seed * 1009 + i)
            start = len(p_ids) + (n_soft if with_sp else 0) - 1
            hidden = response_hidden(student, embed_text, ins, soft[cond.soft] if with_sp else None,
                                     start, n_resp)
            stats = token_stats(hidden, head_s, target, teacher_lp, watch_ids)
            for key, val in stats.items():
                # recovery takes differences of sums, so keep KL and log-probs in fp32
                val = val.half() if key == "watch" else val.float()
                result[f"{cond.name}__{key}"] = val.cpu().numpy()
            if is_c0:
                result[f"{cond.name}__sp_mass"] = torch.stack(
                    [STATE.mass[l] for l in full_layers]).numpy()
            if cond.kv:
                # SP mass of the swapped heads at R rows, averaged over swapped layers
                result[f"{cond.name}__kv_mass"] = torch.stack(
                    list(STATE.kv_mass.values())).mean(0).half().numpy()
            routers[cond.name] = dict(STATE.router)
        # expert choices at R positions, compared with the main C0
        for name, rt in routers.items():
            result[f"{name}__router_overlap"] = router_overlap(
                rt, routers[f"{args.main}.C0"], n_layers)
        result["T__logp"] = torch.cat([
            lp.gather(1, target[s:s + CHUNK, None])[:, 0]
            for lp, s in zip(teacher_lp, range(0, n_resp, CHUNK))]).half().cpu().numpy()
        np.savez(out_path, **result)
        r_start = len(p_ids) + n_soft
        last_sp = ins + n_soft - 1
        meta = {k: tr.get(k) for k in ("source", "dataset", "question_id", "sample", "correct",
                                         "truncated", "level")}
        meta |= {
            "n_response": n_resp,
            "final_start": final_channel_start(r_ids, final_seq),
            "group_counts": {g: int((groups_sp == i).sum()) for i, g in enumerate(GROUPS)},
            # response tokens that still see SP inside a sliding window
            "r_in_window": max(0, min(n_resp, last_sp + window - r_start)),
            "watch_tokens": watch,
            "full_layers": full_layers,
        }
        out_path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False))
        elapsed = time.time() - t0
        print(f"[{n + 1}/{len(traces)}] {trace_key(tr)} {n_resp} tok, "
              f"{elapsed / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
