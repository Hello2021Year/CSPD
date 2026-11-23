#!/usr/bin/env python3
"""Correctness checks for sp_pathway.py on a tiny random GPT-OSS.

Run this on the target machine before the real experiment.  It needs only the
student tokenizer (for the harmony special tokens) and builds a 4-layer model
with a 16-token sliding window, so every check takes seconds.

Reference: transformers' own eager attention given explicit per-layer-type 4D
masks.  Comparisons run in fp32, where the implementation must agree exactly
or to float rounding; the flex backend is additionally compared with
transformers' flex_attention in bf16 for the unmasked case, which is what
teacher_forced_scoring.py used.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer, GptOssConfig, GptOssForCausalLM

import sp_pathway as spp

PROMPT = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
          "Knowledge cutoff: 2024-06\nCurrent date: 2026-08-21\n\nReasoning: medium\n\n"
          "# Valid channels: analysis, commentary, final. Channel must be included for every "
          "message.<|end|><|start|>user<|message|>Find all real x such that x^2=4.<|end|>"
          "<|start|>assistant")
RESPONSE = ("<|channel|>analysis<|message|>We need x^2=4. So x=2 or x=-2. But check: 2^2=4, "
            "(-2)^2=4. Good. Thus answer.<|end|><|start|>assistant<|channel|>final<|message|>"
            "x = 2 or x = -2<|return|>")
INSERT_AT, N_SOFT, WINDOW = 3, 8, 16
LAYER_TYPES = ["sliding_attention", "full_attention"] * 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    dev = args.device
    backends = ["eager"] + ([] if dev == "cpu" else ["flex"])

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    p_ids = tok.encode(PROMPT, add_special_tokens=False)
    r_ids = tok.encode(RESPONSE, add_special_tokens=False)
    n_resp = len(r_ids)
    cfg = GptOssConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=64,
                       num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
                       head_dim=16, num_local_experts=4, num_experts_per_tok=2,
                       sliding_window=WINDOW, layer_types=LAYER_TYPES)

    def make(attn, dtype):
        torch.manual_seed(0)
        m = GptOssForCausalLM._from_config(copy.deepcopy(cfg), attn_implementation=attn,
                                           dtype=dtype)
        for mod in m.modules():
            if hasattr(mod, "sinks"):
                torch.nn.init.normal_(mod.sinks, 0.0, 1.0)  # make sinks matter
        return m.to(dev).eval()

    start_id = tok.convert_tokens_to_ids("<|start|>")
    role_ids = {tok.encode(r, add_special_tokens=False)[0]: g for r, g in spp.ROLE_GROUPS.items()}
    groups = spp.label_groups(p_ids, INSERT_AT, N_SOFT, n_resp, start_id, role_ids).to(dev)
    T = len(groups)
    start = len(p_ids) + N_SOFT - 1
    conds = {c.name: c for c in spp.build_conditions(
        {"core", "layers", "heads", "readout", "kv"}, LAYER_TYPES, 8, 2, "u180", "u0", [1, 3],
        ["direct", "indirect"], ["all", "l1", "l3g1"], ["direct", "open"])}
    c0 = conds["u180.C0"]
    results = []

    def check(name, diff, ok):
        results.append((name, diff, ok))

    for dtype in (torch.float32, torch.bfloat16):
        ko = make(spp.ATTN_NAME, dtype)
        spp.attach_router_hooks(ko)
        torch.manual_seed(1)
        soft = (torch.randn(N_SOFT, 64) * 0.05).to(dev, dtype)
        donor = (torch.randn(N_SOFT, 64) * 0.05).to(dev, dtype)
        embed_text = ko.get_input_embeddings()(torch.tensor([p_ids + r_ids], device=dev))
        embed_sp = torch.cat([embed_text[:, :INSERT_AT], soft[None], embed_text[:, INSERT_AT:]], 1)

        def run(cond, backend, mass=(), readout=None, sp=None):
            spp.STATE.backend = backend
            spp.STATE.set_sequence(groups, 0)
            if cond.kv:  # donor keys/values, as sp_pathway.main does
                spp.STATE.set_condition(spp.Condition("capture", "u0"), None,
                                        capture_layers=range(4))
                spp.response_hidden(ko, embed_text, INSERT_AT, donor, start, 1)
            spp.STATE.set_condition(cond, spp.blocked_tensor(cond, 4, 8, dev), mass_layers=mass,
                                    record_router=True)
            if readout:
                spp.STATE.readout = readout
            sp = soft if sp is None else sp
            return spp.response_hidden(ko, embed_text, INSERT_AT, sp, start, n_resp).float()

        def reference(attn, masks=None):
            m = make(attn, dtype)
            m.load_state_dict(ko.state_dict())
            out = m.model(inputs_embeds=embed_sp, attention_mask=masks, use_cache=False)
            return out.last_hidden_state[0, start:start + n_resp].float()

        def max_diff(a, b):
            return (a - b).abs().max().item()

        if dtype == torch.bfloat16:
            if "flex" in backends:
                d = max_diff(run(c0, "flex"), reference("flex_attention"))
                check("bf16 C0 flex == transformers flex_attention", d, d == 0)
            continue

        # fp32: every condition against transformers eager with explicit 4D masks
        d = max_diff(run(c0, "eager"), reference("eager"))
        check("C0 eager == transformers eager", d, d < 1e-5)
        qi, ki = torch.arange(T)[:, None], torch.arange(T)[None]
        g = groups.cpu()
        for name in ("u180.C_all", "u180.C_direct", "u180.C_indirect", "u180.C_ind_sys",
                     "u180.C_ind_q", "u180.C_win", "u180.C_full"):
            blk = spp.blocked_tensor(conds[name], 4, 8, "cpu")
            masks = {}
            for lt, layer in (("sliding_attention", 0), ("full_attention", 1)):
                ok = ki <= qi
                if lt == "sliding_attention":
                    ok = ok & (qi - ki < WINDOW)
                ok = ok & ~(blk[layer, 0][g][:, None] & (g == spp.SP)[None])
                masks[lt] = torch.zeros(T, T).masked_fill(~ok, float("-inf"))[None, None].to(dev)
            ref = reference("eager", masks)
            for backend in backends:
                d = max_diff(run(conds[name], backend), ref)
                check(f"{name} {backend} == transformers eager + 4D mask", d, d < 1e-4)
        for name in ("u180.direct.head_l1_g1", "u180.indirect.only_l2",
                     "u180.direct.read_mean_l1", "u180.direct.read_shuffle_l3"):
            if "flex" in backends:
                d = max_diff(run(conds[name], "flex"), run(conds[name], "eager"))
                check(f"{name} flex == eager", d, d < 1e-4)
        for backend in backends:
            base = run(c0, backend)
            d = max_diff(run(c0, backend, readout={1: "identity", 3: "identity"}), base)
            check(f"identity readout == C0 ({backend})", d, d < 1e-5)
            d = max_diff(run(c0, backend, mass=[1, 3]), base)
            check(f"recording SP mass changes nothing ({backend})", d, d == 0)
            for name in ("u180.direct.read_mean_l1", "u180.direct.head_l1_g1"):
                d = max_diff(run(conds[name], backend), run(conds["u180.C_direct"], backend))
                check(f"{name} differs from C_direct ({backend})", d, d > 1e-4)
            # R is more than WINDOW tokens past SP, so sliding layers cannot reach it
            cond = spp.Condition("x", "u180", (((0, 2), (spp.R,), None),))
            d = max_diff(run(cond, backend), base)
            check(f"R->SP blocked in sliding layers is a no-op ({backend})", d, d == 0)
            cond = spp.Condition("y", "u180", (((1,), (spp.R,), tuple(range(8))),))
            same = spp.Condition("z", "u180", (((1,), (spp.R,), None),))
            d = max_diff(run(cond, backend), run(same, backend))
            check(f"blocking every head == blocking the layer ({backend})", d, d == 0)
        masses = {}
        for backend in backends:
            run(c0, backend, mass=[1, 3])
            masses[backend] = {k: v.float() for k, v in spp.STATE.mass.items()}
            ok = all(bool((v >= 0).all() and (v <= 1 + 1e-3).all()) for v in masses[backend].values())
            check(f"SP mass within [0, 1] ({backend})", 0.0, ok)
        if "flex" in backends:
            d = max(max_diff(masses["flex"][k], masses["eager"][k]) for k in (1, 3))
            check("SP mass flex == eager", d, d < 1e-3)

        # kv swaps: main keys/values reproduce the base condition, donor ones at every layer
        # reproduce the donor run, and the swap's SP mass matches the split's
        for backend in backends:
            for path, base in (("direct", "u180.C_direct"), ("open", "u180.C0")):
                ref = run(conds[base], backend)
                d = max_diff(run(conds[f"u180.{path}.kv_all_ttt"], backend), ref)
                check(f"kv {path} all ttt == {base} ({backend})", d, d < 1e-4)
                d = max_diff(run(conds[f"u180.{path}.kv_all_000"], backend),
                             run(conds[base.replace("u180", "u0")], backend, sp=donor))
                check(f"kv {path} all 000 == donor {base[5:]} ({backend})", d, d < 1e-4)
                for combo in ("tt0", "0t0", "rrr", "ttp"):
                    d = max_diff(run(conds[f"u180.{path}.kv_l1_{combo}"], backend), ref)
                    check(f"kv {path} l1 {combo} differs from base ({backend})", d, d > 1e-4)
            run(c0, backend, mass=[1])
            split_mass = spp.STATE.mass[1].float().mean(0)
            for combo in ("ttt", "t0n", "tt0"):
                run(conds[f"u180.open.kv_l1_{combo}"] if combo != "t0n" else
                    spp.Condition("m", "u180", kv=(((1,), None, spp.KV_PATHS["open"], "t0n"),)),
                    backend)
                d = max_diff(spp.STATE.kv_mass[1].float(), split_mass)
                check(f"kv mass of {combo} == SP mass of C0 ({backend})", d, d < 1e-3)
            run(conds["u180.open.kv_l1_0t0"], backend)
            m_0t0 = spp.STATE.kv_mass[1].clone()
            run(conds["u180.open.kv_l1_000"], backend)
            d = max_diff(spp.STATE.kv_mass[1], m_0t0)
            check(f"mass follows the mass key only ({backend})", d, d < 1e-6)
            # one kv head swapped to the donor is neither the base nor that head blocked
            a = run(conds["u180.direct.kv_l3g1_000"], backend)
            d1 = max_diff(a, run(conds["u180.C_direct"], backend))
            d2 = max_diff(a, run(conds["u180.direct.kv_l3g1_blk"], backend))
            check(f"kv l3g1 000 differs from base and from blocking ({backend})", min(d1, d2),
                  d1 > 1e-4 and d2 > 1e-4)
        if "flex" in backends:
            for name in ("u180.direct.kv_l3g1_t0t", "u180.open.kv_all_nn0", "u180.open.kv_l1_rrt"):
                d = max_diff(run(conds[name], "flex"), run(conds[name], "eager"))
                check(f"{name} flex == eager", d, d < 1e-4)

    fails = 0
    for name, diff, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  (max diff {diff:.3g})")
        fails += not ok
    print(f"{len(results) - fails}/{len(results)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
