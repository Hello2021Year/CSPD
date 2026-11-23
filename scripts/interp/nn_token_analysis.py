#!/usr/bin/env python3
"""Nearest-neighbour token analysis of one soft-prompt run (CPU, read-only).

Appendix G embedding geometry: against the input
embedding table, top-k neighbours, category counts, training drift, SVD of the
prompt update, and an output-side (lm_head) projection.  Only the embedding,
final-norm and lm_head tensors are read from the student checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import unicodedata

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

TOPK = 10
SEED = 1234

REASONING_WORDS = {
    "wait", "check", "verify", "verifying", "verification", "step", "steps",
    "therefore", "thus", "hence", "so", "but", "however", "actually", "let",
    "lets", "recall", "compute", "calculate", "answer", "final", "solve",
    "think", "thinking", "reason", "reasoning", "double", "confirm", "careful",
    "carefully", "hmm", "alternatively", "consider", "maybe", "perhaps",
    "correct", "mistake", "again", "re", "recheck", "sanity", "need", "must",
    "should", "first", "next", "then", "finally", "result", "solution",
    "problem", "proof", "prove", "assume", "suppose", "since", "because",
}
HARMONY_WORDS = {
    "analysis", "commentary", "final", "channel", "channels", "message",
    "system", "assistant", "user", "developer", "reasoning", "chatgpt",
    "openai", "model", "instructions", "medium", "high", "low",
}
MATH_CHARS = set("+-*/=^_<>≤≥≠±×÷√∑∏∫∞πθαβγδλμσφω∂∈∉⊂⊆∪∩∀∃→⇒⇔≡≈∝°%|")


def categorize(text: str, token_id: int, special_ids: set[int]) -> str:
    stripped = text.strip()
    if token_id in special_ids or re.fullmatch(r"<\|[^|]*\|>", stripped):
        return "harmony_special"
    if not stripped:
        return "whitespace"
    if any(
        "぀" <= ch <= "ヿ" or "ｦ" <= ch <= "ﾟ" for ch in stripped
    ):
        return "japanese_kana"
    if any("一" <= ch <= "鿿" for ch in stripped):
        return "cjk_han"
    word = re.sub(r"[^a-z]", "", stripped.lower())
    if word and word in HARMONY_WORDS:
        return "harmony_word"
    if word and word in REASONING_WORDS:
        return "reasoning_word"
    if stripped.startswith("\\") or any(ch in MATH_CHARS for ch in stripped) or (
        stripped.isdigit()
    ):
        return "math_symbol"
    if re.search(r"[{}();\[\]]|::|->|==|def$|^def|return|import|self", stripped):
        return "code_like"
    if all(ch.isascii() for ch in stripped):
        if stripped.isalpha():
            return "ascii_word"
        return "ascii_other"
    scripts = {unicodedata.name(ch, "?").split(" ")[0] for ch in stripped if ch.isalpha()}
    return "other_script" if scripts else "other_symbol"


def load_tensor(model_dir: Path, name: str) -> torch.Tensor:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        shard = model_dir / json.loads(index_path.read_text())["weight_map"][name]
        with safe_open(str(shard), framework="pt") as handle:
            return handle.get_tensor(name).float()
    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as handle:
            return handle.get_tensor(name).float()
    state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", mmap=True,
                       weights_only=True)
    return state[name].float()


def pick_detail_updates(updates: list[int], best: int) -> list[int]:
    """Init, ~20% and ~45% of training, the best observed point, and the final point."""

    last = updates[-1]
    picks = {updates[0], best, last}
    for frac in (0.2, 0.45):
        picks.add(min(updates, key=lambda u: abs(u - frac * last)))
    return sorted(picks)


def quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.flatten().float()
    qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    out = dict(zip(["p05", "p25", "median", "p75", "p95"], torch.quantile(values, qs).tolist()))
    out["mean"] = values.mean().item()
    return out


def topk_cos(x: torch.Tensor, e_unit: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    sims = F.normalize(x, dim=-1) @ e_unit.T
    return sims.topk(k, dim=-1)


def topk_l2(x: torch.Tensor, e: torch.Tensor, e_sq: torch.Tensor, k: int) -> torch.Tensor:
    dist = (x * x).sum(-1, keepdim=True) + e_sq[None, :] - 2 * x @ e.T
    return dist.topk(k, dim=-1, largest=False).indices


def participation(sigma: torch.Tensor) -> dict[str, float]:
    var = sigma**2
    return {
        "pr_variance": (var.sum() ** 2 / (var**2).sum()).item(),
        "pr_sigma": (sigma.sum() ** 2 / var.sum()).item(),
        "top1_var_frac": (var[0] / var.sum()).item(),
        "top5_var_frac": (var[:5].sum() / var.sum()).item(),
        "top10_var_frac": (var[:10].sum() / var.sum()).item(),
    }


def principal_cosines(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    # a, b: [k, d] orthonormal rows
    return torch.linalg.svdvals(a @ b.T).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="package run dir, e.g. .../math-gptoss")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="local student snapshot directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    config = json.loads((args.model_dir / "config.json").read_text())

    # ---- data -------------------------------------------------------------
    E_full = load_tensor(args.model_dir, "model.embed_tokens.weight")
    # rows past the tokenizer are padding the tokenizer never emits
    n_real = min(len(tokenizer), E_full.size(0))
    special_ids = {int(i) for i in tokenizer.all_special_ids}
    pieces = [tokenizer.decode([i]) for i in range(n_real)]
    categories = [categorize(pieces[i], i, special_ids) for i in range(n_real)]
    E = E_full[:n_real].contiguous()
    U = (E_full if config.get("tie_word_embeddings") else
         load_tensor(args.model_dir, "lm_head.weight"))[:n_real].contiguous()
    norm_w = load_tensor(args.model_dir, "model.norm.weight")
    eps = float(config.get("rms_norm_eps", 1e-5))
    E_unit = F.normalize(E, dim=-1)
    E_sq = (E * E).sum(-1)
    E_norm = E.norm(dim=-1)
    pad_norm = E_full[n_real:].norm(dim=-1)

    steps = {}
    with (args.run_dir / "steps.csv").open() as handle:
        for row in csv.DictReader(handle):
            steps[int(row["update"])] = float(row["recovery_percent"])
    ckpt_dir = args.run_dir / "train" / "checkpoints"
    P = {}
    for path in sorted(ckpt_dir.glob("update_*.safetensors")):
        t = int(path.stem.split("_")[1])
        with safe_open(str(path), framework="pt") as handle:
            P[t] = handle.get_tensor("prompt_embeddings").float()
    updates = sorted(P)
    pair = json.loads((args.run_dir / "pair.json").read_text())
    REFERENCE_UPDATE = int(pair["best_observed_update"])
    DETAIL_UPDATES = pick_detail_updates(updates, REFERENCE_UPDATE)
    ref = f"u{REFERENCE_UPDATE}"
    P0, Pref = P[0], P[REFERENCE_UPDATE]

    # init token recovery: exact row match against the full table
    # cdist uses a matmul expansion with rounding error, so confirm candidates
    # by exact element-wise comparison.
    init_ids = torch.cdist(P0, E_full).argmin(dim=-1).tolist()
    for i, j in enumerate(init_ids):
        if not torch.equal(P0[i], E_full[j]):
            # e.g. several all-zero unused rows: take an exactly equal row if any
            exact = (E_full == P0[i]).all(dim=-1).nonzero().flatten()
            if exact.numel():
                init_ids[i] = int(exact[0])
    init_exact = [bool(torch.equal(P0[i], E_full[j])) for i, j in enumerate(init_ids)]
    zero_rows = int((E_full.abs().amax(-1) < 1e-6).sum())
    init_zero_rows = int((P0.abs().amax(-1) < 1e-6).sum())
    # the student may have been loaded in bf16 while the table here is wider
    init_exact_bf16 = [
        bool(torch.equal(P0[i].bfloat16(), E_full[j].bfloat16())) for i, j in enumerate(init_ids)
    ]
    # informational: replay the training sampler on a CPU generator
    gen = torch.Generator().manual_seed(42)
    allowed = torch.tensor([i for i in range(min(n_real, E_full.size(0))) if i not in special_ids])
    replay = allowed[torch.randint(len(allowed), (P0.size(0),), generator=gen)].tolist()

    # baselines (reference = Pref)
    ref_norms = Pref.norm(dim=-1, keepdim=True)
    non_special = torch.tensor([i for i in range(n_real) if i not in special_ids])
    b1_ids = non_special[torch.randint(len(non_special), (Pref.size(0),))]
    B1 = E[b1_ids]
    B2 = F.normalize(torch.randn_like(Pref), dim=-1) * ref_norms
    B3 = torch.stack([row[torch.randperm(row.numel())] for row in Pref])
    points = {f"u{t}": P[t] for t in updates}
    points.update({"B1_random_tokens": B1, "B2_gauss_normmatched": B2, "B3_shuffled_ref": B3})

    # ---- 2.1 geometry + 2.2 neighbours -------------------------------------
    e_med = E_norm.median().item()
    geometry = {
        "vocab_rows_used": n_real,
        "embedding_norm": quantiles(E_norm),
        "padding_row_norm": quantiles(pad_norm) if pad_norm.numel() else None,
        "reference_update": REFERENCE_UPDATE,
        "detail_updates": DETAIL_UPDATES,
        "embedding_coord_std": E.std().item(),
        "embedding_mean_vector_norm": E.mean(0).norm().item(),
        "init_rows_exact_match": f"{sum(init_exact)}/{len(init_exact)}",
        "init_rows_exact_match_bf16": f"{sum(init_exact_bf16)}/{len(init_exact_bf16)}",
        "embedding_near_zero_rows": zero_rows,
        "init_rows_near_zero": init_zero_rows,
        "init_ids_match_cpu_seed42_replay": sum(a == b for a, b in zip(init_ids, replay)),
        "points": {},
    }
    neighbour_cache = {}
    for name, X in points.items():
        vals, idx = topk_cos(X, E_unit, TOPK)
        l2_idx = topk_l2(X, E, E_sq, TOPK)
        jacc = []
        for a, b in zip(idx.tolist(), l2_idx.tolist()):
            sa, sb = set(a), set(b)
            jacc.append(len(sa & sb) / len(sa | sb))
        top1_tok = idx[:, 0]
        geometry["points"][name] = {
            "recovery_percent": steps.get(int(name[1:])) if name.startswith("u") else None,
            "norm_ratio_to_median_vocab": quantiles(X.norm(dim=-1) / e_med),
            "top1_cos": quantiles(vals[:, 0]),
            "margin_top1_top2": quantiles(vals[:, 0] - vals[:, 1]),
            "cos_l2_top10_jaccard": quantiles(torch.tensor(jacc)),
            "top1_l2_is_top1_cos_frac": (l2_idx[:, 0] == top1_tok).float().mean().item(),
            "distinct_top1_tokens": len(set(top1_tok.tolist())),
        }
        neighbour_cache[name] = (vals, idx)

    for t in DETAIL_UPDATES:
        vals, idx = neighbour_cache[f"u{t}"]
        with (out / f"nn_top10_u{t}.csv").open("w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["position", "init_token", "rank", "token_id", "token", "category", "cos"])
            for i in range(idx.size(0)):
                for r in range(TOPK):
                    tid = idx[i, r].item()
                    w.writerow([i, repr(pieces[init_ids[i]]), r + 1, tid, repr(pieces[tid]),
                                categories[tid], f"{vals[i, r].item():.4f}"])

    # aggregate category counts over top-10 neighbours
    cat_names = sorted(set(categories))
    vocab_counts = {c: 0 for c in cat_names}
    for c in categories:
        vocab_counts[c] += 1
    agg = {}
    for name in [f"u{REFERENCE_UPDATE}", "u0", "B1_random_tokens", "B2_gauss_normmatched", "B3_shuffled_ref"]:
        counts = {c: 0 for c in cat_names}
        for tid in neighbour_cache[name][1].flatten().tolist():
            counts[categories[tid]] += 1
        agg[name] = counts
    with (out / "aggregate_categories.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["category", "vocab_frac"] + [f"{n}_frac" for n in agg])
        total = 256 * TOPK
        for c in cat_names:
            w.writerow([c, f"{vocab_counts[c] / n_real:.4f}"] + [f"{agg[n][c] / total:.4f}" for n in agg])

    # most frequent neighbour tokens for the reference point
    freq: dict[int, int] = {}
    for tid in neighbour_cache[f"u{REFERENCE_UPDATE}"][1].flatten().tolist():
        freq[tid] = freq.get(tid, 0) + 1
    top_freq = sorted(freq.items(), key=lambda kv: -kv[1])[:50]

    # reverse neighbours: tokens closest to any SP row
    rev = (E_unit @ F.normalize(Pref, dim=-1).T).max(dim=-1)
    rev_vals, rev_idx = rev.values.topk(50)
    reverse = [
        {"token_id": i, "token": pieces[i], "category": categories[i], "cos": v,
         "nearest_position": rev.indices[i].item()}
        for i, v in zip(rev_idx.tolist(), rev_vals.tolist())
    ]

    # ---- 2.3 drift --------------------------------------------------------
    drift_rows = []
    first_leave = [None] * P0.size(0)
    prev = None
    for t in updates:
        X = P[t]
        cos0 = F.cosine_similarity(X, P0, dim=-1)
        d0 = (X - P0).norm(dim=-1)
        sims = F.normalize(X, dim=-1) @ E_unit.T
        init_sim = sims.gather(1, torch.tensor(init_ids)[:, None])
        init_rank = (sims > init_sim).sum(-1)  # 0 means still top-1
        for i, rnk in enumerate(init_rank.tolist()):
            if rnk > 0 and first_leave[i] is None:
                first_leave[i] = t
        step_change = (X - prev).norm().item() if prev is not None else 0.0
        drift_rows.append({
            "update": t,
            "recovery_percent": steps.get(t),
            "cos_to_init_median": cos0.median().item(),
            "cos_to_init_p05": torch.quantile(cos0, 0.05).item(),
            "l2_from_init_median": d0.median().item(),
            "row_norm_median": X.norm(dim=-1).median().item(),
            "init_token_top1_frac": (init_rank == 0).float().mean().item(),
            "init_token_rank_median": init_rank.float().median().item(),
            "init_token_cos_median": init_sim.median().item(),
            "frobenius_change_from_prev": step_change,
            "delta_coord_rms": (X - P0).pow(2).mean().sqrt().item(),
            "delta_coord_rms_over_embedding_std": ((X - P0).pow(2).mean().sqrt() / E.std()).item(),
            "top1_cos_median": sims.max(dim=-1).values.median().item(),
            "delta_mean_component_share": (
                X.size(0) * (X - P0).mean(0).norm() ** 2 / (X - P0).norm() ** 2
            ).item() if t > 0 else 0.0,
            "delta_pairwise_row_cos_mean": (
                lambda Z: Z[~torch.eye(Z.size(0), dtype=torch.bool)].mean().item()
            )(F.normalize(X - P0, dim=-1) @ F.normalize(X - P0, dim=-1).T) if t > 0 else 0.0,
        })
        prev = X
    with (out / "drift.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(drift_rows[0]))
        w.writeheader()
        w.writerows(drift_rows)
    left = [t for t in first_leave if t is not None]

    # ---- 2.4 SVD of the update + 2.5 output-side projection ---------------
    def rmsnorm(x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * norm_w

    def describe_direction(v: torch.Tensor, k: int = 20) -> dict:
        res = {}
        for sign, vec in (("+", v), ("-", -v)):
            cv, ci = (F.normalize(vec, dim=0) @ E_unit.T).topk(k)
            lv, li = (U @ rmsnorm(vec)).topk(k)
            res[sign] = {
                "input_nn": [[pieces[i], round(c, 4)] for i, c in zip(ci.tolist(), cv.tolist())],
                "lm_head_top": [[pieces[i], round(c, 3)] for i, c in zip(li.tolist(), lv.tolist())],
            }
        return res

    svd = {}
    deltas = {f"dP_u{t}": P[t] - P0 for t in updates if t > 0}
    dref = deltas[f"dP_u{REFERENCE_UPDATE}"]
    deltas["dP_ref_shuffled"] = torch.stack([row[torch.randperm(row.numel())] for row in dref])
    deltas["gauss_same_shape"] = torch.randn_like(dref) * dref.std()
    bases = {}
    for name, D in deltas.items():
        _, S, Vh = torch.linalg.svd(D, full_matrices=False)
        svd[name] = {
            "sigma_top10": S[:10].tolist(),
            **participation(S),
            "abs_cos_mean_vs_v0": F.cosine_similarity(D.mean(0), Vh[0], dim=0).abs().item(),
        }
        bases[name] = Vh[:10]
    svd["subspace_overlap_top10_vs_ref"] = {
        name: principal_cosines(bases[name], bases[f"dP_u{REFERENCE_UPDATE}"])
        for name in bases
    }
    _, S, Vh = torch.linalg.svd(dref, full_matrices=False)
    svd["ref_directions"] = [
        {"k": k, "sigma": S[k].item(), **describe_direction(Vh[k])} for k in range(10)
    ]
    svd["mean_delta_direction"] = describe_direction(dref.mean(0))
    svd["mean_ref_row_direction"] = describe_direction(Pref.mean(0))

    # output side for each SP row at the reference update
    logits = rmsnorm(Pref) @ U.T
    lv, li = logits.topk(TOPK, dim=-1)
    out_freq: dict[int, int] = {}
    for tid in li.flatten().tolist():
        out_freq[tid] = out_freq.get(tid, 0) + 1
    out_cat = {c: 0 for c in cat_names}
    for tid in li.flatten().tolist():
        out_cat[categories[tid]] += 1
    with (out / f"lm_head_top10_u{REFERENCE_UPDATE}.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["position", "rank", "token_id", "token", "category", "logit"])
        for i in range(li.size(0)):
            for r in range(TOPK):
                tid = li[i, r].item()
                w.writerow([i, r + 1, tid, repr(pieces[tid]), categories[tid], f"{lv[i, r].item():.3f}"])
    # the same projection for random tokens, as a reference for lm_head "hubs"
    b_logits = rmsnorm(B1) @ U.T
    b_freq: dict[int, int] = {}
    for tid in b_logits.topk(TOPK, dim=-1).indices.flatten().tolist():
        b_freq[tid] = b_freq.get(tid, 0) + 1

    def fmt_freq(f: dict[int, int], n: int = 40) -> list:
        return [[pieces[i], categories[i], c] for i, c in sorted(f.items(), key=lambda kv: -kv[1])[:n]]

    summary = {
        "geometry": geometry,
        "init_first_leave_top1": {
            "left_by_end": f"{len(left)}/{len(first_leave)}",
            "first_leave_update_quantiles": quantiles(torch.tensor(left, dtype=torch.float)) if left else None,
        },
        "top_neighbour_tokens_ref": [[pieces[i], categories[i], c] for i, c in top_freq],
        "reverse_neighbours_ref": reverse,
        "lm_head_top10_freq_ref": fmt_freq(out_freq),
        "lm_head_top10_category_ref": out_cat,
        "lm_head_top10_freq_B1": fmt_freq(b_freq),
        "category_counts_top10": agg,
        "vocab_category_counts": vocab_counts,
    }
    (out / "geometry.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (out / "svd_directions.json").write_text(json.dumps(svd, ensure_ascii=False, indent=1))
    torch.save({"init_ids": init_ids}, out / "init_ids.pt")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
