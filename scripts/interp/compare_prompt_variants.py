#!/usr/bin/env python3
"""Compare natural-language prompt arms with the soft prompt at token level.

For every arm A (soft prompt or text variant) and trace source we report the
KL(T||A) reduction relative to the student, analysis-channel probabilities of
the watched decision tokens, and how the per-position effect
Delta_A = logp[A] - logp[S] relates to the soft prompt's Delta_ref:
Pearson correlation, the least-squares slope of Delta_A on Delta_ref, and sign
agreement where the soft prompt moves a token by more than 0.5 nats.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from analyze_token_gains import col, load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    D, traces, meta = load(args.scores)
    arms = meta["arms"]
    ref = meta["reference_arm"]
    others = [a for a in arms if a.startswith("S+")]
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    ids_final = tok.encode("<|channel|>final", add_special_tokens=False)

    n = len(D["token"])
    source = np.empty(n, dtype=object)
    in_final = np.zeros(n, dtype=bool)
    for tr in traces:
        o, L = tr["offset"], tr["length"]
        source[o:o + L] = tr["source"]
        toks = D["token"][o:o + L].tolist()
        for i in range(L - len(ids_final) + 1):
            if toks[i:i + len(ids_final)] == ids_final:
                in_final[o + i:o + L] = True
                break

    logp = {a: D[f"logp__{col(a)}"].astype(np.float64) for a in arms}
    ent_s = D["entropy__S"].astype(np.float64)
    result: dict = {"arms": arms, "reference_arm": ref, "by_source": {}}
    for src in ["teacher", "student", "student_sp", "all"]:
        m = np.ones(n, bool) if src == "all" else source == src
        if not m.any():
            continue
        kl_s = float(D["kl_T__S"][m].mean())
        d_ref = logp[ref][m] - logp["S"][m]
        big = np.abs(d_ref) > 0.5
        hi = ent_s[m] > np.quantile(ent_s[m], 0.7)
        out = {}
        for a in others:
            d = logp[a][m] - logp["S"][m]
            kl = float(D[f"kl_T__{col(a)}"][m].mean())
            entry = {
                "kl_T": kl,
                "kl_reduction_vs_S": (kl_s - kl) / kl_s,
                "mean_nll": float(-logp[a][m].mean()),
                "corr_with_ref": float(np.corrcoef(d, d_ref)[0, 1]),
                "corr_with_ref_high_entropy": float(np.corrcoef(d[hi], d_ref[hi])[0, 1]),
                "slope_on_ref": float((d * d_ref).sum() / (d_ref * d_ref).sum()),
                "sign_agreement_where_ref_moves_gt_0p5": float(
                    (np.sign(d[big]) == np.sign(d_ref[big])).mean()),
                "mean_abs_delta": float(np.abs(d).mean()),
            }
            y = D["token"][m]
            order = np.argsort(d)
            entry["most_promoted"] = Counter(
                tok.decode([int(t)]) for t in y[order[::-1][:2000]]).most_common(20)
            entry["most_suppressed"] = Counter(
                tok.decode([int(t)]) for t in y[order[:2000]]).most_common(20)
            watch = meta.get("watch_tokens", {})
            ana = ~in_final[m]
            entry["watch_prob_analysis"] = {
                name: float(np.exp(D[f"watch_logp__{col(a)}"][m][:, j].astype(np.float64))[ana].mean())
                for j, name in enumerate(watch)}
            out[a] = entry
        out["S_watch_prob_analysis"] = {
            name: float(np.exp(D["watch_logp__S"][m][:, j].astype(np.float64))[~in_final[m]].mean())
            for j, name in enumerate(meta.get("watch_tokens", {}))}
        out["T_watch_prob_analysis"] = {
            name: float(np.exp(D["watch_logp__T"][m][:, j].astype(np.float64))[~in_final[m]].mean())
            for j, name in enumerate(meta.get("watch_tokens", {}))}
        out["kl_T_S"] = kl_s
        result["by_source"][src] = out
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
