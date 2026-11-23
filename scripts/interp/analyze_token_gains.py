#!/usr/bin/env python3
"""Aggregate teacher-forced per-token scores.

Reads the npz/meta shards written by teacher_forced_scoring.py and writes a
summary JSON plus CSV tables.  Delta = logp[S+ref] - logp[S] on the realised
trace token; gap = logp[T] - logp[S].
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import re

import numpy as np
from transformers import AutoTokenizer

CONNECTIVES = {
    "wait", "but", "so", "let", "actually", "hmm", "alternatively", "check", "verify",
    "thus", "therefore", "hence", "however", "now", "then", "first", "next", "also",
    "double", "recheck", "confirm", "maybe", "perhaps", "ok", "okay", "good", "great",
    "compute", "consider", "note", "since", "because", "we", "need", "answer", "final",
}
TOP_FRACS = (0.01, 0.05, 0.10, 0.25)


def load(paths: list[Path]):
    cols: dict[str, list[np.ndarray]] = {}
    traces = []
    base = 0
    meta0 = None
    for path in paths:
        data = np.load(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        meta0 = meta0 or meta
        for key in data.files:
            cols.setdefault(key, []).append(data[key])
        for tr in meta["traces"]:
            traces.append({**tr, "offset": tr["offset"] + base})
        base += len(data["token"])
    return {k: np.concatenate(v) for k, v in cols.items()}, traces, meta0


def col(arm: str) -> str:
    return arm.replace("+", "_")


def categorize(piece: str, tid: int, special: set[int]) -> str:
    s = piece.strip()
    if tid in special or re.fullmatch(r"<\|[^|]*\|>", s):
        return "harmony_special"
    if not s:
        return "newline" if "\n" in piece else "whitespace"
    if re.fullmatch(r"\d+", s):
        return "digits"
    word = re.sub(r"[^a-z]", "", s.lower())
    if word and word in CONNECTIVES and s.replace("'", "").isalpha():
        return "connective"
    if s.startswith("\\") or re.search(r"[=+\-*/^_{}()\[\]<>|$]", s):
        return "math_latex"
    if s.isalpha():
        return "word"
    return "other"


def concentration(values: np.ndarray) -> dict:
    order = np.sort(values)[::-1]
    total = order.sum()
    pos = order[order > 0].sum()
    out = {"net_total": float(total), "positive_mass": float(pos),
           "negative_mass": float(order[order < 0].sum()), "n": int(len(order))}
    for f in TOP_FRACS:
        k = max(1, int(round(f * len(order))))
        out[f"top{int(f * 100)}pct_share_of_net"] = float(order[:k].sum() / total) if total else None
        out[f"top{int(f * 100)}pct_share_of_positive"] = float(order[:k].clip(min=0).sum() / pos) if pos else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    D, traces, meta = load(args.scores)
    arms = meta["arms"]
    ref = meta["reference_arm"]
    sp_arms = [a for a in arms if a.startswith("S+")]
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    special = {int(i) for i in tok.all_special_ids}
    ids_final = tok.encode("<|channel|>final", add_special_tokens=False)
    end_id = tok.convert_tokens_to_ids("<|end|>")

    n = len(D["token"])
    source = np.empty(n, dtype=object)
    correct = np.zeros(n, dtype=bool)
    pos = np.zeros(n, dtype=np.int64)
    relpos = np.zeros(n, dtype=np.float64)
    in_final = np.zeros(n, dtype=bool)
    is_switch = np.zeros(n, dtype=bool)  # the <|end|> that closes the analysis channel
    for tr in traces:
        o, L = tr["offset"], tr["length"]
        source[o:o + L] = tr["source"]
        correct[o:o + L] = tr["correct"]
        pos[o:o + L] = np.arange(L)
        relpos[o:o + L] = np.arange(L) / max(1, L)
        toks = D["token"][o:o + L].tolist()
        for i in range(L - len(ids_final) + 1):
            if toks[i:i + len(ids_final)] == ids_final:
                in_final[o + i:o + L] = True
                for j in range(i - 1, -1, -1):
                    if toks[j] == end_id:
                        is_switch[o + j] = True
                        break
                break

    vocab_cat = {}
    def cat_of(t: int) -> str:
        if t not in vocab_cat:
            vocab_cat[t] = categorize(tok.decode([t]), t, special)
        return vocab_cat[t]
    category = np.array([cat_of(int(t)) for t in D["token"]], dtype=object)

    logp = {a: D[f"logp__{col(a)}"].astype(np.float64) for a in arms}
    summary: dict = {"arms": arms, "reference_arm": ref, "n_tokens": n,
                     "n_traces": len(traces), "by_source": {}}

    for src in ["teacher", "student", "student_sp", "all"]:
        m = np.ones(n, bool) if src == "all" else (source == src)
        if not m.any():
            continue
        S = {}
        nll = {a: float(-logp[a][m].mean()) for a in arms}
        S["mean_nll"] = nll
        S["token_recovery"] = {
            a: (nll["S"] - nll[a]) / (nll["S"] - nll["T"]) if nll["S"] != nll["T"] else None
            for a in sp_arms}
        S["mean_kl_from_T"] = {a: float(D[f"kl_T__{col(a)}"][m].mean()) for a in arms if a != "T"}
        S["n_tokens"] = int(m.sum())
        S["trace_accuracy"] = float(np.mean([t["correct"] for t in traces if src in ("all", t["source"])]))
        S["trace_len_mean"] = float(np.mean([t["length"] for t in traces if src in ("all", t["source"])]))
        S["truncated_frac"] = float(np.mean([t["truncated"] for t in traces if src in ("all", t["source"])]))

        for a in sp_arms:
            delta = logp[a][m] - logp["S"][m]
            block = {"delta_concentration": concentration(delta)}
            ent = D["entropy__S"][m]
            edges = np.quantile(ent, np.linspace(0, 1, 11))
            dec = np.clip(np.searchsorted(edges, ent, side="right") - 1, 0, 9)
            block["by_student_entropy_decile"] = [
                {"decile": int(d), "entropy_range": [float(edges[d]), float(edges[d + 1])],
                 "mean_delta": float(delta[dec == d].mean()),
                 "share_of_net_delta": float(delta[dec == d].sum() / delta.sum())}
                for d in range(10)]
            am_s, am_a = D["argmax__S"][m], D[f"argmax__{col(a)}"][m]
            y = D["token"][m]
            block["positive_flip_frac"] = float(((am_s != y) & (am_a == y)).mean())
            block["negative_flip_frac"] = float(((am_s == y) & (am_a != y)).mean())
            cats = category[m]
            block["by_category"] = {
                c: {"token_share": float((cats == c).mean()),
                    "mean_delta": float(delta[cats == c].mean()),
                    "share_of_net_delta": float(delta[cats == c].sum() / delta.sum())}
                for c in sorted(set(cats))}
            rp = relpos[m]
            dec10 = np.clip((rp * 10).astype(int), 0, 9)
            block["by_relative_position"] = [float(delta[dec10 == d].mean()) for d in range(10)]
            p = pos[m]
            block["first64_mean_delta"] = float(delta[p < 64].mean())
            block["after64_mean_delta"] = float(delta[p >= 64].mean())
            fin = in_final[m]
            block["analysis_channel_mean_delta"] = float(delta[~fin].mean())
            block["final_channel_mean_delta"] = float(delta[fin].mean()) if fin.any() else None
            sw = is_switch[m]
            block["analysis_end_token"] = {
                "n": int(sw.sum()),
                **{f"mean_logp_{b}": float(logp[b][m][sw].mean()) for b in arms}} if sw.any() else None
            S[a] = block

        # flips and SP-driven decisions for the reference arm
        y = D["token"][m]
        am_s, am_r, am_t = D["argmax__S"][m], D[f"argmax__{col(ref)}"][m], D["argmax__T"][m]
        flip = (am_s != y) & (am_r == y)
        pairs = Counter(zip(y[flip].tolist(), am_s[flip].tolist()))
        S["positive_flip_top_pairs"] = [
            {"trace_token": tok.decode([a]), "student_argmax": tok.decode([b]), "count": c}
            for (a, b), c in pairs.most_common(50)]
        S["positive_flip_token_categories"] = dict(Counter(category[m][flip].tolist()))
        S["positive_flip_teacher_agrees_frac"] = float((am_t[flip] == y[flip]).mean()) if flip.any() else None
        S["positive_flip_mean_logp_T"] = float(logp["T"][m][flip].mean()) if flip.any() else None
        S["all_tokens_teacher_argmax_agrees_frac"] = float((am_t == y).mean())
        rank_s = D["rank__S"][m]
        S["student_rank_ge10_frac"] = float((rank_s >= 10).mean())
        big = rank_s >= 10
        S["student_rank_ge10_teacher_agrees_frac"] = float((am_t[big] == y[big]).mean()) if big.any() else None
        delta = logp[ref][m] - logp["S"][m]
        worst = np.argsort(delta)[:40]
        best = np.argsort(delta)[::-1][:40]
        S["most_suppressed_tokens"] = Counter(tok.decode([int(t)]) for t in y[np.argsort(delta)[:2000]]).most_common(40)
        S["most_promoted_tokens"] = Counter(tok.decode([int(t)]) for t in y[np.argsort(delta)[::-1][:2000]]).most_common(40)
        S["examples_suppressed"] = [{"token": tok.decode([int(y[i])]), "delta": float(delta[i]),
                                     "student_argmax": tok.decode([int(am_s[i])])} for i in worst[:15]]
        S["examples_promoted"] = [{"token": tok.decode([int(y[i])]), "delta": float(delta[i]),
                                   "student_argmax": tok.decode([int(am_s[i])])} for i in best[:15]]
        # decision-token probabilities at every analysis-channel position
        watch = meta.get("watch_tokens", {})
        if watch:
            ana = ~in_final[m]
            rp = relpos[m]
            dec10 = np.clip((rp * 10).astype(int), 0, 9)
            wt = {}
            for j, name in enumerate(watch):
                per_arm = {}
                for a in arms:
                    prob = np.exp(D[f"watch_logp__{col(a)}"][m][:, j].astype(np.float64))
                    per_arm[a] = {
                        "mean_prob_analysis": float(prob[ana].mean()),
                        "mean_prob_by_relpos_analysis": [
                            float(prob[ana & (dec10 == d)].mean()) if (ana & (dec10 == d)).any() else None
                            for d in range(10)],
                    }
                wt[name] = per_arm
            S["watch_tokens_analysis_channel"] = wt
        if src == "teacher":
            for label, cm in (("correct", correct[m]), ("incorrect", ~correct[m])):
                if cm.any():
                    S[f"teacher_traces_{label}"] = {
                        "n_tokens": int(cm.sum()),
                        "mean_nll": {a: float(-logp[a][m][cm].mean()) for a in arms}}
        summary["by_source"][src] = S

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    with (out / "per_trace.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["source", "dataset", "question_id", "sample", "correct", "truncated", "length"]
                   + [f"mean_logp_{a}" for a in arms])
        for tr in traces:
            o, L = tr["offset"], tr["length"]
            w.writerow([tr["source"], tr["dataset"], tr["question_id"], tr["sample"], tr["correct"],
                        tr["truncated"], L] + [f"{logp[a][o:o + L].mean():.4f}" for a in arms])
    np.savez(out / "derived.npz", relpos=relpos, in_final=in_final, is_switch=is_switch,
             category=category.astype(str), source=source.astype(str), correct=correct)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
