#!/usr/bin/env python3
"""Recovery fractions per knockout condition.

Reads the per-trace npz/json written by sp_pathway.py.  For a condition c,
recovery = (KL_S - KL_c) / (KL_S - KL_C0), with KL = KL(T||.) summed over
response tokens; 1 means the soft prompt's effect is intact, 0 means it is gone.
The same ratio is reported on decision positions (the top fraction of
Delta = logp[C0] - logp[S], pooled within the subset) and for analysis-channel
probabilities of a few decision tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

DECISION_WATCH = (" But", " So", "<|end|>")


def load(results_dir: Path):
    conditions = [c["name"] for c in json.loads((results_dir / "conditions.json").read_text())]
    traces = []
    for path in sorted(results_dir.glob("*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())
        traces.append((meta, dict(np.load(path))))
    if not traces:
        raise SystemExit(f"no results in {results_dir}")
    return conditions, traces


def ratio(num, den):
    return float(num / den) if den != 0 else float("nan")


def summarize(conditions, traces, main, top_frac, n_boot=1000):
    meta0 = traces[0][0]
    watch_idx = {t: list(meta0["watch_tokens"]).index(t) for t in DECISION_WATCH
                 if t in meta0["watch_tokens"]}
    cat = {c: {k: np.concatenate([d[f"{c}__{k}"] for _, d in traces]).astype(np.float64)
               for k in ("kl", "logp")} for c in conditions}
    watch = {c: np.concatenate([d[f"{c}__watch"] for _, d in traces]).astype(np.float64)
             for c in conditions}
    analysis = np.concatenate([np.arange(m["n_response"]) < m["final_start"] for m, _ in traces])
    ref = f"{main}.C0"
    delta = cat[ref]["logp"] - cat["S"]["logp"]
    k = max(1, int(round(top_frac * len(delta))))
    decision = np.zeros(len(delta), dtype=bool)
    decision[np.argsort(-delta)[:k]] = True

    kl_s, kl_ref = cat["S"]["kl"], cat[ref]["kl"]
    # 95% interval of the KL recovery by resampling whole traces
    owner = np.repeat(np.arange(len(traces)), [m["n_response"] for m, _ in traces])
    per_trace = lambda x: np.bincount(owner, weights=x, minlength=len(traces))
    boot = np.random.default_rng(0).integers(0, len(traces), (n_boot, len(traces)))
    gap = (per_trace(kl_s) - per_trace(kl_ref))[boot].sum(1)
    p_s = {t: np.exp(watch["S"][analysis, j]).mean() for t, j in watch_idx.items()}
    p_ref = {t: np.exp(watch[ref][analysis, j]).mean() for t, j in watch_idx.items()}
    rows = []
    for c in conditions:
        kl_c = cat[c]["kl"]
        row = {
            "condition": c,
            "kl_mean": float(kl_c.mean()),
            "recovery": ratio(kl_s.sum() - kl_c.sum(), kl_s.sum() - kl_ref.sum()),
            "recovery_ci": np.percentile(
                (per_trace(kl_s) - per_trace(kl_c))[boot].sum(1) / gap, [2.5, 97.5]).tolist(),
            "recovery_decision": ratio(kl_s[decision].sum() - kl_c[decision].sum(),
                                       kl_s[decision].sum() - kl_ref[decision].sum()),
        }
        for t, j in watch_idx.items():
            p_c = np.exp(watch[c][analysis, j]).mean()
            row[f"P({t.strip() or t})"] = float(p_c)
            row[f"rec_P({t.strip() or t})"] = ratio(p_c - p_s[t], p_ref[t] - p_s[t])
        overlap = np.stack([d[f"{c}__router_overlap"] for _, d in traces])
        row["router_overlap"] = float(np.nanmean(overlap))
        if f"{c}__kv_mass" in traces[0][1]:
            mass = np.concatenate([d[f"{c}__kv_mass"] for _, d in traces]).astype(np.float64)
            row["kv_mass"] = float(mass.mean())
            row["kv_mass_decision"] = float(mass[decision].mean())
        rows.append(row)
    return rows, decision


def attention_mass(traces, decision, sp_names):
    """Mean attention mass on SP per (full layer, head), on decision vs other R positions."""
    out = {}
    offsets = np.cumsum([0] + [m["n_response"] for m, _ in traces])
    for name in sp_names:
        key = f"{name}.C0__sp_mass"
        if key not in traces[0][1]:
            continue
        mass = np.concatenate([d[key].astype(np.float64) for _, d in traces], axis=-1)
        assert mass.shape[-1] == offsets[-1]
        out[name] = {
            "full_layers": traces[0][0]["full_layers"],
            "decision": mass[..., decision].mean(-1).tolist(),
            "other": mass[..., ~decision].mean(-1).tolist(),
        }
    return out


def layer_scans(rows):
    """{path: {kind: {layer: recovery}}} for the drop / only / upto / from scans."""
    scans: dict = {}
    for r in rows:
        m = re.match(r"[^.]+\.(direct|indirect)\.(drop|only|upto|from)_l(\d+)$", r["condition"])
        if m:
            scans.setdefault(m[1], {}).setdefault(m[2], {})[int(m[3])] = r["recovery"]
    return scans


def kv_scans(rows):
    """{path: {target: {combo or 'blk': recovery}}} for the kv set."""
    scans: dict = {}
    for r in rows:
        m = re.match(r"[^.]+\.(direct|open)\.kv_([a-z0-9]+)_([0tnrp]{3}|blk)$", r["condition"])
        if m:
            scans.setdefault(m[1], {}).setdefault(m[2], {})[m[3]] = r["recovery"]
    return scans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--main", default="u180")
    parser.add_argument("--control", default="u0")
    parser.add_argument("--top-frac", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    conditions, traces = load(args.results_dir)
    subsets = {"all": traces}
    for src in sorted({m["source"] for m, _ in traces}):
        subsets[src] = [t for t in traces if t[0]["source"] == src]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    csv_rows = []
    for name, sub in subsets.items():
        rows, decision = summarize(conditions, sub, args.main, args.top_frac)
        summary[name] = {
            "n_traces": len(sub),
            "n_tokens": int(sum(m["n_response"] for m, _ in sub)),
            "r_in_window_tokens": int(sum(m["r_in_window"] for m, _ in sub)),
            "conditions": rows,
            "layer_scans": layer_scans(rows),
            "kv_scans": kv_scans(rows),
            "sp_mass": attention_mass(sub, decision, [args.main, args.control]),
        }
        csv_rows += [{"subset": name} | r for r in rows]

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    with (args.output_dir / "conditions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"{'condition':40s} {'recovery':>9s} {'decision':>9s} {'router':>7s}")
    for r in summary["all"]["conditions"]:
        if re.search(r"_l\d+|\.kv_", r["condition"]):
            continue  # scans are in summary.json
        print(f"{r['condition']:40s} {r['recovery']:9.3f} {r['recovery_decision']:9.3f} "
              f"{r['router_overlap']:7.3f}")
    for path, targets in summary["all"]["kv_scans"].items():
        combos = sorted({c for t in targets.values() for c in t}, key=lambda c: (c == "blk", c))
        print(f"\nkv {path}: recovery")
        print(f"{'target':8s} " + " ".join(f"{c:>6s}" for c in combos))
        for target, vals in targets.items():
            print(f"{target:8s} " + " ".join(f"{vals.get(c, float('nan')):6.2f}" for c in combos))
    print(f"wrote {args.output_dir / 'summary.json'} and conditions.csv")


if __name__ == "__main__":
    main()
