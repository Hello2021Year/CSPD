#!/usr/bin/env python3
"""Figures for the attention-pathway analysis (Appendix G).

Reads the ``analysis/conditions.csv`` and ``analysis/summary.json`` files written by
``analyze_sp_pathway.py`` for the core, layers and heads runs, and draws:

  sp_pathway_layers.png   per-layer necessity / sufficiency and cumulative curves, both paths
  sp_pathway_heads.png    per-KV-group knockout on the key layers, both paths
  sp_pathway_mass.png     R->SP attention mass per global layer, decision vs other, u180 vs u0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read_rows(path: Path, subset: str = "all") -> dict[str, dict[str, float]]:
    rows = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["subset"] == subset:
                rows[row["condition"]] = {k: float(v) for k, v in row.items()
                                          if k not in ("subset", "condition") and v != ""}
    return rows


def plot_layers(rows, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    for col, (path, layers) in enumerate([("direct", range(1, 24, 2)), ("indirect", range(24))]):
        ax = axes[0, col]
        drop = [rows[f"u180.{path}.drop_l{l}"]["recovery"] for l in layers]
        only = [rows[f"u180.{path}.only_l{l}"]["recovery"] for l in layers]
        base = rows["u180.C_direct" if path == "direct" else "u180.C_indirect"]["recovery"] \
            if "u180.C_direct" in rows else None
        ax.plot(list(layers), drop, "o-", label="block only this layer (necessity)")
        ax.plot(list(layers), only, "s-", label="open only this layer (sufficiency)")
        if base is not None:
            ax.axhline(base, color="gray", ls="--", lw=1, label=f"path baseline {base:.2f}")
        ax.set_title(f"{path} path: single layer")
        ax.set_xlabel("layer")
        ax.legend(fontsize=8)
        ax = axes[1, col]
        ls = list(range(1, 24))
        upto = [rows[f"u180.{path}.upto_l{l}"]["recovery"] for l in ls]
        frm = [rows[f"u180.{path}.from_l{l}"]["recovery"] for l in ls]
        ax.plot(ls, frm, "o-", label="open layers [0, l) only")
        ax.plot(ls, upto, "s-", label="open layers [l, 24) only")
        ax.set_title(f"{path} path: cumulative")
        ax.set_xlabel("l")
        ax.legend(fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("KL recovery (1 = S+u180, 0 = S)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_heads(rows_d, rows_i, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, rows, path, layers in [(axes[0], rows_d, "direct", [1, 3, 5, 7]),
                                   (axes[1], rows_i, "indirect", [0, 1, 7])]:
        width = 0.8 / len(layers)
        for k, l in enumerate(layers):
            vals = [rows[f"u180.{path}.head_l{l}_g{g}"]["recovery"] for g in range(8)]
            ax.bar(np.arange(8) + k * width, vals, width, label=f"L{l}")
        ax.set_xticks(np.arange(8) + 0.4 - width / 2)
        ax.set_xticklabels([f"g{g}" for g in range(8)])
        ax.set_title(f"{path} path: block one KV group (8 query heads)")
        ax.set_xlabel("KV group")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("KL recovery")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_mass(summary: dict, out: Path) -> None:
    m = summary["all"]["sp_mass"]
    layers = m["u180"]["full_layers"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, style in [("u180", "-"), ("u0", "--")]:
        for part, marker in [("decision", "o"), ("other", "s")]:
            vals = np.array(m[name][part]).mean(1)
            ax.plot(layers, vals, style, marker=marker, label=f"{name} {part}")
    ax.set_xlabel("global layer")
    ax.set_ylabel("mean R->SP attention mass (over 64 heads)")
    ax.set_title("attention to the soft prompt, decision vs other positions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pathway-dir", type=Path, required=True,
                        help="directory holding core/, layers/, heads_direct/, heads_indirect/")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    core = read_rows(args.pathway_dir / "core/analysis/conditions.csv")
    layers = read_rows(args.pathway_dir / "layers/analysis/conditions.csv")
    layers.update({k: v for k, v in core.items() if k in ("u180.C_direct", "u180.C_indirect")})
    plot_layers(layers, args.output_dir / "sp_pathway_layers.png")
    plot_heads(read_rows(args.pathway_dir / "heads_direct/analysis/conditions.csv"),
               read_rows(args.pathway_dir / "heads_indirect/analysis/conditions.csv"),
               args.output_dir / "sp_pathway_heads.png")
    summary = json.loads((args.pathway_dir / "core/analysis/summary.json").read_text())
    plot_mass(summary, args.output_dir / "sp_pathway_mass.png")
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
