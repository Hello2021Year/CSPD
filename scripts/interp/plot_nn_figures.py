#!/usr/bin/env python3
"""Figures for the math-gptoss nearest-neighbour report (CPU, read-only)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from safetensors import safe_open

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEED = 1234


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": MUTED})

    rows = list(csv.DictReader((args.analysis_dir / "drift.csv").open()))
    t = [int(r["update"]) for r in rows]
    rec = [float(r["recovery_percent"]) for r in rows]
    cos0 = [float(r["cos_to_init_median"]) for r in rows]
    ckpt = args.run_dir / "train" / "checkpoints"

    def load(u: int) -> torch.Tensor:
        with safe_open(str(ckpt / f"update_{u:08d}.safetensors"), framework="pt") as h:
            return h.get_tensor("prompt_embeddings").float()

    P0 = load(0)
    rel = [((load(u) - P0).norm(dim=-1) / P0.norm(dim=-1)).median().item() * 100 for u in t]

    # Figure 1: three small multiples sharing the update axis.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), facecolor=SURFACE)
    panels = [
        (rec, "Recovery (%)", "Recovery vs. student→teacher gap"),
        (rel, "Median ‖ΔP row‖ / ‖P₀ row‖ (%)", "Relative movement from init"),
        (cos0, "Median cos(P_t row, P₀ row)", "Similarity to initial token"),
    ]
    for ax, (y, ylabel, title) in zip(axes, panels):
        style(ax)
        ax.plot(t, y, color=SERIES[0], linewidth=2, marker="o", markersize=4)
        ax.set_xlabel("Optimizer update", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.axvline(180, color=MUTED, linewidth=0.8, linestyle=":")
    axes[0].annotate("best u180", (180, rec[t.index(180)]), textcoords="offset points",
                     xytext=(5, 5), fontsize=8, color=MUTED)
    axes[0].set_ylim(-10, 105)
    axes[2].ticklabel_format(axis="y", useOffset=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "math_gptoss_drift.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # Figure 2: cumulative variance captured by top-k singular directions.
    torch.manual_seed(SEED)
    D = load(180) - P0
    variants = {
        "ΔP (u180 − u0)": D,
        "ΔP, coordinates shuffled per row": torch.stack([r[torch.randperm(r.numel())] for r in D]),
        "Gaussian, same shape & scale": torch.randn_like(D) * D.std(),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4), facecolor=SURFACE)
    style(ax)
    k = torch.arange(1, 257)
    for color, (name, M) in zip(SERIES, variants.items()):
        var = torch.linalg.svdvals(M) ** 2
        cum = (var.cumsum(0) / var.sum() * 100).tolist()
        ax.plot(k.tolist(), cum, color=color, linewidth=2, label=f"{name}  (top-10: {cum[9]:.0f}%)")
    ax.axvline(10, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Top-k singular directions", fontsize=9)
    ax.set_ylabel("Cumulative variance (%)", fontsize=9)
    ax.set_title("ΔP is high-rank, but more concentrated than controls", fontsize=10,
                 color=TEXT, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.output_dir / "math_gptoss_dP_spectrum.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
