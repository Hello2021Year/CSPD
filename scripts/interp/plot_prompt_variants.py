#!/usr/bin/env python3
"""Figure: text-prompt arms vs. the soft prompt (KL reduction, alignment with SP effect)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
ORDER = ["S+u180", "S+H_verify", "S+H_check", "S+H_neutral", "S+R_high"]
LABEL = {"S+u180": "soft prompt u180", "S+H_verify": "H_verify (long)",
         "S+H_check": "H_check (short)", "S+H_neutral": "H_neutral",
         "S+R_high": "Reasoning: high"}


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": MUTED})
    S = json.loads(args.compare.read_text())["by_source"]["teacher"]
    labels = [LABEL[a] for a in ORDER][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
    panels = [
        ([S[a]["kl_reduction_vs_S"] * 100 for a in ORDER][::-1], "KL(T‖arm) reduction vs. student (%)",
         "Movement of the student toward the teacher", "{:.1f}%"),
        ([S[a]["corr_with_ref"] for a in ORDER][::-1], "Correlation of per-token Δ with soft prompt's Δ",
         "Same tokens changed as the soft prompt?", "{:.2f}"),
    ]
    for ax, (vals, xlabel, title, fmt) in zip(axes, panels):
        style(ax)
        ax.barh(labels, vals, color=SERIES, height=0.6)
        for i, v in enumerate(vals):
            ax.annotate(fmt.format(v), (max(v, 0), i), textcoords="offset points", xytext=(4, -3),
                        fontsize=8, color=MUTED)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
