#!/usr/bin/env python3
"""Cross-model figure: absolute prompt-update size vs. embedding scale (CPU)."""

from __future__ import annotations

import argparse
import csv
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
GROUPS = ["math-gptoss", "math-poro", "math-mimo", "code-nemotron", "code-mellum", "code-areal8"]


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interp-dir", type=Path, required=True,
                        help="directory holding <group>-nn analysis outputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": MUTED})

    rows = []
    for group in GROUPS:
        base = args.interp_dir / f"{group}-nn"
        geo = json.loads((base / "geometry.json").read_text())["geometry"]
        ref = geo["reference_update"]
        drift = {int(r["update"]): r for r in csv.DictReader((base / "drift.csv").open())}
        rows.append({
            "group": group,
            "estd": geo["embedding_coord_std"],
            "drms": float(drift[ref]["delta_coord_rms"]),
            "cos0": float(drift[ref]["cos_to_init_median"]),
            "ref": ref,
        })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    style(ax1)
    xs = [r["estd"] for r in rows]
    lo, hi = min(xs) / 2, max(xs) * 2
    for ratio, label in ((0.05, "update = 5% of token scale"), (4.0, "update = 4× token scale")):
        ax1.plot([lo, hi], [lo * ratio, hi * ratio], color=GRID, linewidth=1.2, linestyle="--")
        y = 0.2 if ratio > 1 else 0.15
        ax1.annotate(label, (y / ratio, y), textcoords="offset points", xytext=(6, 0),
                     fontsize=8, color=MUTED, ha="left", va="center")
    ax1.scatter(xs, [r["drms"] for r in rows], s=48, color=SERIES, zorder=3,
                edgecolor=SURFACE, linewidth=2)
    for r in rows:
        left = r["group"] == "math-mimo"
        ax1.annotate(r["group"], (r["estd"], r["drms"]), textcoords="offset points",
                     xytext=(-7 if left else 7, -3), ha="right" if left else "left",
                     fontsize=8, color=TEXT)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(0.01, 0.3)
    ax1.set_xlabel("Embedding coordinate std (student vocab)", fontsize=9)
    ax1.set_ylabel("ΔP coordinate RMS at best update", fontsize=9)
    ax1.set_title("Updates are similar in size; embedding scales are not", fontsize=10,
                  color=TEXT, loc="left")

    style(ax2)
    ax2.grid(axis="x", visible=False)
    order = sorted(rows, key=lambda r: r["cos0"])
    ax2.barh([r["group"] for r in order], [r["cos0"] for r in order], color=SERIES, height=0.6)
    for i, r in enumerate(order):
        ax2.annotate(f"{r['cos0']:.3f}", (r["cos0"], i), textcoords="offset points",
                     xytext=(4, -3), fontsize=8, color=MUTED)
    ax2.set_xlim(0, 1.12)
    ax2.set_xlabel("Median cos(P_best row, its initial token)", fontsize=9)
    ax2.set_title("Only GPT-OSS stays at its initial tokens", fontsize=10, color=TEXT, loc="left")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
