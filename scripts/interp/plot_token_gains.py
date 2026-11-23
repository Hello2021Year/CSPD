#!/usr/bin/env python3
"""Figures for the teacher-forced token-gain report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = {"S+u40": "#2a78d6", "S+u180": "#eb6834", "T": "#1baf7a"}
LABEL = {"S+u40": "student + u40", "S+u180": "student + u180", "T": "teacher"}


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
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": MUTED})
    S = json.loads(args.summary.read_text())["by_source"]["teacher"]

    # Figure 1: mean delta by student-entropy decile on teacher traces.
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    style(ax)
    x = np.arange(10)
    width = 0.38
    for i, arm in enumerate(("S+u40", "S+u180")):
        vals = [d["mean_delta"] for d in S[arm]["by_student_entropy_decile"]]
        ax.bar(x + (i - 0.5) * width, vals, width=width - 0.04, color=SERIES[arm], label=LABEL[arm])
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, [f"D{d + 1}" for d in range(10)])
    ax.set_xlabel("Student entropy decile at the position (D1 = most certain)", fontsize=9)
    ax.set_ylabel("Mean Δ log-prob of teacher token (nats)", fontsize=9)
    ax.set_title("Soft-prompt gains sit at the student's uncertain positions", fontsize=10,
                 color=TEXT, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(args.output_dir / "math_gptoss_delta_by_entropy.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # Figure 2: decision-token probability in the analysis channel, relative to the student.
    watch = S["watch_tokens_analysis_channel"]
    names = ["<|end|>", " So", " Therefore", " But", "Wait"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor=SURFACE)
    style(ax)
    x = np.arange(len(names))
    width = 0.26
    for i, arm in enumerate(("S+u40", "S+u180", "T")):
        ratios = [watch[n][arm]["mean_prob_analysis"] / watch[n]["S"]["mean_prob_analysis"]
                  for n in names]
        ax.bar(x + (i - 1) * width, ratios, width=width - 0.03, color=SERIES[arm], label=LABEL[arm])
    ax.axhline(1, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_xticks(x, [repr(n) for n in names])
    ax.set_ylabel("Mean probability / student's (log scale)", fontsize=9)
    ax.set_title("Analysis channel, teacher traces: SP moves decision tokens toward the teacher",
                 fontsize=10, color=TEXT, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.output_dir / "math_gptoss_decision_tokens.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
