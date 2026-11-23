"""Plot the paper's B200 curves from the archived closed-loop summary."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).resolve().parents[1] / "data/b200-paper.csv")
    parser.add_argument("--output", type=Path, default=Path("b200-paper.pdf"))
    args = parser.parse_args()
    with args.csv.open(newline="") as stream:
        rows = {
            (row["arm"], int(row["concurrency"])): row
            for row in csv.DictReader(stream)
            if int(row["kv_tokens"]) == (524288 if int(row["concurrency"]) <= 16 else 1048576)
        }
    levels = sorted({level for _, level in rows})
    fig, axes = plt.subplots(2, 2, sharex="col", figsize=(8.6, 5.4), height_ratios=[2.5, 1])
    for column, (key, ylabel, scale) in enumerate([
        ("total_output_tokens_s", "Output tokens / s", 1.0),
        ("ttft_p95_ms", "P95 TTFT (s)", 0.001),
    ]):
        top, bottom = axes[:, column]
        values = {}
        for arm, marker, color in [("Base", "o", "#444444"), ("SP256", "s", "#118373"), ("LoRA r4", "^", "#ae5446")]:
            values[arm] = [float(rows[arm, level][key]) * scale for level in levels]
            top.plot(levels, values[arm], marker=marker, color=color, label=arm)
        bottom.axhline(0, color="#444444", linewidth=0.8)
        bottom.plot(levels, [100 * (sp / base - 1) for sp, base in zip(values["SP256"], values["Base"])],
                    marker="s", color="#118373")
        top.set_ylabel(ylabel)
        bottom.set_ylabel("SP vs. base (%)")
        bottom.set_xlabel("Concurrent requests")
        bottom.set_xticks(levels)
        top.grid(axis="y", alpha=0.2)
        bottom.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
