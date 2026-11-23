"""Audit three matched B200 seeds and render throughput and P95 TTFT."""
import argparse
import csv
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from audit_registered_20k import audit_cell, read

ARMS = {"base": ("Base", "#4b5563", "o"), "sp": ("SP256", "#147d92", "s"),
        "r4": ("LoRA r4", "#b35c00", "D")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--kv", required=True, type=int)
    parser.add_argument("--rates", required=True, type=float, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    winner = read(args.selection)["winner"]
    rates = sorted(set(args.rates))
    rows = []
    for seed in (1, 2, 3):
        for rate in rates:
            matched = [
                audit_cell(args.root / f"seed-{seed}" / f"{arm}-rate-{rate:g}",
                           arm, rate, winner, gpu="B200", duration=180, kv_tokens=args.kv, seed=seed)
                for arm in ARMS
            ]
            assert len({row["shared_trace_sha256"] for row in matched}) == 1, (seed, rate)
            rows.extend(matched)
    assert all(row["provenance"] == rows[0]["provenance"] for row in rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [key for key in rows[0] if key != "provenance"]
    with args.out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    fig, (throughput, latency) = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for arm, (label, color, marker) in ARMS.items():
        centers = []
        for axis, key in ((throughput, "output_tokens_s"), (latency, "ttft_p95_lower_bound_s")):
            centers.clear()
            for rate in rates:
                group = [r for r in rows if r["arm"] == arm and r["rate"] == rate]
                values = [r[key] for r in group]
                if any(value is None for value in values):
                    centers.append(float("nan"))
                    continue
                center = median(values)
                is_lower_bound = axis is latency and any(r["ttft_censored"] for r in group)
                centers.append(float("nan") if is_lower_bound else center)
                if is_lower_bound:
                    axis.scatter(rate, center, marker="^", facecolors="none", edgecolors=color)
                    axis.annotate("", xy=(rate, center * 1.15), xytext=(rate, center),
                                  arrowprops={"arrowstyle": "->", "color": color})
                else:
                    axis.vlines(rate, min(values), max(values), color=color, alpha=0.65)
            axis.plot(rates, centers, marker=marker, color=color, label=label)
    for axis in (throughput, latency):
        axis.set_xscale("log", base=2)
        axis.set_xticks(rates, [f"{rate:g}" for rate in rates])
        axis.set_xlabel("Offered request rate (requests/s)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    latency.set_yscale("log")
    max_ttft = max(r["ttft_p95_lower_bound_s"] or 0 for r in rows)
    if any(r["ttft_censored"] for r in rows):
        latency.set_ylim(top=max(latency.get_ylim()[1], 1.25 * max_ttft))
    throughput.set_ylabel("Total output throughput (tokens/s)")
    latency.set_ylabel("P95 TTFT from /generate (s)")
    fig.suptitle("GPT-OSS-20B BF16 on one B200")
    note = "20,480 text tokens, 256 output tokens. Median and range across three 180 s traces."
    if any(r["ttft_censored"] for r in rows):
        note += "\nArrows show conservative lower bounds when a seed has missing first tokens."
    fig.text(0.5, 0.01, note, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.095, 1, 0.94))
    for ext in (".pdf", ".png"):
        fig.savefig(args.out.with_suffix(ext), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
