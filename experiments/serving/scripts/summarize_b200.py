"""Audit the raw B200 runs and write the paper's per-arm, per-level metrics."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import median

from report_b200_prefill_decode import measure

GROUPS = {
    "low": (524288, (1, 4, 8, 16)),
    "c2": (524288, (2,)),
    "high": (1048576, (24,)),
    "c32": (1048576, (32,)),
}
ARM_NAMES = {"base": "Base", "sp": "SP256", "r4": "LoRA r4"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for group in GROUPS:
        parser.add_argument(f"--{group}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = []
    for group, (kv, levels) in GROUPS.items():
        by_cell = defaultdict(list)
        for row in measure(getattr(args, group), closed_loop=True, levels=levels, kv_tokens=kv):
            by_cell[row["concurrency"], row["arm"]].append(row)
        for (level, arm), rows in by_cell.items():
            summaries.append({
                "concurrency": level,
                "arm": ARM_NAMES[arm],
                "kv_tokens": kv,
                "measured_requests": len(rows),
                "prefill_median_ms": median(row["prefill_ms"] for row in rows),
                "total_output_tokens_s": rows[0]["run_output_tokens_s"],
                "ttft_p95_ms": rows[0]["run_ttft_p95_ms"],
                "decode_median_tokens_s_per_request": median(row["decode_tokens_s"] for row in rows),
            })
    summaries.sort(key=lambda row: row["concurrency"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
