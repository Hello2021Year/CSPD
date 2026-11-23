"""Audit fixed-concurrency bursts and plot server-side prefill and decode."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import re
from statistics import median, quantiles
import struct

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from audit_registered_20k import lines, read
from report_b200_registered_long import ARMS

LEVELS = (1, 4, 8, 16)


def measure(root, extra_runs=(), closed_loop=False, levels=LEVELS, kv_tokens=524288):
    samples = []
    inputs = {}
    roots = [(root, levels), *extra_runs]
    for arm in ARMS:
        hashes = []
        for run_root, levels in roots:
            samples.extend(measure_arm(run_root / arm, arm, levels, inputs, hashes, closed_loop, kv_tokens))
        assert len(hashes) == len(set(hashes))
    return samples


def measure_arm(server, arm, levels, inputs, hashes, closed_loop, kv_tokens):
    samples = []
    assert len(read(server / "allocated-gpus.json")) == 1
    assert "B200" in read(server / "allocated-gpus.json")[0]["name"]
    config = read(server / "experiment.json")
    assert config["max_total_tokens"] == kv_tokens and config["tp_size"] == 1
    capacities = re.findall(r"max_total_num_tokens=(\d+).*?max_running_requests=(\d+)",
                            (server / "server.log").read_text())
    assert capacities and all(int(kv) == kv_tokens and int(slots) >= max(levels)
                              for kv, slots in capacities), (server, capacities)
    tokens = defaultdict(list)
    for ns, rid, count in struct.iter_unpack("<QII", (server / "engine.tokens.bin").read_bytes()):
        tokens[rid].append((ns, count))
    for level in levels:
        stages = ("warmup", "measurement") if closed_loop else (
            "warmup", "repeat-1", "repeat-2", "repeat-3")
        for stage in stages:
            cell = server / f"concurrency-{level}" / stage
            bench = read(cell / "config.json")
            sustained = stage == "measurement"
            count = 8 * level if sustained else level
            assert (bench["count"], bench["text_input_tokens"], bench["output_tokens"]) == (count, 20480, 512)
            assert bench["adapter_preparation"] == "registered"
            assert bench["arrival_pattern"] == ("closed-loop" if sustained else "burst")
            if arm == "r4":
                assert bench["kernel"] == "csgmv"
            signature = read(cell / "input-hashes.json")
            assert inputs.setdefault((level, stage), signature) == signature
            assets = read(cell / "adapter-manifest.json")
            assert len(assets) == (0 if arm == "base" else count)
            hashes.extend(asset["sha256"] for asset in assets)
            requests = lines(cell / "requests.jsonl")
            assert len(requests) == count
            if sustained:
                assert bench["slots"] == level
                events = sorted((r[key], delta) for r in requests
                                for key, delta in (("generate_sent", 1), ("finished", -1)))
                running = peak = 0
                for _, delta in events:
                    running += delta
                    peak = max(peak, running)
                assert running == 0 and peak == level
            else:
                assert len({r["scheduled"] for r in requests}) == 1
            elapsed = max(r["finished"] for r in requests) - min(r["generate_sent"] for r in requests)
            ttft = [r["first_token_received"] - r["generate_sent"] for r in requests]
            p95 = quantiles(ttft, n=100, method="inclusive")[94] if count > 1 else ttft[0]
            for request in requests:
                assert request["status"] == "success" and request["num_retractions"] == 0
                events = tokens[int(request["request_id"][3:])]
                first, start_count = events[0]
                last, end_count = events[-1]
                assert (start_count, end_count) == (1, 512) and last > first
                prefill_ms = 1000 * (request["prefill_finished_time"] - request["forward_entry_time"])
                assert prefill_ms > 0
                if stage != "warmup":
                    samples.append(dict(arm=arm, concurrency=level, repeat=stage,
                                        request_id=request["request_id"], prefill_ms=prefill_ms,
                                        decode_tokens_s=511e9 / (last - first),
                                        ttft_ms=1000 * (request["first_token_received"] - request["generate_sent"]),
                                        run_ttft_p95_ms=1000 * p95,
                                        run_output_tokens_s=count * 512 / elapsed))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--concurrency-24", type=Path)
    parser.add_argument("--concurrency-32", type=Path)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--levels", nargs="+", type=int)
    parser.add_argument("--kv-tokens", type=int, default=524288)
    args = parser.parse_args()
    assert not (args.closed_loop and args.concurrency_24)
    assert not args.concurrency_32 or args.closed_loop
    primary_levels = tuple(args.levels) if args.levels else ((*LEVELS, 24) if args.closed_loop else LEVELS)
    extra_runs = [(root, (level,)) for root, level in (
        (args.concurrency_24, 24), (args.concurrency_32, 32)) if root is not None]
    rows = measure(args.root, extra_runs, args.closed_loop, primary_levels, args.kv_tokens)
    levels = (*primary_levels, *(level for _, (level,) in extra_runs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for axis, key, title in zip(axes, ("prefill_ms", "decode_tokens_s"),
                               ("Median prefill time (ms/request)", "Median decode speed (tokens/s/request)")):
        for arm, (label, color, marker) in ARMS.items():
            center = [median(r[key] for r in rows if r["arm"] == arm and r["concurrency"] == level)
                      for level in levels]
            axis.plot(levels, center, label=label, color=color, marker=marker)
        axis.set_xticks(levels)
        axis.set_xlabel("In-flight request limit" if args.closed_loop else "Concurrent requests per burst")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle(f"One B200, 20,480 input tokens, 512 output tokens, {args.kv_tokens // 1024}K KV")
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(args.out.with_suffix(suffix), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
