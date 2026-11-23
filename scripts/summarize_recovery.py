#!/usr/bin/env python3
"""Render the recovery tables in docs/RESULTS.md from finished run directories.

`experiments/runs/` is gitignored, so the numbers in the document are the only
copy that travels with the repository. Regenerate it after a pipeline finishes:

    python3 scripts/summarize_recovery.py experiments/runs/<run> [<run> ...]

Each argument is a run directory that already contains
`evaluate_student_soft_prompt/recovery_curve.json`.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# Question counts are what decide whether a recovery column means anything: a
# benchmark contributes (teacher - student) * n_questions questions of headroom,
# and a column whose headroom is a handful of questions moves in huge steps.
QUESTION_COUNTS = {
    "aime_2024": 30,
    "aime_2025": 30,
    "aime_2026": 30,
    "amc_2022_2023": 83,
    "hmmt_feb_2025": 30,
    "math_500": 500,
    "olympiad_bench": 675,
    "minerva_math": 272,
    "gsm8k": 1319,
}

SHORT = {
    "aime_2024": "aime24",
    "aime_2025": "aime25",
    "aime_2026": "aime26",
    "amc_2022_2023": "amc",
    "hmmt_feb_2025": "hmmt25",
    "math_500": "math500",
    "olympiad_bench": "olympiad",
    "minerva_math": "minerva",
}


def load(run: Path) -> dict:
    return json.loads(
        (run / "evaluate_student_soft_prompt" / "recovery_curve.json").read_text()
    )


def benchmarks(curve: dict) -> list[str]:
    keys = [k for k in curve["baselines"]["student"] if k != "overall"]
    return ["overall", *sorted(keys)]


def denominator_table(curve: dict) -> list[str]:
    rows = [
        "| benchmark | questions | student | teacher | headroom | headroom in questions |",
        "|---|---|---|---|---|---|",
    ]
    for bm in benchmarks(curve):
        s = curve["baselines"]["student"][bm]
        t = curve["baselines"]["teacher"][bm]
        n = QUESTION_COUNTS.get(bm)
        if n:
            headroom = (t - s) * n
            # A headroom under ten questions is the whole story for that column,
            # so do not let rounding hide a sub-question denominator.
            q = f"{headroom:.1f}" if abs(headroom) < 10 else f"{headroom:.0f}"
        else:
            q = "-"
        flag = "" if t - s > 0 else "  **<= 0**"
        rows.append(
            f"| {SHORT.get(bm, bm)} | {n or '-'} | {s:.4f} | {t:.4f} | {t - s:+.4f}{flag} | {q} |"
        )
    return rows


def _percent(value: float | None) -> str:
    """A recovery ratio is undefined when the teacher did not beat the student.

    The pipeline writes null in that case rather than inventing a number, and a
    published pair really can land there: oreal-7b's teacher scores below its own
    student on aime24 under this evaluation. Render it as n/a instead of crashing,
    so the rest of the table stays readable.
    """

    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _aggregate(points: list[dict], benchmark: str, function) -> str:
    values = [
        p["recovery_ratio"][benchmark]
        for p in points
        if p["recovery_ratio"][benchmark] is not None
    ]
    if not values:
        return "**n/a**"
    return f"**{function(values) * 100:.1f}%**"


def recovery_table(curve: dict) -> list[str]:
    bms = benchmarks(curve)
    head = "| step | " + " | ".join(SHORT.get(b, b) for b in bms) + " |"
    rows = [head, "|---|" + "---|" * len(bms)]
    points = sorted(curve["points"], key=lambda p: p["step"])
    for p in points:
        cells = " | ".join(_percent(p["recovery_ratio"][b]) for b in bms)
        rows.append(f"| {p['step']} | {cells} |")
    trained = [p for p in points if p["step"] > 0]
    if trained:
        mean = " | ".join(_aggregate(trained, b, statistics.mean) for b in bms)
        best = " | ".join(_aggregate(trained, b, max) for b in bms)
        rows.append(f"| **mean (step>0)** | {mean} |")
        rows.append(f"| **max (step>0)** | {best} |")
    return rows


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    for arg in argv[1:]:
        run = Path(arg)
        curve = load(run)
        print(f"### {run.name}\n")
        print("\n".join(denominator_table(curve)))
        print()
        print("\n".join(recovery_table(curve)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
