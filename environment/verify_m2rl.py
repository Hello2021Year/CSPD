"""Check Docker benchmark scores against the published M2RL tables."""

import argparse
import json
import statistics
import sys
from pathlib import Path


TABLES = {
    "2": (
        ("aime24", "lcb_v5", "gpqa_diamond", "ifbench"),
        {
            "MCSI": (80.47, 64.67, 57.77, 68.42, 67.83),
            "student": (55.94, 58.61, 44.38, 42.71, 50.41),
            "ISCM": (79.90, 65.72, 57.92, 67.33, 67.72),
            "single_sp": (80.89, 64.67, 49.24, 66.71, 65.38),
            "oracle": (81.77, 66.92, 50.44, 67.33, 66.61),
            "mopd": (83.28, 66.24, 55.21, 66.42, 67.79),
        },
    ),
    "3": (
        ("xstest", "truthfulqa", "bfcl", "triviaqa", "bbeh", "finqa"),
        {
            "MCSI": (88.2, 64.2, 55.6, 33.5, 25.8, 72.5, 56.6),
            "student": (80.4, 64.6, 47.6, 31.5, 11.1, 66.3, 50.3),
            "mopd": (84.2, 61.0, 49.4, 31.9, 23.9, 70.8, 53.5),
        },
    ),
}

# Official question counts and samples per question from experiments/m2rl/BENCHMARKS.md.
BENCHMARKS = {
    "aime24": (30, 64),
    "lcb_v5": (167, 8),
    "gpqa_diamond": (198, 16),
    "ifbench": (300, 8),
    "xstest": (450, 4),
    "truthfulqa": (817, 4),
    "bfcl": (4041, 1),
    "triviaqa": (2000, 1),
    "bbeh": (460, 1),
    "finqa": (1147, 1),
}


def read_score(path: Path, benchmark: str) -> float:
    """Read an upstream grader summary and verify the expected run coverage."""
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    questions, k = BENCHMARKS[benchmark]
    if benchmark == "bfcl":
        expected = {"cases": questions, "missing": 0}
    else:
        expected = {"bench": benchmark, "questions": questions, "samples": questions * k}
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(f"{path} has {field}={result.get(field)!r}, expected {value!r}")

    score = result.get("score")
    if type(score) not in (int, float) or not 0 <= score <= 100:
        raise ValueError(f"{path} has an invalid percentage score")
    return float(score)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="A node output directory or the parent of node-0 and node-1")
    parser.add_argument("--table", choices=TABLES, default="2")
    parser.add_argument("--method", action="append", help="Method to check, defaults to all methods for the table")
    args = parser.parse_args()

    benchmarks, reference = TABLES[args.table]
    methods = args.method or list(reference)
    for method in methods:
        if method not in reference:
            parser.error(f"Table {args.table} methods: {', '.join(reference)}")

    single_node = args.run_dir / "results"
    result_dirs = [single_node] if single_node.is_dir() else sorted(args.run_dir.glob("node-*/results"))
    if not result_dirs:
        parser.error(f"No node results under {args.run_dir}")

    print(f"## Table {args.table}")
    print()
    print("| Method | Benchmark | Measured | Paper | Δ (points) |")
    print("| --- | --- | ---: | ---: | ---: |")
    mean_digits = 2 if args.table == "2" else 1
    incomplete = False
    for method in methods:
        measured = []
        for benchmark, paper in zip(benchmarks, reference[method]):
            paths = [directory / method / benchmark / "score.json" for directory in result_dirs]
            matches = [path for path in paths if path.is_file()]
            try:
                if len(matches) != 1:
                    raise ValueError("Missing score.json" if not matches else "Duplicate node scores")
                actual = read_score(matches[0], benchmark)
            except ValueError as error:
                print(f"| {method} | {benchmark} | — | {paper:.2f} | — |")
                print(f"{method}/{benchmark}: {error}", file=sys.stderr)
                incomplete = True
                continue
            measured.append(actual)
            print(f"| {method} | {benchmark} | {actual:.2f} | {paper:.2f} | {actual - paper:+.2f} |")

        paper_mean = reference[method][-1]
        if len(measured) == len(benchmarks):
            mean = round(statistics.mean(measured), mean_digits)
            print(f"| {method} | Mean | {mean:.2f} | {paper_mean:.2f} | {mean - paper_mean:+.2f} |")
        else:
            print(f"| {method} | Mean | — | {paper_mean:.2f} | — |")
    return int(incomplete)


if __name__ == "__main__":
    raise SystemExit(main())
