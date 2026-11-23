"""Recovery-curve artifacts built from completed baseline/checkpoint evaluations."""

from __future__ import annotations

import csv
from html import escape
import json
from math import comb
from pathlib import Path
from typing import Any, Iterable


RECOVERY_DEFINITION = (
    "(checkpoint_accuracy - student_accuracy) / "
    "(teacher_accuracy - student_accuracy)"
)
RECOVERY_DENOMINATOR_EPSILON = 1e-6


def _read_accuracy_by_benchmark(
    evaluation_dir: Path, results_filename: str = "results.jsonl"
) -> dict[str, float]:
    path = evaluation_dir / results_filename
    grouped: dict[str, list[bool]] = {}
    try:
        # split("\n") rather than str.splitlines(): completions written with
        # ensure_ascii=False can contain U+2028 or U+0085, which JSON leaves
        # verbatim but str.splitlines() would treat as a record boundary.
        # Unlike splitlines(), split() keeps the empty tail a trailing newline
        # leaves behind, so blank entries are dropped here.
        lines = [
            line
            for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
    except OSError as exc:
        raise RuntimeError(f"cannot read evaluation results {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid JSON in {path} line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict) or not isinstance(row.get("dataset"), str):
            raise RuntimeError(f"invalid evaluation row in {path} line {line_number}")
        if not isinstance(row.get("correct"), bool):
            raise RuntimeError(f"evaluation row lacks boolean correct in {path} line {line_number}")
        grouped.setdefault(row["dataset"], []).append(row["correct"])
    if not grouped:
        raise RuntimeError(f"evaluation results are empty: {path}")
    scores = {
        benchmark: sum(values) / len(values)
        for benchmark, values in sorted(grouped.items())
    }
    overall = sum(scores.values()) / len(scores)
    return {"overall": overall, **scores}


def _pass_at_k_estimate(n: int, correct: int, k: int) -> float:
    if not 1 <= k <= n:
        raise ValueError("pass@k requires 1 <= k <= n")
    if not 0 <= correct <= n:
        raise ValueError("correct sample count must be between 0 and n")
    if n - correct < k:
        return 1.0
    return 1.0 - comb(n - correct, k) / comb(n, k)


class NonUniformSamplingError(RuntimeError):
    """A pass@k precondition failed, not a defect in the run.

    pass@k is only defined when every question was sampled the same number of
    times, but the size-tier evaluation policy deliberately varies
    samples_per_prompt per benchmark (avg@32 on a 30-question AIME, avg@4 on a
    500-question MATH-500). Those runs still produce a perfectly good recovery
    curve, so this is raised separately from real errors and callers degrade the
    pass@k artifact instead of discarding hours of finished evaluation.
    """


def _read_pass_at_k_by_benchmark(
    evaluation_dir: Path, results_filename: str = "results.jsonl"
) -> tuple[int, dict[str, dict[str, float]]]:
    path = evaluation_dir / results_filename
    grouped: dict[str, dict[str, list[bool]]] = {}
    try:
        # split("\n") rather than str.splitlines(): completions written with
        # ensure_ascii=False can contain U+2028 or U+0085, which JSON leaves
        # verbatim but str.splitlines() would treat as a record boundary.
        # Unlike splitlines(), split() keeps the empty tail a trailing newline
        # leaves behind, so blank entries are dropped here.
        lines = [
            line
            for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
    except OSError as exc:
        raise RuntimeError(f"cannot read evaluation results {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid JSON in {path} line {line_number}: {exc}"
            ) from exc
        dataset = row.get("dataset") if isinstance(row, dict) else None
        question_id = row.get("question_id") if isinstance(row, dict) else None
        correct = row.get("correct") if isinstance(row, dict) else None
        if not isinstance(dataset, str) or not isinstance(question_id, str):
            raise RuntimeError(
                f"pass@k row lacks dataset/question_id in {path} line {line_number}"
            )
        if not isinstance(correct, bool):
            raise RuntimeError(
                f"pass@k row lacks boolean correct in {path} line {line_number}"
            )
        grouped.setdefault(dataset, {}).setdefault(question_id, []).append(correct)
    questions = [
        samples
        for benchmark_questions in grouped.values()
        for samples in benchmark_questions.values()
    ]
    if not questions:
        raise RuntimeError(f"evaluation results are empty: {path}")
    sample_counts = {len(samples) for samples in questions}
    if len(sample_counts) != 1:
        raise NonUniformSamplingError(
            f"pass@k requires uniform samples per question: {path} has "
            f"{sorted(sample_counts)}"
        )
    n = sample_counts.pop()
    if n <= 0:
        raise RuntimeError(f"pass@k has no samples per question: {path}")

    k_values = []
    k = 1
    while k <= n:
        k_values.append(k)
        k *= 2

    def summarize(question_samples: list[list[bool]]) -> dict[str, float]:
        return {
            str(k): sum(
                _pass_at_k_estimate(len(samples), sum(samples), k)
                for samples in question_samples
            )
            / len(question_samples)
            for k in k_values
        }

    scores = {
        benchmark: summarize(list(benchmark_questions.values()))
        for benchmark, benchmark_questions in sorted(grouped.items())
    }
    overall = {
        str(k): sum(score[str(k)] for score in scores.values()) / len(scores)
        for k in k_values
    }
    return n, {"overall": overall, **scores}


def build_pass_at_k_report(
    experiment_dir: Path,
    evaluations: Iterable[dict[str, Any]],
    results_filename: str = "results.jsonl",
) -> dict[str, Any]:
    student_n, student_scores = _read_pass_at_k_by_benchmark(
        experiment_dir / "evaluate_student", results_filename
    )
    teacher_n, teacher_scores = _read_pass_at_k_by_benchmark(
        experiment_dir / "evaluate_teacher", results_filename
    )
    if student_n != teacher_n or student_scores.keys() != teacher_scores.keys():
        raise RuntimeError("student and teacher pass@k evaluation shapes differ")
    points = []
    for evaluation in evaluations:
        update = evaluation.get("update")
        evaluation_dir = evaluation.get("evaluation_dir")
        if not isinstance(update, int) or update < 0:
            continue
        if not isinstance(evaluation_dir, str):
            raise RuntimeError("checkpoint evaluation lacks evaluation_dir")
        checkpoint_n, checkpoint_scores = _read_pass_at_k_by_benchmark(
            Path(evaluation_dir), results_filename
        )
        if checkpoint_n != student_n or checkpoint_scores.keys() != student_scores.keys():
            raise RuntimeError(f"checkpoint {update} pass@k evaluation shape differs")
        recoveries = {
            benchmark: {
                k: _recovery(
                    checkpoint_scores[benchmark][k],
                    student_scores[benchmark][k],
                    teacher_scores[benchmark][k],
                )
                for k in checkpoint_scores[benchmark]
            }
            for benchmark in checkpoint_scores
        }
        points.append(
            {
                "step": update,
                "checkpoint_name": evaluation.get("checkpoint_name"),
                "pass_at_k": checkpoint_scores,
                "recovery_at_k": recoveries,
            }
        )
    points.sort(key=lambda point: point["step"])
    if not points:
        raise RuntimeError("pass@k curve requires at least one evaluated checkpoint")
    return {
        "status": "complete",
        "aggregation": "macro_average_over_benchmarks",
        "estimator": "1 - C(n-c, k) / C(n, k)",
        "recovery_definition": (
            "(checkpoint_pass_at_k - student_pass_at_k) / "
            "(teacher_pass_at_k - student_pass_at_k)"
        ),
        "recovery_denominator_epsilon": RECOVERY_DENOMINATOR_EPSILON,
        "max_k": student_n,
        "evaluated_k": [int(k) for k in student_scores["overall"]],
        "benchmarks": list(student_scores),
        "includes_step_zero": points[0]["step"] == 0,
        "baselines": {"student": student_scores, "teacher": teacher_scores},
        "points": points,
    }


def _recovery(
    checkpoint_score: float, student_score: float, teacher_score: float
) -> float | None:
    denominator = teacher_score - student_score
    # Recovery is only defined when the teacher establishes positive headroom
    # over the student. A negative gap can otherwise turn a worse checkpoint
    # into an apparently large positive recovery percentage.
    if denominator <= RECOVERY_DENOMINATOR_EPSILON:
        return None
    return (checkpoint_score - student_score) / denominator


def build_recovery_report(
    experiment_dir: Path,
    evaluations: Iterable[dict[str, Any]],
    results_filename: str = "results.jsonl",
    grader: dict[str, Any] | None = None,
) -> dict[str, Any]:
    student_scores = _read_accuracy_by_benchmark(
        experiment_dir / "evaluate_student", results_filename
    )
    teacher_scores = _read_accuracy_by_benchmark(
        experiment_dir / "evaluate_teacher", results_filename
    )
    if student_scores.keys() != teacher_scores.keys():
        raise RuntimeError("student and teacher benchmark sets differ")
    points = []
    for evaluation in evaluations:
        update = evaluation.get("update")
        evaluation_dir = evaluation.get("evaluation_dir")
        if not isinstance(update, int) or update < 0:
            continue
        if not isinstance(evaluation_dir, str):
            raise RuntimeError("checkpoint evaluation lacks evaluation_dir")
        checkpoint_scores = _read_accuracy_by_benchmark(
            Path(evaluation_dir), results_filename
        )
        if checkpoint_scores.keys() != student_scores.keys():
            raise RuntimeError(f"checkpoint {update} benchmark set differs from baselines")
        recoveries = {
            benchmark: _recovery(
                checkpoint_scores[benchmark],
                student_scores[benchmark],
                teacher_scores[benchmark],
            )
            for benchmark in student_scores
        }
        points.append(
            {
                "step": update,
                "checkpoint_name": evaluation.get("checkpoint_name"),
                "scores": checkpoint_scores,
                "recovery_ratio": recoveries,
            }
        )
    points.sort(key=lambda point: point["step"])
    if not points:
        raise RuntimeError("recovery curve requires at least one evaluated checkpoint")
    report = {
        "status": "complete",
        "aggregation": "macro_average_over_benchmarks",
        "definition": RECOVERY_DEFINITION,
        "denominator_epsilon": RECOVERY_DENOMINATOR_EPSILON,
        "benchmarks": list(student_scores),
        "includes_step_zero": points[0]["step"] == 0,
        "baselines": {"student": student_scores, "teacher": teacher_scores},
        "points": points,
    }
    if grader is not None:
        report["grader"] = grader
    return report


def _safe_benchmark_name(value: str) -> str:
    result = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    )
    return result or "benchmark"


def _write_step_recovery_svg(
    path: Path,
    *,
    title: str,
    series: list[tuple[str, list[tuple[int, float | None]]]],
    annotate_points: bool = False,
) -> None:
    width, height = 1080, 640
    left, right, top, bottom = 95, 260, 50, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    steps = sorted({step for _, values in series for step, _ in values})
    finite_values = [
        value for _, values in series for _, value in values if value is not None
    ]
    y_min = min(0.0, min(finite_values, default=0.0))
    y_max = max(1.0, max(finite_values, default=1.0))
    if y_min == y_max:
        y_max = y_min + 1.0

    def x_position(step: int) -> float:
        if steps[-1] == steps[0]:
            return left + plot_width / 2
        return left + (step - steps[0]) / (steps[-1] - steps[0]) * plot_width

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="32" y="30" font-family="sans-serif" font-size="20" font-weight="600">{escape(title)}</text>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_position(value)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{value * 100:.0f}%</text>')
    for step in steps:
        x = x_position(step)
        svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#f3f4f6"/>')
        svg.append(f'<text x="{x:.2f}" y="{top + plot_height + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{step}</text>')
    for reference, dash in ((0.0, ""), (1.0, ' stroke-dasharray="6 4"')):
        if y_min <= reference <= y_max:
            y = y_position(reference)
            svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#6b7280"{dash}/>')
    for series_index, (label, values) in enumerate(series):
        color = colors[series_index % len(colors)]
        finite_series = [(step, value) for step, value in values if value is not None]
        coordinates = " ".join(
            f'{x_position(step):.2f},{y_position(value):.2f}'
            for step, value in finite_series
        )
        if coordinates:
            svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for step, value in finite_series:
            x = x_position(step)
            y = y_position(value)
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}"><title>{escape(label)} step {step}: {value * 100:.2f}%</title></circle>')
            if annotate_points:
                svg.append(
                    f'<text x="{x:.2f}" y="{y - 10:.2f}" text-anchor="middle" '
                    'font-family="sans-serif" font-size="12" font-weight="600" '
                    f'fill="{color}">{value * 100:.1f}%</text>'
                )
        legend_y = top + 17 + series_index * 24
        svg.append(f'<line x1="{left + plot_width + 25}" y1="{legend_y}" x2="{left + plot_width + 53}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{left + plot_width + 62}" y="{legend_y + 4}" font-family="sans-serif" font-size="13">{escape(label)}</text>')
    svg.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="14">Optimizer step</text>',
            f'<text x="20" y="{top + plot_height / 2:.2f}" transform="rotate(-90 20 {top + plot_height / 2:.2f})" text-anchor="middle" font-family="sans-serif" font-size="14">Teacher-gap recovery</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _write_recovery_csv(csv_path: Path, report: dict[str, Any]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "benchmark",
                "checkpoint_accuracy",
                "student_accuracy",
                "teacher_accuracy",
                "recovery_ratio",
            ]
        )
        for point in report["points"]:
            for benchmark in report["benchmarks"]:
                writer.writerow(
                    [
                        point["step"],
                        benchmark,
                        point["scores"][benchmark],
                        report["baselines"]["student"][benchmark],
                        report["baselines"]["teacher"][benchmark],
                        point["recovery_ratio"][benchmark],
                    ]
                )


def write_recovery_artifacts(
    experiment_dir: Path,
    output_dir: Path,
    evaluations: Iterable[dict[str, Any]],
    *,
    results_filename: str = "results.jsonl",
    grader: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation_list = list(evaluations)
    report = build_recovery_report(
        experiment_dir,
        evaluation_list,
        results_filename=results_filename,
        grader=grader,
    )
    # The recovery curve is the primary artifact and does not depend on pass@k.
    # A size-tier evaluation policy varies samples_per_prompt per benchmark,
    # which pass@k cannot express, and letting that precondition abort the whole
    # function would throw away every finished evaluation in the run.
    try:
        pass_report = build_pass_at_k_report(
            experiment_dir,
            evaluation_list,
            results_filename=results_filename,
        )
    except NonUniformSamplingError as exc:
        pass_report = None
        pass_at_k_status = {"status": "unavailable", "reason": str(exc)}
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "recovery_curve.json"
    csv_path = output_dir / "recovery_curve.csv"
    svg_path = output_dir / "recovery_curve.svg"
    overall_svg_path = output_dir / "recovery_curve_overall.svg"
    pass_json_path = output_dir / "pass_at_k_curve.json"
    pass_csv_path = output_dir / "pass_at_k_curve.csv"
    recovery_by_benchmark = [
        (
            benchmark,
            [
                (point["step"], point["recovery_ratio"][benchmark])
                for point in report["points"]
            ],
        )
        for benchmark in report["benchmarks"]
        if benchmark != "overall"
    ]
    _write_step_recovery_svg(
        svg_path,
        title="pass@1 recovery by benchmark",
        series=recovery_by_benchmark,
    )
    overall_series = [
        (
            "overall pass@1",
            [
                (point["step"], point["recovery_ratio"]["overall"])
                for point in report["points"]
            ],
        )
    ]
    _write_step_recovery_svg(
        overall_svg_path,
        title="Overall pass@1 recovery over training",
        series=overall_series,
        annotate_points=True,
    )
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_json.replace(json_path)
    temporary_pass_json = pass_json_path.with_suffix(".json.tmp")
    temporary_pass_json.write_text(
        json.dumps(pass_report if pass_report is not None else pass_at_k_status,
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_pass_json.replace(pass_json_path)
    if pass_report is None:
        _write_recovery_csv(csv_path, report)
        return {
            "json": str(json_path),
            "csv": str(csv_path),
            "svg": str(svg_path),
            "overall_svg": str(overall_svg_path),
            "definition": RECOVERY_DEFINITION,
            "pass_at_k": {"json": str(pass_json_path), **pass_at_k_status},
        }
    with pass_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["series", "step", "benchmark", "k", "pass_at_k", "recovery_at_k"])
        for series_name in ("student", "teacher"):
            for benchmark in pass_report["benchmarks"]:
                for k in pass_report["evaluated_k"]:
                    writer.writerow(
                        [
                            series_name,
                            "",
                            benchmark,
                            k,
                            pass_report["baselines"][series_name][benchmark][str(k)],
                            "",
                        ]
                    )
        for point in pass_report["points"]:
            for benchmark in pass_report["benchmarks"]:
                for k in pass_report["evaluated_k"]:
                    writer.writerow(
                        [
                            "checkpoint",
                            point["step"],
                            benchmark,
                            k,
                            point["pass_at_k"][benchmark][str(k)],
                            point["recovery_at_k"][benchmark][str(k)],
                        ]
                    )
    pass_svg_paths = {}
    for benchmark in (
        benchmark for benchmark in pass_report["benchmarks"] if benchmark != "overall"
    ):
        pass_svg_path = output_dir / (
            f"pass_at_k_curve_{_safe_benchmark_name(benchmark)}.svg"
        )
        _write_step_recovery_svg(
            pass_svg_path,
            title=f"pass@k recovery over training — {benchmark}",
            series=[
                (
                    f"pass@{k}",
                    [
                        (point["step"], point["recovery_at_k"][benchmark][str(k)])
                        for point in pass_report["points"]
                    ],
                )
                for k in pass_report["evaluated_k"]
            ],
        )
        pass_svg_paths[benchmark] = str(pass_svg_path)
    # Keep the historical path fresh so old automation does not expose a stale chart.
    legacy_overall_svg_path = output_dir / "pass_at_k_curve_overall.svg"
    _write_step_recovery_svg(
        legacy_overall_svg_path,
        title="Overall pass@1 recovery over training",
        series=overall_series,
    )
    _write_recovery_csv(csv_path, report)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "svg": str(svg_path),
        "overall_svg": str(overall_svg_path),
        "definition": RECOVERY_DEFINITION,
        "pass_at_k": {
            "json": str(pass_json_path),
            "csv": str(pass_csv_path),
            "svg": pass_svg_paths,
            "pass_at_1_by_benchmark_svg": str(svg_path),
            "overall_pass_at_1_svg": str(overall_svg_path),
            "max_k": pass_report["max_k"],
            "evaluated_k": pass_report["evaluated_k"],
        },
    }


__all__ = [
    "NonUniformSamplingError",
    "build_pass_at_k_report",
    "build_recovery_report",
    "write_recovery_artifacts",
]
