from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from soft_prompt_distillation.reporting import (
    NonUniformSamplingError,
    write_recovery_artifacts,
)


def _write_results(path, spec):
    """spec: {benchmark: (questions, samples_per_question, correct_per_question)}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for benchmark, (questions, samples, correct) in spec.items():
            for q in range(questions):
                for s in range(samples):
                    handle.write(json.dumps({
                        "dataset": benchmark,
                        "question_id": f"{benchmark}-{q}",
                        "correct": s < correct,
                    }) + "\n")
    summary = {
        "accuracy": 0.5,
        "benchmarks": {
            b: {"accuracy": c / s, "questions": q, "samples": q * s}
            for b, (q, s, c) in spec.items()
        },
    }
    (path.parent / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


# avg@4 on a 500-question benchmark and avg@8 on an 83-question one is the
# size-tier policy several configs use on purpose, not a broken run.
SIZE_TIERED = {"math_500": (4, 4, 2), "amc_2022_2023": (4, 8, 4)}
UNIFORM = {"math_500": (4, 4, 2), "amc_2022_2023": (4, 4, 2)}


def _build(tmp_path, spec, checkpoint_spec=None):
    run = tmp_path / "run"
    _write_results(run / "evaluate_student" / "results.jsonl", spec)
    _write_results(run / "evaluate_teacher" / "results.jsonl", spec)
    out = run / "evaluate_student_soft_prompt"
    evaluations = []
    # Include step 0 in the complete-curve fixture.
    for update in (0, 20):
        ckpt = out / f"update_{update:08d}"
        _write_results(ckpt / "results.jsonl", checkpoint_spec or spec)
        evaluations.append({
            "checkpoint_name": ckpt.name,
            "update": update,
            "evaluation_dir": str(ckpt),
            "summary": json.loads((ckpt / "summary.json").read_text()),
        })
    return run, out, evaluations


def test_size_tiered_evaluation_still_writes_the_recovery_curve(tmp_path):
    run, out, evaluations = _build(tmp_path, SIZE_TIERED)

    result = write_recovery_artifacts(run, out, evaluations)

    # The recovery curve is the primary artifact and must survive.
    assert (out / "recovery_curve.json").is_file()
    assert (out / "recovery_curve.csv").is_file()
    assert (out / "recovery_curve.svg").is_file()
    assert (out / "recovery_curve_overall.svg").is_file()
    assert result["svg"] == str(out / "recovery_curve.svg")
    assert result["overall_svg"] == str(out / "recovery_curve_overall.svg")
    benchmark_svg = (out / "recovery_curve.svg").read_text(encoding="utf-8")
    assert 'font-weight="600" fill=' not in benchmark_svg
    assert result["pass_at_k"]["status"] == "unavailable"
    assert "uniform samples per question" in result["pass_at_k"]["reason"]

    # The stub records why, rather than leaving a stale or absent file.
    stub = json.loads((out / "pass_at_k_curve.json").read_text())
    assert stub["status"] == "unavailable"


def test_uniform_evaluation_still_produces_pass_at_k(tmp_path):
    run, out, evaluations = _build(tmp_path, UNIFORM)

    result = write_recovery_artifacts(run, out, evaluations)

    assert (out / "recovery_curve.json").is_file()
    assert "status" not in result["pass_at_k"]
    assert result["pass_at_k"]["evaluated_k"]
    assert (out / "pass_at_k_curve.csv").is_file()


def test_non_uniform_error_is_its_own_type(tmp_path):
    # A genuine defect must still raise, so the degrade path is narrow.
    assert issubclass(NonUniformSamplingError, RuntimeError)


def test_recovery_can_use_a_regraded_results_sidecar(tmp_path):
    run, out, evaluations = _build(tmp_path, UNIFORM)
    for directory in [
        run / "evaluate_student",
        run / "evaluate_teacher",
        *(Path(item["evaluation_dir"]) for item in evaluations),
    ]:
        original = directory / "results.jsonl"
        rows = [json.loads(line) for line in original.read_text().splitlines()]
        for row in rows:
            row["correct"] = True
        (directory / "results.official_math.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    write_recovery_artifacts(
        run,
        out,
        evaluations,
        results_filename="results.official_math.jsonl",
        grader={"name": "official-math", "source": "unit-test"},
    )

    curve = json.loads((out / "recovery_curve.json").read_text())
    assert curve["grader"] == {"name": "official-math", "source": "unit-test"}
    assert curve["baselines"]["student"]["overall"] == 1.0
