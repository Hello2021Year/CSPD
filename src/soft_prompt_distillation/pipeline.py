"""Full-chain orchestration; stage implementations live in dedicated modules."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile

from .common import RunOptions, snapshot_config
from .distillation import train
from .evaluation import evaluate
from .exporting import export_soft_prompt_model
from .reporting import write_recovery_artifacts
from .sampling import sample


def resolve_checkpoint_paths(train_dir: Path, selector: str) -> list[Path]:
    """Resolve one numbered checkpoint or every checkpoint in requested order."""

    checkpoint_dir = train_dir / "checkpoints"
    if selector == "final":
        path = train_dir / "soft_prompt.safetensors"
        if not path.is_file():
            raise RuntimeError(f"final checkpoint does not exist: {path}")
        return [path]
    if selector in {"all", "all-desc"}:
        paths = sorted(checkpoint_dir.glob("update_*.safetensors"))
        if not paths:
            raise RuntimeError(f"no intermediate checkpoints found in {checkpoint_dir}")
        if selector == "all-desc":
            paths.reverse()
        return paths
    if selector.startswith("every:"):
        interval_text = selector.removeprefix("every:")
        try:
            interval = int(interval_text)
        except ValueError as exc:
            raise RuntimeError(
                "periodic checkpoint selector must be every:<positive integer>"
            ) from exc
        if interval <= 0 or str(interval) != interval_text:
            raise RuntimeError(
                "periodic checkpoint selector must be every:<positive integer>"
            )
        paths = sorted(checkpoint_dir.glob("update_*.safetensors"))
        if not paths:
            raise RuntimeError(f"no intermediate checkpoints found in {checkpoint_dir}")
        selected = [
            path
            for path in paths
            if int(path.stem.removeprefix("update_")) % interval == 0
        ]
        final_path = paths[-1]
        if (train_dir / "manifest.json").is_file() and final_path not in selected:
            selected.append(final_path)
        return selected
    try:
        update = int(selector)
    except ValueError as exc:
        raise RuntimeError(
            "checkpoint must be a nonnegative update number, final, all, all-desc, "
            "or every:<positive integer>"
        ) from exc
    if update < 0 or str(update) != selector:
        raise RuntimeError(
            "checkpoint must be a nonnegative update number, final, all, all-desc, "
            "or every:<positive integer>"
        )
    path = checkpoint_dir / f"update_{update:08d}.safetensors"
    if not path.is_file():
        raise RuntimeError(f"checkpoint for update {update} does not exist: {path}")
    return [path]


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON receipt is not an object: {path}")
    return value


def _write_json_object(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _quarantine_incomplete_directory(path: Path) -> Path:
    """Preserve a partial stage while freeing its canonical path for retry."""

    counter = 1
    while True:
        quarantine = path.with_name(f"{path.name}.incomplete-{counter:03d}")
        if not quarantine.exists():
            break
        counter += 1
    path.replace(quarantine)
    print(
        f"checkpoint evaluation: moved incomplete directory {path} to "
        f"{quarantine}",
        flush=True,
    )
    return quarantine


def _collect_checkpoint_evaluations(
    output_dir: Path, train_dir: Path
) -> list[dict[str, object]]:
    evaluations: list[dict[str, object]] = []
    for summary_path in sorted(output_dir.glob("*/summary.json")):
        name = summary_path.parent.name
        if name == "final":
            checkpoint_path = train_dir / "soft_prompt.safetensors"
            update: int | None = None
        elif name.startswith("update_") and name.removeprefix("update_").isdigit():
            update = int(name.removeprefix("update_"))
            checkpoint_path = (
                train_dir / "checkpoints" / f"update_{update:08d}.safetensors"
            )
        else:
            continue
        evaluations.append(
            {
                "checkpoint_name": name,
                "update": update,
                "checkpoint": str(checkpoint_path),
                "evaluation_dir": str(summary_path.parent),
                "summary": _read_json_object(summary_path),
            }
        )
    return evaluations


def _refresh_checkpoint_report(
    options: RunOptions, train_dir: Path, selector: str
) -> Path:
    """Atomically refresh incremental checkpoint metrics and recovery plots."""

    evaluations = _collect_checkpoint_evaluations(options.output, train_dir)
    report: dict[str, object] = {
        "status": "complete",
        "checkpoint": selector,
        "evaluations": evaluations,
    }
    if selector in {"all", "all-desc"} or selector.startswith("every:"):
        experiment_dir = train_dir.parent
        baseline_results = [
            experiment_dir / "evaluate_student" / "results.jsonl",
            experiment_dir / "evaluate_teacher" / "results.jsonl",
        ]
        if all(path.is_file() for path in baseline_results):
            artifacts = write_recovery_artifacts(
                experiment_dir, options.output, evaluations
            )
            recovery_report = _read_json_object(Path(artifacts["json"]))
            points = recovery_report.get("points")
            if not isinstance(points, list):
                raise RuntimeError("recovery report does not contain point rows")
            points_by_step = {
                point["step"]: point
                for point in points
                if isinstance(point, dict) and isinstance(point.get("step"), int)
            }
            for evaluation in evaluations:
                update = evaluation.get("update")
                point = points_by_step.get(update)
                if point is None:
                    continue
                recovery = {
                    "definition": recovery_report.get("definition"),
                    "aggregation": recovery_report.get("aggregation"),
                    "baselines": recovery_report.get("baselines"),
                    "scores": point.get("scores"),
                    "recovery_ratio": point.get("recovery_ratio"),
                }
                recovery_path = Path(str(evaluation["evaluation_dir"])) / "recovery.json"
                _write_json_object(recovery_path, recovery)
                evaluation["recovery"] = recovery
                evaluation["recovery_path"] = str(recovery_path)
            report["recovery_curve"] = artifacts
        else:
            report["recovery_curve"] = {
                "status": "pending",
                "reason": "student and teacher baseline evaluations are required",
            }

    report_path = options.output / "checkpoint_summary.json"
    _write_json_object(report_path, report)
    return report_path


def evaluate_checkpoints(
    options: RunOptions,
    train_dir: Path,
    selector: str,
    temporary_dir: Path | None = None,
) -> Path:
    """Evaluate selected prompt-only checkpoints, retaining receipts but not model copies."""

    checkpoints = resolve_checkpoint_paths(train_dir, selector)
    options.output.mkdir(parents=True, exist_ok=True)
    snapshot_config(options)
    if temporary_dir is not None:
        temporary_dir.mkdir(parents=True, exist_ok=True)
    required = (
        "results.jsonl",
        "summary.json",
        "manifest.json",
        "export_manifest.json",
    )
    report_path: Path | None = None
    for checkpoint in checkpoints:
        evaluation_name = (
            "final" if checkpoint.name == "soft_prompt.safetensors" else checkpoint.stem
        )
        evaluation_dir = options.output / evaluation_name
        if evaluation_dir.exists():
            missing = [
                name for name in required if not (evaluation_dir / name).is_file()
            ]
            if missing:
                _quarantine_incomplete_directory(evaluation_dir)
            else:
                continue
        evaluation_staging = options.output / f".{evaluation_name}.staging"
        if evaluation_staging.exists():
            _quarantine_incomplete_directory(evaluation_staging)
        with tempfile.TemporaryDirectory(
            prefix=f"soft-prompt-{checkpoint.stem}-",
            dir=temporary_dir,
        ) as staging_name:
            derived_model = Path(staging_name) / "student_soft_prompt_model"
            export_options = replace(
                options, output=derived_model, soft_prompt_model_path=None
            )
            export_soft_prompt_model(export_options, checkpoint)
            eval_options = replace(
                options,
                output=evaluation_staging,
                soft_prompt_model_path=derived_model,
            )
            evaluate(eval_options, "student_soft_prompt")
            shutil.copy2(
                derived_model / "manifest.json",
                evaluation_staging / "export_manifest.json",
            )
            metadata = derived_model / "soft_prompt_tokens.json"
            if metadata.is_file():
                shutil.copy2(metadata, evaluation_staging / metadata.name)
            evaluation_staging.replace(evaluation_dir)
        report_path = _refresh_checkpoint_report(options, train_dir, selector)

    # A resume in which every requested checkpoint is already complete still
    # needs one final report refresh, but recomputing the recovery/pass@k
    # artifacts once per existing checkpoint is needlessly quadratic in the
    # number and size of result files.
    if report_path is None:
        report_path = _refresh_checkpoint_report(options, train_dir, selector)
    return report_path


def evaluate_target(
    options: RunOptions,
    target: str,
    *,
    train_dir: Path | None = None,
    checkpoint: str | None = None,
    temporary_dir: Path | None = None,
) -> Path:
    """Evaluate a plain model or one-or-many learned soft prompts."""

    if target in {"student", "teacher"}:
        if train_dir is not None or checkpoint is not None or temporary_dir is not None:
            raise RuntimeError(
                "--train-dir, --checkpoint, and --temporary-dir are only valid "
                "with --target soft-prompt"
            )
        return evaluate(options, target)
    if target != "soft-prompt":
        raise RuntimeError(f"unsupported evaluation target: {target!r}")
    if train_dir is None:
        raise RuntimeError("--target soft-prompt requires --train-dir")
    return evaluate_checkpoints(
        options, train_dir, checkpoint or "final", temporary_dir
    )


def run_smoke(options: RunOptions) -> Path:
    options.output.mkdir(parents=True, exist_ok=False)
    snapshot_config(options)
    common = {
        "config": options.config,
        "cache_dir": options.cache_dir,
        "student_model_path": options.student_model_path,
        "teacher_model_path": options.teacher_model_path,
        "sglang_tp_size": options.sglang_tp_size,
        "sglang_dp_size": options.sglang_dp_size,
        "sglang_batch_size": options.sglang_batch_size,
        "sglang_log_level": options.sglang_log_level,
        "train_model_placement": options.train_model_placement,
    }
    sample_options = RunOptions(output=options.output / "sample", **common)
    train_options = RunOptions(output=options.output / "train", **common)
    snapshot = sample(sample_options)
    train(train_options, snapshot)
    eval_options = RunOptions(
        output=options.output / "evaluate_student_soft_prompt", **common
    )
    return evaluate_target(
        eval_options,
        "soft-prompt",
        train_dir=train_options.output,
    )


__all__ = [
    "RunOptions",
    "evaluate",
    "evaluate_checkpoints",
    "evaluate_target",
    "export_soft_prompt_model",
    "resolve_checkpoint_paths",
    "run_smoke",
    "sample",
    "train",
]
