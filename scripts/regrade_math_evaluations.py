#!/usr/bin/env python3
"""Regrade completed math runs with the published Swallow verifier.

The original ``results.jsonl`` files remain immutable evidence.  This command
writes ``results.official_math.jsonl`` sidecars, rebuilds the recovery artifacts
from those sidecars, and optionally publishes the portable JSON curve under
``experiments/results``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import shutil
from typing import Any

from huggingface_hub import hf_hub_download

from soft_prompt_distillation.common import _select, read_dataset_rows, sha256_file
from soft_prompt_distillation.config import load_config
from soft_prompt_distillation.reporting import write_recovery_artifacts
from soft_prompt_distillation.tasks import TaskAdapter, get_task
from soft_prompt_distillation.tasks.official_math_grader import (
    SWALLOW_MATH_GRADER,
    prime_official_math_reference,
)


RESULTS_SIDECAR = "results.official_math.jsonl"
SWALLOW_EVALUATION_COMMIT = "f9f7c69875777ce95fe0454bbe5ed5cd13a2dcd9"

_WORKER_ROWS: list[dict[str, Any]] = []
_WORKER_REFERENCES: dict[tuple[str, str], dict[str, Any]] = {}
_WORKER_TASK: TaskAdapter | None = None
_WORKER_HARD_TIMEOUT: float | None = None


def _manifest_datasets(run: Path) -> list[dict[str, Any]]:
    path = run / "evaluate_student" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    datasets = manifest.get("source", {}).get("datasets")
    if not isinstance(datasets, list) or not all(
        isinstance(item, dict) for item in datasets
    ):
        raise RuntimeError(f"baseline manifest has no dataset identities: {path}")
    return datasets


def _reference_rows(
    run: Path, cache_dir: Path, task: TaskAdapter
) -> dict[tuple[str, str], dict[str, Any]]:
    config = load_config(run / "config.yaml")
    identities = iter(_manifest_datasets(run))
    normalized: list[dict[str, Any]] = []
    for label, selection in config.data.evaluation.items():
        tagged: list[dict[str, Any]] = []
        for source_index, source in enumerate(task.get_evaluation_datasets(label)):
            try:
                identity = next(identities)
            except StopIteration as exc:
                raise RuntimeError("baseline manifest has too few dataset identities") from exc
            if identity.get("name") != source.repo or identity.get("filename") != Path(
                source.filename
            ).name:
                raise RuntimeError(
                    "dataset identity order differs from config: "
                    f"expected {source.repo}/{source.filename}, got {identity}"
                )
            revision = identity.get("resolved_revision")
            if not isinstance(revision, str) or not revision:
                raise RuntimeError(f"dataset identity has no revision: {identity}")
            path = Path(
                hf_hub_download(
                    source.repo,
                    source.filename,
                    repo_type="dataset",
                    revision=revision,
                    cache_dir=cache_dir,
                )
            )
            expected_sha = identity.get("content_sha256")
            if expected_sha != sha256_file(path):
                raise RuntimeError(
                    f"dataset content differs from recorded manifest: {source.repo}"
                )
            tagged.extend(
                {"source_index": source_index, "row": row}
                for row in read_dataset_rows(path)
            )
        for tagged_row in _select(
            tagged, selection.num_prompts, selection.selection_seed
        ):
            normalized.append(task.normalize_evaluation_row(tagged_row["row"], label))
    try:
        next(identities)
    except StopIteration:
        pass
    else:
        raise RuntimeError("baseline manifest has extra dataset identities")

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in normalized:
        key = (str(row["dataset"]), str(row["question_id"]))
        previous = result.get(key)
        if previous is not None and (
            previous.get("evaluation_reference")
            != row.get("evaluation_reference")
            or previous.get("evaluation_reference_format")
            != row.get("evaluation_reference_format")
        ):
            raise RuntimeError(f"conflicting evaluation references for {key}")
        result[key] = row
    return result


def _result_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _grade_direct(index: int) -> bool:
    row = _WORKER_ROWS[index]
    key = (str(row.get("dataset")), str(row.get("question_id")))
    reference = _WORKER_REFERENCES[key]
    completion = row.get("raw_completion")
    if not isinstance(completion, str):
        completion = row.get("completion")
    if not isinstance(completion, str):
        raise RuntimeError(f"result has no completion text for {key}")
    if _WORKER_TASK is None:
        raise RuntimeError("regrade worker was not initialized")
    return _WORKER_TASK.grade_evaluation(completion, reference)


def _grade_child(index: int, connection: Any) -> None:
    try:
        connection.send(("ok", _grade_direct(index)))
    except BaseException as exc:  # pragma: no cover - exercised in child process
        connection.send(("error", repr(exc)))
    finally:
        connection.close()


def _grade_worker_index(index: int) -> tuple[bool, bool]:
    if _WORKER_HARD_TIMEOUT is None:
        return _grade_direct(index), False
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_grade_child, args=(index, send))
    process.start()
    send.close()
    process.join(_WORKER_HARD_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join()
        receive.close()
        return False, True
    if not receive.poll():
        receive.close()
        return False, True
    status, value = receive.recv()
    receive.close()
    if status == "error":
        raise RuntimeError(f"grader child failed for result index {index}: {value}")
    return bool(value), False


def _regrade_directory(
    directory: Path,
    references: dict[tuple[str, str], dict[str, Any]],
    task: TaskAdapter,
    workers: int,
    hard_timeout: float | None,
    resume: bool,
) -> dict[str, Any]:
    source = directory / "results.jsonl"
    target = directory / RESULTS_SIDECAR
    summary_path = directory / "summary.official_math.json"
    if resume and target.is_file() and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("grader") == SWALLOW_MATH_GRADER
            and summary.get("source_results") == source.name
            and summary.get("results") == target.name
        ):
            return summary
    rows = _result_rows(source)
    for row in rows:
        key = (str(row.get("dataset")), str(row.get("question_id")))
        if key not in references:
            raise RuntimeError(f"no recorded reference for result {key} in {source}")
        if not isinstance(row.get("correct"), bool):
            raise RuntimeError(f"result has no legacy boolean grade for {key} in {source}")

    if workers == 1:
        global _WORKER_ROWS, _WORKER_REFERENCES, _WORKER_TASK, _WORKER_HARD_TIMEOUT
        _WORKER_ROWS = rows
        _WORKER_REFERENCES = references
        _WORKER_TASK = task
        _WORKER_HARD_TIMEOUT = hard_timeout
        grade_values = [_grade_worker_index(index) for index in range(len(rows))]
    else:
        # The supported execution surface is Linux. Forking after assigning
        # these read-only globals avoids serializing multi-hundred-MB completion
        # rows through the worker queue, while dynamic one-row scheduling keeps
        # individual 5-second parser timeouts from blocking an entire checkpoint.
        _WORKER_ROWS = rows
        _WORKER_REFERENCES = references
        _WORKER_TASK = task
        _WORKER_HARD_TIMEOUT = hard_timeout
        for reference in references.values():
            prime_official_math_reference(
                str(reference["evaluation_reference"]),
                str(reference["evaluation_reference_format"]),
            )
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            grade_values = list(
                executor.map(_grade_worker_index, range(len(rows)), chunksize=1)
            )

    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "samples": 0,
            "correct": 0,
            "legacy_correct": 0,
            "changed": 0,
            "hard_timeouts": 0,
        }
    )
    regraded = []
    for row, (correct, hard_timed_out) in zip(rows, grade_values, strict=True):
        key = (str(row.get("dataset")), str(row.get("question_id")))
        legacy = row.get("correct")
        assert isinstance(legacy, bool)
        item = dict(row)
        item["legacy_correct"] = legacy
        item["correct"] = correct
        item["grader"] = task.evaluation_grader(references[key])
        item["grader_hard_timeout"] = hard_timed_out
        regraded.append(item)
        benchmark = counts[key[0]]
        benchmark["samples"] += 1
        benchmark["legacy_correct"] += int(legacy)
        benchmark["correct"] += int(correct)
        benchmark["changed"] += int(legacy != correct)
        benchmark["hard_timeouts"] += int(hard_timed_out)

    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in regraded:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(target)
    summary = {
        "grader": SWALLOW_MATH_GRADER,
        "source_results": source.name,
        "results": target.name,
        "outer_hard_timeout_seconds": hard_timeout,
        "benchmarks": {
            benchmark: {
                **values,
                "accuracy": values["correct"] / values["samples"],
                "legacy_accuracy": values["legacy_correct"] / values["samples"],
            }
            for benchmark, values in sorted(counts.items())
        },
    }
    _write_json_atomic(summary_path, summary)
    return summary


def regrade_run(
    run: Path,
    cache_dir: Path,
    publish_dir: Path | None,
    workers: int = 1,
    hard_timeout: float | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    config = load_config(run / "config.yaml")
    task = get_task(config.task)
    references = _reference_rows(run, cache_dir, task)
    checkpoint_root = run / "evaluate_student_soft_prompt"
    checkpoint_dirs = sorted(
        path
        for path in checkpoint_root.glob("update_*")
        if (path / "results.jsonl").is_file()
    )
    directories = [
        run / "evaluate_student",
        run / "evaluate_teacher",
        *checkpoint_dirs,
    ]
    summary_values = [
        _regrade_directory(
            directory,
            references,
            task,
            workers,
            hard_timeout,
            resume,
        )
        for directory in directories
    ]
    summaries = {
        directory.name: summary
        for directory, summary in zip(directories, summary_values, strict=True)
    }
    evaluations = [
        {
            "checkpoint_name": directory.name,
            "update": int(directory.name.removeprefix("update_")),
            "evaluation_dir": str(directory),
        }
        for directory in checkpoint_dirs
    ]
    grader = {
        "name": SWALLOW_MATH_GRADER,
        "implementation": "math-verify==0.5.2",
        "swallow_evaluation_commit": SWALLOW_EVALUATION_COMMIT,
        "source_results": "results.jsonl",
        "regraded_results": RESULTS_SIDECAR,
    }
    if hard_timeout is not None:
        grader["outer_hard_timeout_seconds"] = hard_timeout
    artifacts = write_recovery_artifacts(
        run,
        checkpoint_root,
        evaluations,
        results_filename=RESULTS_SIDECAR,
        grader=grader,
    )
    if publish_dir is not None:
        publish_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            artifacts["json"], publish_dir / f"{run.name}.recovery_curve.json"
        )
    result = {
        "run": run.name,
        "references": len(references),
        "checkpoints": len(checkpoint_dirs),
        "grader": grader,
        "summaries": summaries,
        "curve": artifacts["json"],
    }
    _write_json_atomic(checkpoint_root / "official_math_regrade.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--publish-dir", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="result rows to regrade in parallel (default: 1)",
    )
    parser.add_argument(
        "--hard-timeout",
        type=float,
        help="kill a grader subprocess after this many seconds and mark it false",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete official sidecars already present in a run",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.hard_timeout is not None and args.hard_timeout <= 0:
        parser.error("--hard-timeout must be positive")
    for run in args.runs:
        result = regrade_run(
            run,
            args.cache_dir,
            args.publish_dir,
            args.workers,
            args.hard_timeout,
            args.resume,
        )
        print(
            f"{result['run']}: {result['checkpoints']} checkpoints, "
            f"{result['references']} references -> {result['curve']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
