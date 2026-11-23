#!/usr/bin/env python3
"""Create an auditable uniform q-k subsample of a completed teacher snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_sample_indices(
    source_row_index: int,
    *,
    source_samples_per_prompt: int,
    selected_samples_per_prompt: int,
    selection_seed: int,
) -> frozenset[int]:
    """Return one reproducible uniform subset, independently for each prompt."""

    payload = f"{selection_seed}:{source_row_index}".encode()
    row_seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
    selected = random.Random(row_seed).sample(
        range(source_samples_per_prompt), selected_samples_per_prompt
    )
    return frozenset(selected)


def _git_metadata() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"])
    return {
        "commit": commit,
        "dirty": bool(diff),
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def subsample_snapshot(
    source_sample_dir: Path,
    output_sample_dir: Path,
    config: Path,
    *,
    source_samples_per_prompt: int,
    selected_samples_per_prompt: int,
    selection_seed: int,
) -> dict[str, Any]:
    if not 0 < selected_samples_per_prompt <= source_samples_per_prompt:
        raise ValueError("selected sample count must be in [1, source sample count]")
    if output_sample_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_sample_dir}")

    source_snapshot = source_sample_dir / "teacher_trajectories.jsonl"
    source_manifest_path = source_sample_dir / "manifest.json"
    source_config = source_sample_dir / "config.yaml"
    for path in (source_snapshot, source_manifest_path, source_config, config):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    expected_snapshot_sha = source_manifest["content_sha256"][
        "teacher_trajectories.jsonl"
    ]
    actual_snapshot_sha = sha256_file(source_snapshot)
    if actual_snapshot_sha != expected_snapshot_sha:
        raise RuntimeError("source snapshot checksum differs from its manifest")

    output_sample_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_sample_dir.name}.tmp-",
            dir=output_sample_dir.parent,
        )
    )
    output_snapshot = temporary / "teacher_trajectories.jsonl"
    selected_cache: dict[int, frozenset[int]] = {}
    sample_index_counts: Counter[int] = Counter()
    kept_question_ids: set[str] = set()
    input_trajectories = 0
    output_trajectories = 0
    seen_keys: set[tuple[int, int]] = set()
    try:
        with source_snapshot.open(encoding="utf-8") as source, output_snapshot.open(
            "w", encoding="utf-8"
        ) as output:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                input_trajectories += 1
                record = json.loads(line)
                source_row_index = record.get("source_row_index")
                sample_index = record.get("sample_index")
                if (
                    not isinstance(source_row_index, int)
                    or source_row_index < 0
                    or not isinstance(sample_index, int)
                    or not 0 <= sample_index < source_samples_per_prompt
                ):
                    raise RuntimeError(
                        f"invalid source/sample index at {source_snapshot}:{line_number}"
                    )
                key = (source_row_index, sample_index)
                if key in seen_keys:
                    raise RuntimeError(f"duplicate source/sample key {key}")
                seen_keys.add(key)
                selected = selected_cache.setdefault(
                    source_row_index,
                    selected_sample_indices(
                        source_row_index,
                        source_samples_per_prompt=source_samples_per_prompt,
                        selected_samples_per_prompt=selected_samples_per_prompt,
                        selection_seed=selection_seed,
                    ),
                )
                if sample_index not in selected:
                    continue
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_trajectories += 1
                sample_index_counts[sample_index] += 1
                question_id = record.get("question_id")
                if isinstance(question_id, str):
                    kept_question_ids.add(question_id)

        shutil.copyfile(config, temporary / "config.yaml")
        output_snapshot_sha = sha256_file(output_snapshot)
        original_counts = source_manifest["record_counts"]
        original_sampling = deepcopy(source_manifest["source"]["sampling"])
        derived_sampling = deepcopy(original_sampling)
        derived_sampling["generation"]["samples_per_prompt"] = (
            selected_samples_per_prompt
        )
        derived_sampling["generation"]["provenance"] = (
            f"uniform random q{selected_samples_per_prompt}-of-q"
            f"{source_samples_per_prompt} subsample; selection seed {selection_seed}"
        )
        derived_sampling["subsampling"] = {
            "method": "uniform_without_replacement_per_source_row",
            "source_samples_per_prompt": source_samples_per_prompt,
            "selected_samples_per_prompt": selected_samples_per_prompt,
            "selection_seed": selection_seed,
            "row_seed": "first_128_bits_of_sha256(f'{selection_seed}:{source_row_index}')",
            "selection_domain": list(range(source_samples_per_prompt)),
            "selection_occurs_before_correct_only_filter": True,
        }
        source_identity = deepcopy(source_manifest["source"])
        source_identity["config"] = str(config.resolve())
        source_identity["sampling"] = derived_sampling
        selected_request_count = (
            int(original_counts["questions"]) * selected_samples_per_prompt
        )
        manifest = {
            "stage": "sample",
            "source": source_identity,
            "record_counts": {
                "source_rows": int(original_counts["source_rows"]),
                "questions": int(original_counts["questions"]),
                "generation_requests": selected_request_count,
                "correct_trajectories_before_deduplication": output_trajectories,
                "duplicate_responses_filtered": 0,
                "trajectories": output_trajectories,
                "questions_with_kept_trajectory": len(kept_question_ids),
                "source_trajectories": input_trajectories,
                "sample_index_counts": {
                    str(index): sample_index_counts[index]
                    for index in range(source_samples_per_prompt)
                },
            },
            "content_sha256": {
                "config.yaml": sha256_file(temporary / "config.yaml"),
                "teacher_trajectories.jsonl": output_snapshot_sha,
            },
            "derived_from": {
                "sample_directory": str(source_sample_dir.resolve()),
                "manifest_sha256": sha256_file(source_manifest_path),
                "config_sha256": sha256_file(source_config),
                "teacher_trajectories_sha256": actual_snapshot_sha,
                "generation_requests": int(original_counts["generation_requests"]),
                "trajectories": int(original_counts["trajectories"]),
            },
            "code": _git_metadata(),
            "source_sha256": {
                "stage": sha256_file(Path(__file__)),
                "config": sha256_file(config),
            },
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_sample_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_sample_dir", type=Path)
    parser.add_argument("output_sample_dir", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--source-samples-per-prompt", type=int, default=8)
    parser.add_argument("--selected-samples-per-prompt", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=42)
    args = parser.parse_args()
    manifest = subsample_snapshot(
        args.source_sample_dir,
        args.output_sample_dir,
        args.config,
        source_samples_per_prompt=args.source_samples_per_prompt,
        selected_samples_per_prompt=args.selected_samples_per_prompt,
        selection_seed=args.selection_seed,
    )
    print(json.dumps(manifest["record_counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
