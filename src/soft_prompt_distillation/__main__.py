from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def _request_batch_size(value: str) -> int | str:
    if value == "all":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer or all") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer or all")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Soft-prompt distillation tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate an experiment config")
    validate.add_argument("config", type=Path, metavar="CONFIG")
    for command, help_text in (
        ("sample", "sample frozen teacher trajectories"),
        ("train", "train a soft prompt from a frozen snapshot"),
        ("export", "export a soft prompt as virtual token embeddings"),
        ("evaluate", "evaluate a student, teacher, or learned soft prompt"),
        ("run-smoke", "run sample, train, and final soft-prompt evaluation"),
    ):
        stage = subparsers.add_parser(command, help=help_text)
        stage.add_argument("config", type=Path, metavar="CONFIG")
        stage.add_argument("--output", type=Path, required=True)
        stage.add_argument("--cache-dir", type=Path, required=True)
        stage.add_argument("--student-model-path", type=Path)
        stage.add_argument("--teacher-model-path", type=Path)
        if command in {"sample", "evaluate", "run-smoke"}:
            if command == "evaluate":
                stage.add_argument(
                    "--generation-backend",
                    choices=("sglang", "vllm"),
                    default="sglang",
                    help="inference backend for baseline evaluation (default: sglang)",
                )
            stage.add_argument(
                "--sglang-tp-size",
                type=int,
                default=1,
                help="GPUs per SGLang model replica (default: 1)",
            )
            stage.add_argument(
                "--sglang-dp-size",
                type=int,
                default=1,
                help="SGLang model replicas; total GPUs = TP x DP (default: 1)",
            )
            stage.add_argument(
                "--sglang-batch-size",
                type=_request_batch_size,
                default=8192,
                help="requests per SGLang submission: positive integer or all (default: 8192)",
            )
            stage.add_argument(
                "--sglang-log-level",
                choices=("debug", "info", "warning", "error", "critical"),
                default="error",
                help="SGLang engine log level (default: error)",
            )
            stage.add_argument(
                "--code-verifier-command",
                type=Path,
                help=(
                    "executable runner for isolated code grading; receives "
                    "REQUESTS_JSONL RESPONSES_JSONL"
                ),
            )
            stage.add_argument(
                "--code-verifier-timeout-seconds",
                type=int,
                default=21600,
                help="outer timeout for one isolated verifier batch (default: 21600)",
            )
        if command == "sample":
            stage.add_argument(
                "--reuse-sample-dir",
                type=Path,
                help=(
                    "completed sample directory whose exact overlapping questions "
                    "are reused without regeneration"
                ),
            )
            stage.add_argument(
                "--sample-shard-index",
                type=int,
                default=0,
                help="zero-based sampling-part shard owned by this process",
            )
            stage.add_argument(
                "--sample-shard-count",
                type=int,
                default=1,
                help="number of mutually exclusive sampling-part shards",
            )
            stage.add_argument(
                "--sample-finalize-only",
                action="store_true",
                help="verify and merge completed shards without starting SGLang",
            )
        if command == "train":
            stage.add_argument(
                "--resume",
                action="store_true",
                help=(
                    "reuse an existing training directory and continue from "
                    "training_state.pt"
                ),
            )
            stage.add_argument("--snapshot", type=Path, required=True)
        if command in {"train", "run-smoke"}:
            stage.add_argument(
                "--train-model-placement",
                choices=("replicated", "balanced"),
                default="replicated",
                help=(
                    "replicated: one full student per torchrun rank; balanced: "
                    "single-process frozen-backbone sharding across visible GPUs"
                ),
            )
        if command == "export":
            stage.add_argument("--checkpoint", type=Path, required=True)
        if command == "evaluate":
            target_group = stage.add_mutually_exclusive_group(required=True)
            target_group.add_argument(
                "--target",
                choices=("student", "teacher", "soft-prompt"),
            )
            target_group.add_argument(
                "--arm",
                dest="legacy_arm",
                choices=("student", "teacher", "student_soft_prompt"),
                help=argparse.SUPPRESS,
            )
            stage.add_argument(
                "--soft-prompt-model-path",
                type=Path,
                help=argparse.SUPPRESS,
            )
            stage.add_argument(
                "--train-dir",
                type=Path,
                help="training directory; required for target soft-prompt",
            )
            stage.add_argument(
                "--checkpoint",
                help=(
                    "final (default), a nonnegative optimizer update, all, or "
                    "all-desc, or every:<positive integer>"
                ),
            )
            stage.add_argument(
                "--temporary-dir",
                type=Path,
                help="parent for temporary derived models, for example SLURM_TMPDIR",
            )
    args = parser.parse_args()
    if args.command == "validate":
        config = load_config(args.config)
        print(
            json.dumps(
                {
                    "name": config.name,
                    "task": config.task,
                    "student": config.student.checkpoint,
                    "teacher": config.teacher.checkpoint,
                    "sampled_trajectory_count": config.sampled_trajectory_count,
                    "objective": config.distillation.objective,
                    "template_source": config.template.source,
                    "template_render_mode": config.template.render_mode,
                    "generation_backend": "sglang",
                    "evaluation_arms": sorted(config.evaluation.arms),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    from .pipeline import (
        RunOptions,
        evaluate,
        evaluate_target,
        export_soft_prompt_model,
        run_smoke,
        sample,
        train,
    )

    options = RunOptions(
        config=args.config,
        output=args.output,
        cache_dir=args.cache_dir,
        student_model_path=args.student_model_path,
        teacher_model_path=args.teacher_model_path,
        soft_prompt_model_path=getattr(args, "soft_prompt_model_path", None),
        sglang_tp_size=getattr(args, "sglang_tp_size", 1),
        sglang_dp_size=getattr(args, "sglang_dp_size", 1),
        sglang_batch_size=getattr(args, "sglang_batch_size", 8192),
        sglang_log_level=getattr(args, "sglang_log_level", "error"),
        generation_backend=getattr(args, "generation_backend", "sglang"),
        train_model_placement=getattr(args, "train_model_placement", "replicated"),
        resume_training=bool(getattr(args, "resume", False)),
        reuse_sample_dir=getattr(args, "reuse_sample_dir", None),
        sample_shard_index=getattr(args, "sample_shard_index", 0),
        sample_shard_count=getattr(args, "sample_shard_count", 1),
        sample_finalize_only=bool(getattr(args, "sample_finalize_only", False)),
        code_verifier_command=getattr(args, "code_verifier_command", None),
        code_verifier_timeout_seconds=getattr(
            args, "code_verifier_timeout_seconds", 21600
        ),
    )
    if args.command == "sample":
        result = sample(options)
    elif args.command == "train":
        result = train(options, args.snapshot)
    elif args.command == "export":
        result = export_soft_prompt_model(options, args.checkpoint)
    elif args.command == "evaluate":
        if args.target is not None and args.legacy_arm is not None:
            parser.error("evaluate accepts only one of --target and legacy --arm")
        if args.target is not None:
            result = evaluate_target(
                options,
                args.target,
                train_dir=args.train_dir,
                checkpoint=args.checkpoint,
                temporary_dir=args.temporary_dir,
            )
        elif args.legacy_arm is not None:
            if args.train_dir is not None or args.checkpoint is not None:
                parser.error("legacy --arm cannot be combined with checkpoint options")
            result = evaluate(options, args.legacy_arm)
        else:
            parser.error("evaluate requires --target")
    elif args.command == "run-smoke":
        result = run_smoke(options)
    else:
        raise AssertionError(args.command)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
