#!/usr/bin/env python3
"""Sample teacher, student and student+soft-prompt traces for token-level analysis.

Step 1 of the token-level analysis (Appendix G).

Subcommands:
  prepare   write the fixed held-out problem list (AIME 2024/2025 + stratified MATH-500)
  build-sp  write a student directory whose spare embedding rows hold the soft prompt
  generate  sample traces from one source with vLLM

Prompts are rendered with the repository's own template code, so the token IDs
match the original evaluation contract.  The soft-prompt model reuses the
student's padding rows for 256 virtual tokens (as ``virtual_tokens.py`` does)
and generation applies the same -100 logit bias to those IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from soft_prompt_distillation.common import RunOptions, load_tokenizer, read_dataset_rows
from soft_prompt_distillation.config import load_config
from soft_prompt_distillation.rendering import render_for_execution
from soft_prompt_distillation.tasks import get_task
from soft_prompt_distillation.virtual_tokens import LOGIT_BIAS, insert_virtual_token_ids

from prompt_variants import VARIANTS, apply_variant

DATASETS = ("aime_2024", "aime_2025", "math_500")
MATH500_PER_LEVEL = 8
SELECTION_SEED = 0


def template_tokenizer(config_path: Path, teacher_dir: Path, student_dir: Path):
    config = load_config(config_path)
    options = RunOptions(
        config=config_path, output=Path("."), cache_dir=Path("."),
        student_model_path=student_dir, teacher_model_path=teacher_dir,
    )
    tokenizer, _ = load_tokenizer(options, config, config.template.source)
    return config, tokenizer


def cmd_prepare(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    task = get_task(load_config(args.config).task)
    problems = []
    for name in DATASETS:
        source = task.evaluation_datasets[name]
        revision = HfApi().dataset_info(source.repo).sha
        path = hf_hub_download(source.repo, source.filename, repo_type="dataset",
                               revision=revision)
        raw = read_dataset_rows(Path(path))
        rows = [(r, task.normalize_evaluation_row(r, name)) for r in raw]
        if name == "math_500":
            rng = random.Random(SELECTION_SEED)
            picked = []
            for level in range(1, 6):
                pool = [x for x in rows if int(x[0]["level"]) == level]
                picked += rng.sample(pool, MATH500_PER_LEVEL)
            rows = picked
        for r, norm in rows:
            problems.append({
                **norm,
                "level": r.get("level"),
                "subject": r.get("subject"),
                "source_repo": source.repo,
                "source_revision": revision,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in problems:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(problems)} problems to {args.output}")


def cmd_build_sp(args: argparse.Namespace) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    student = args.student_dir
    index = json.loads((student / "model.safetensors.index.json").read_text())
    embed_shard = index["weight_map"]["model.embed_tokens.weight"]
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for item in student.iterdir():
        if item.name != embed_shard:
            (out / item.name).symlink_to(item.resolve())
    with safe_open(str(args.checkpoint), framework="pt") as handle:
        prompt = handle.get_tensor("prompt_embeddings")
    with safe_open(str(student / embed_shard), framework="pt") as handle:
        metadata = handle.metadata()
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    from transformers import AutoTokenizer

    n_tok = len(AutoTokenizer.from_pretrained(str(student)))
    embed = tensors["model.embed_tokens.weight"].clone()
    if n_tok + prompt.size(0) > embed.size(0):
        raise SystemExit("not enough spare embedding rows for the soft prompt")
    ids = list(range(n_tok, n_tok + prompt.size(0)))
    embed[ids] = prompt.to(embed.dtype)
    tensors["model.embed_tokens.weight"] = embed
    save_file(tensors, str(out / embed_shard), metadata=metadata)
    (out / "soft_prompt_virtual_ids.json").write_text(json.dumps({
        "virtual_ids": ids,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "logit_bias": LOGIT_BIAS,
    }, indent=1))
    print(f"wrote {out} (virtual ids {ids[0]}..{ids[-1]})")


def request_seed(source: str, question_id: str, k: int) -> int:
    digest = hashlib.sha256(f"{source}|{question_id}|{k}".encode()).hexdigest()
    return int(digest[:8], 16)


def cmd_generate(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    config, tokenizer = template_tokenizer(args.config, args.teacher_dir, args.student_dir)
    task = get_task(config.task)
    problems = [json.loads(line) for line in args.problems.open()]
    if args.limit:
        problems = problems[: args.limit]
    virtual_ids: list[int] = []
    if args.source == "student_sp":
        meta = json.loads((args.model_dir / "soft_prompt_virtual_ids.json").read_text())
        virtual_ids = meta["virtual_ids"]
    execution_model = "teacher" if args.source == "teacher" else "student"
    if args.variant and args.source != "student":
        raise SystemExit("--variant applies to the plain student source only")
    label = f"student_{args.variant}" if args.variant else args.source
    gen = config.evaluation.arms["student"].generation

    requests, records = [], []
    for problem in problems:
        rendered = render_for_execution(config, execution_model=execution_model,
                                        row=problem, tokenizer=tokenizer)
        base_ids = list(rendered.token_ids)
        insert_at = int(rendered.soft_prompt_insert_at)
        ids = insert_virtual_token_ids(base_ids, insert_at, virtual_ids) if virtual_ids else base_ids
        if args.variant:
            ids = apply_variant(base_ids, insert_at, args.variant, tokenizer)
        for k in range(args.samples):
            seed = request_seed(label, problem["question_id"], k)
            params = SamplingParams(
                temperature=gen.temperature, top_p=gen.top_p, top_k=gen.top_k,
                max_tokens=args.max_new_tokens, seed=seed,
                logit_bias={i: LOGIT_BIAS for i in virtual_ids} if virtual_ids else None,
            )
            requests.append(({"prompt_token_ids": ids}, params))
            records.append({
                "source": label, "dataset": problem["dataset"],
                "question_id": problem["question_id"], "sample": k, "seed": seed,
                "answer": problem["answer"], "level": problem.get("level"),
                "prompt_token_ids": base_ids, "soft_prompt_insert_at": insert_at,
            })

    llm = LLM(model=str(args.model_dir), tokenizer=str(args.model_dir),
              max_model_len=args.max_model_len, gpu_memory_utilization=0.9, seed=0)
    outputs = llm.generate([r[0] for r in requests], [r[1] for r in requests])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record, output in zip(records, outputs):
            completion = output.outputs[0]
            text = tokenizer.decode(completion.token_ids, skip_special_tokens=False)
            record.update({
                "response_token_ids": list(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "truncated": completion.finish_reason == "length",
                "text": text,
                "correct": bool(task.grade(text, record["answer"])),
            })
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    n = len(records)
    acc = sum(r["correct"] for r in records) / max(1, n)
    print(f"wrote {n} traces to {args.output}; accuracy {acc:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("build-sp")
    p.add_argument("--student-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("generate")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--source", choices=["teacher", "student", "student_sp"], required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--teacher-dir", type=Path, required=True)
    p.add_argument("--student-dir", type=Path, required=True)
    p.add_argument("--problems", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=32768)
    p.add_argument("--max-model-len", type=int, default=34816)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--variant", choices=VARIANTS, default=None,
                   help="natural-language prompt variant for the student source")
    args = parser.parse_args()
    {"prepare": cmd_prepare, "build-sp": cmd_build_sp, "generate": cmd_generate}[args.cmd](args)


if __name__ == "__main__":
    main()
