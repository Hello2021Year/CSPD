"""Score generations from generate.py with the official graders used in the paper.

  python experiments/m2rl/eval/score.py --bench ifbench --responses runs/eval/MCSI/ifbench.jsonl

* AIME24 and GPQA-Diamond: the M2RL reward functions (``async_rm`` with ``deepscaler`` /
  ``gpqa``), fetched from the pinned M2RL commit and checked against their git blob hashes.
* IFBench: the official IFBench ``evaluation_lib`` (strict prompt accuracy; loose is
  reported alongside), applied to the text after ``</think>``.
* LiveCodeBench v5: the official ``lcb_runner`` (6 s per test), applied to the code
  extracted from the text after ``</think>``. Hidden tests are read from
  ``livecodebench/code_generation_lite`` at a pinned revision.

The score is avg@K: per-question mean over samples, then mean over questions.
Generated code is executed; run LCB scoring inside a sandbox or container.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import hashlib
import json
import random
import subprocess
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

M2RL_COMMIT = "c94ce97ce7f2e0f2800e6dd8c20667f00ca223d4"
M2RL_RM_HUB = f"https://raw.githubusercontent.com/Mosi-AI/M2RL/{M2RL_COMMIT}/M2RL/slime/slime/rollout/rm_hub"
M2RL_SOURCES = {  # file -> git blob sha1
    "math_utils.py": "94ec98b9205d21a22df6c6315d7d44640aabdc2e",
    "deepscaler.py": "3c722ed9ec0fa46a2dedddb5f3b9e5133eede743",
    "gpqa.py": "ca99830f99e6cfe3b67ed6d9c70587b669d3cd3a",
    "__init__.py": "4525a72871c2168d5c6e5731ccd8a7915c95e405",
}
IFBENCH_REPO, IFBENCH_COMMIT = "https://github.com/allenai/IFBench.git", "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d"
LCB_REPO, LCB_COMMIT = "https://github.com/LiveCodeBench/LiveCodeBench.git", "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
LCB_DATASET, LCB_DATASET_REVISION = "livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
CACHE = Path.home() / ".cache" / "cspd-graders"


def final_answer(text):
    """Text after a single well-formed ``</think>``; empty if the thinking is malformed."""
    openings, closings = text.count("<think>"), text.count("</think>")
    if closings == 1 and openings <= 1 and ("<think>" not in text or text.index("<think>") < text.index("</think>")):
        return text.split("</think>", 1)[1].strip()
    if openings or closings:
        return ""
    return text.strip()


def git_checkout(repo, commit):
    """Clone ``repo`` at ``commit`` once into the cache (call before starting worker processes)."""
    import shutil
    import tempfile
    target = CACHE / f"{Path(repo).stem}-{commit[:12]}"
    if not (target / ".checkout-complete").exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=target.name + ".", dir=CACHE))
        subprocess.run(["git", "clone", "-q", repo, str(staging)], check=True)
        subprocess.run(["git", "-C", str(staging), "checkout", "-q", commit], check=True)
        (staging / ".checkout-complete").touch()
        shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)
    return target


# ---------------------------------------------------------------- AIME / GPQA

def m2rl_sources():
    root = CACHE / f"m2rl-rm-hub-{M2RL_COMMIT[:12]}" / "sources"
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").touch()
    for name, blob in M2RL_SOURCES.items():
        target = root / ("rm_hub_init.py" if name == "__init__.py" else name)
        if not target.exists():
            with urllib.request.urlopen(f"{M2RL_RM_HUB}/{name}", timeout=60) as response:
                target.write_bytes(response.read())
        content = target.read_bytes()
        if hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() != blob:
            raise ValueError(f"{name} does not match the pinned M2RL source")
    return root


_RM = {}


def _init_rm(rm_type):
    root = m2rl_sources()
    sys.path.insert(0, str(root.parent))
    from sources.deepscaler import get_deepscaler_rule_based_reward
    from sources.gpqa import compute_gpqa_reward
    tree = ast.parse((root / "rm_hub_init.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "async_rm")
    namespace = {"asyncio": asyncio, "Sample": SimpleNamespace, "compute_gpqa_reward": compute_gpqa_reward,
                 "get_deepscaler_rule_based_reward": get_deepscaler_rule_based_reward}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(root / "rm_hub_init.py"), "exec"), namespace)
    _RM.update(fn=namespace["async_rm"], rm_type=rm_type)


def _score_rm(row):
    metadata = {"valid_letters": "ABCD", "correct_letter": row["gold"]} if _RM["rm_type"] == "gpqa" else {}
    sample = SimpleNamespace(response=row["text"], label=row["gold"], metadata=metadata)
    args = SimpleNamespace(custom_rm_path=None, rm_type=_RM["rm_type"])
    return float(asyncio.run(_RM["fn"](args, sample)))


def score_rm(rows, references, rm_type, workers):
    m2rl_sources()
    jobs = [dict(text=r["text"], gold=str(references[r["question_id"]]["gold"])) for r in rows]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_rm, initargs=(rm_type,)) as pool:
        return list(pool.map(_score_rm, jobs, chunksize=16))


# -------------------------------------------------------------------- IFBench

_IF = {}


def _init_if():
    root = git_checkout(IFBENCH_REPO, IFBENCH_COMMIT)
    sys.path.insert(0, str(root))
    import importlib.util
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    spec = importlib.util.spec_from_file_location("ifbench_evaluation_lib", root / "evaluation_lib.py")
    library = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = library
    spec.loader.exec_module(library)
    _IF["lib"] = library


def _score_if(job):
    reference, answer = job
    random.seed(42)
    lib, rules = _IF["lib"], reference["rules"]
    kwargs = [{k: v for k, v in values.items() if v is not None} for values in copy.deepcopy(rules["kwargs"])]
    example = lib.InputExample(key=rules.get("key", 0), prompt=reference["prompt"],
                               instruction_id_list=rules["instruction_id_list"], kwargs=kwargs)
    response = {reference["prompt"]: answer}
    strict = lib.test_instruction_following_strict(copy.deepcopy(example), response)
    loose = lib.test_instruction_following_loose(copy.deepcopy(example), response)
    return float(strict.follow_all_instructions), float(loose.follow_all_instructions)


def score_ifbench(rows, references, workers):
    git_checkout(IFBENCH_REPO, IFBENCH_COMMIT)
    import nltk
    for resource in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(resource, quiet=True)
    jobs = [(references[r["question_id"]], final_answer(r["text"])) for r in rows]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_if) as pool:
        return list(pool.map(_score_if, jobs, chunksize=8))


# ------------------------------------------------------------- LiveCodeBench

def _lcb_row(reference):
    from huggingface_hub import hf_hub_download
    locator = reference["source_locator"]
    path = hf_hub_download(LCB_DATASET, locator["file"], repo_type="dataset", revision=LCB_DATASET_REVISION)
    with open(path, "rb") as stream:
        stream.seek(locator["byte_offset"])
        raw = stream.read(locator["byte_length"])
    if hashlib.sha256(raw).hexdigest() != locator["raw_line_sha256"]:
        raise ValueError(f"LiveCodeBench row changed for {reference['question_id']}")
    return json.loads(raw)


def _score_lcb_question(job):
    import contextlib
    import os
    reference, answers = job
    sys.path.insert(0, str(git_checkout(LCB_REPO, LCB_COMMIT)))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations_by_problem
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code
    problem = CodeGenerationProblem(**_lcb_row(reference))
    code = [extract_code(answer, LMStyle.OpenAIChat) for answer in answers]
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        outcomes, _ = evaluate_generations_by_problem((code, problem.get_evaluation_sample(), False, 6))
    return [float(bool(values) and all(int(v) > 0 for v in values)) for values in outcomes]


def score_lcb(rows, references, workers):
    git_checkout(LCB_REPO, LCB_COMMIT)
    by_question = defaultdict(list)
    for index, row in enumerate(rows):
        by_question[row["question_id"]].append(index)
    jobs = [(references[q], [final_answer(rows[i]["text"]) for i in indices]) for q, indices in by_question.items()]
    scores = [0.0] * len(rows)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for indices, values in zip(by_question.values(), pool.map(_score_lcb_question, jobs)):
            for i, value in zip(indices, values):
                scores[i] = value
    return scores


# ----------------------------------------------------------------------- main

def avg_at_k(rows, scores):
    per_question = defaultdict(list)
    for row, value in zip(rows, scores):
        per_question[row["question_id"]].append(value)
    return 100 * sum(sum(v) / len(v) for v in per_question.values()) / len(per_question), len(per_question)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True, choices=["aime24", "gpqa_diamond", "ifbench", "lcb_v5"])
    p.add_argument("--responses", type=Path, required=True)
    p.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parent / "inputs")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output", type=Path, help="per-sample scores (default: <responses>.scores.jsonl)")
    args = p.parse_args()

    rows = [json.loads(line) for line in args.responses.open()]
    references = {json.loads(l)["question_id"]: json.loads(l) for l in (args.inputs / args.bench / "references.jsonl").open()}
    loose = None
    if args.bench == "aime24":
        scores = score_rm(rows, references, "deepscaler", args.workers)
    elif args.bench == "gpqa_diamond":
        scores = score_rm(rows, references, "gpqa", args.workers)
    elif args.bench == "ifbench":
        pairs = score_ifbench(rows, references, args.workers)
        scores, loose = [s for s, _ in pairs], [l for _, l in pairs]
    else:
        scores = score_lcb(rows, references, args.workers)
    output = args.output or args.responses.with_suffix(".scores.jsonl")
    with output.open("w") as stream:
        for row, value in zip(rows, scores):
            stream.write(json.dumps({"question_id": row["question_id"], "sample_index": row["sample_index"], "score": value}) + "\n")
    score, questions = avg_at_k(rows, scores)
    report = {"bench": args.bench, "questions": questions, "samples": len(rows), "score": round(score, 2)}
    if loose is not None:
        report["loose_score"] = round(avg_at_k(rows, loose)[0], 2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
