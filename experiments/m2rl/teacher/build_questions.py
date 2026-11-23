"""Build the teacher-generation questions of one M2RL domain from the Nemotron RL blend.

  python experiments/m2rl/teacher/build_questions.py --domain math --output runs/teacher/math/questions.jsonl

The M2RL experts were trained with RL on nvidia/Nemotron-3-Nano-RL-Training-Blend. The paper's
distillation data use the same questions, from the blend revision below:

* ``question_id`` is ``blend:<line>:<sha256 of the raw line, first 16 hex>``, so every id is checked
  against the exact source record.
* Prompts follow M2RL's ``data/process_rl_data.py``. Math gets the "Answer:" instruction in front.
  In science, the leading "Answer the following multiple choice question." becomes "... step by step.";
  science prompts that start differently are left unchanged. IF and code prompts are used as they are.
* The blend stores math questions as placeholders into DAPO-Math-17k and Skywork-OR1-RL-Data.
  They are resolved like the blend's ``create_nanov3_jsonl.py``, from pinned parquet files.
* Grading data: math answer (``deepscaler``), science answer letter and the valid option letters
  (``gpqa``), IF instruction ids and kwargs (``ifevalg``), code unit tests (``unit_test``).

``pools/<domain>.txt.gz`` lists the questions used in the paper:

* Each id is the first record of its normalized question text.
* Questions whose text overlaps an evaluation benchmark are excluded (exact match, or a 9-word-shingle
  containment of at least 0.8 for texts of 30 or more words). The benchmarks are AIME24/25,
  GPQA-Diamond, IFEval, IFBench, LiveCodeBench, HLE and BFCL.
* Science questions whose option letters do not match the options shown in the prompt are dropped.
* Prompts longer than 2,048 tokens are dropped.

Pass ``--pool`` to use another id list.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import re
from pathlib import Path

BLEND = ("nvidia/Nemotron-3-Nano-RL-Training-Blend", "74d1844434885be8237369edaaf534f84fd53d87")
MATH_SOURCES = {  # blend category -> (repo, revision, parquet file)
    "nano_v3_sft_profiled_dapo17k": ("BytedTsinghua-SIA/DAPO-Math-17k", "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
                                     "data/dapo-math-17k.parquet"),
    "nano_v3_sft_profiled_skywork_no_omni": ("Skywork/Skywork-OR1-RL-Data", "1cdedc52e0e2db85fdf252f9be682e63a5a38c33",
                                             "data/math-00000-of-00001.parquet"),
}
CATEGORIES = {
    "math": {"nano_v3_sft_profiled_dapo17k", "nano_v3_sft_profiled_skywork_no_omni"},
    "science": {"nano_v3_sft_profiled_stem_mcqa"},
    "if": {"nano_v3_sft_profiled_instruction_following"},
    "code": {"nano_v3_sft_profiled_comp_coding_50tests"},
}
MATH_PREFIX = ("Solve the following math problem step by step. The last line of your response should be of the "
               "form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n")
SCIENCE_OLD = "Answer the following multiple choice question."
SCIENCE_NEW = "Answer the following multiple choice question step by step."
OPTION_LINE = re.compile(r"^\s*(?:\(([A-Z])\)|([A-Z])[.):])\s+", re.MULTILINE)
DAPO_PREFIX = ("Solve the following math problem step by step. The last line of your response should be of the "
               "form Answer: $Answer (without quotes) where $Answer is the answer to the problem.")
DAPO_SUFFIX = 'Remember to put your answer on its own line after "Answer:".'
POOLS = Path(__file__).resolve().parent / "pools"


def question_id(index, line):
    return f"blend:{index:07d}:{hashlib.sha256(line).hexdigest()[:16]}"


class MathRows:
    """Rows of the two math source datasets, by index, read from the pinned parquet files."""

    def __init__(self):
        self.tables = {}

    def __call__(self, category, index):
        if category not in self.tables:
            import pyarrow.parquet as pq
            from huggingface_hub import hf_hub_download
            repo, revision, filename = MATH_SOURCES[category]
            self.tables[category] = pq.read_table(hf_hub_download(repo, filename, revision=revision, repo_type="dataset"),
                                                  columns=["prompt", "reward_model"])
        return self.tables[category].slice(index, 1).to_pylist()[0]


def restore_math(record, rows):
    """Replace a math placeholder by its source question and answer (as create_nanov3_jsonl.py)."""
    placeholder, category = record["_hf_placeholder"], record["dataset"]
    row = rows(category, int(placeholder["row"]))
    question, template = row["prompt"][0]["content"], placeholder["question_template"]
    if category == "nano_v3_sft_profiled_dapo17k":
        start, end = question.index(DAPO_PREFIX) + len(DAPO_PREFIX), question.rfind(DAPO_SUFFIX)
        inner = question[start:end]
        if template.get("prefix"):
            question = (template["prefix"] + inner).removesuffix("\n\n")
        elif template.get("suffix"):
            question = inner + template["suffix"]
        else:
            raise ValueError(f"unknown DAPO template {template}")
    else:
        question = template["template"].replace("{question}", question)
    answer = row["reward_model"]["ground_truth"].strip()
    if answer.startswith(("[", "{")):
        answer = json.loads(answer)[0]
    return question, answer


def science_letters(record):
    """Valid option letters: the source options, which must match the options shown in the prompt."""
    letters = [next(iter(option)) for option in record["options"]]
    shown = [a or b for a, b in OPTION_LINE.findall(record["responses_create_params"]["input"][0]["content"])]
    if letters != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(letters)]) or shown != letters \
            or record["expected_answer"] not in letters:
        raise ValueError("option letters do not match the prompt")
    return letters


def prepare(domain, record, math_rows):
    """(messages, label, metadata) for one blend record."""
    if domain == "math":
        question, answer = restore_math(record, math_rows)
        return [{"role": "user", "content": MATH_PREFIX + question}], str(answer), {"rm_type": "deepscaler"}
    messages = copy.deepcopy(record["responses_create_params"]["input"])
    if domain == "science":
        text = messages[0]["content"]
        if text.startswith(SCIENCE_OLD):
            messages[0]["content"] = SCIENCE_NEW + text[len(SCIENCE_OLD):]
        return messages, record["expected_answer"], {"rm_type": "gpqa", "valid_letters": science_letters(record)}
    if domain == "if":
        return messages, None, {"rm_type": "ifevalg", "prompt_text": record["prompt"],
                                "instruction_id_list": record["instruction_id_list"], "kwargs": record["kwargs"]}
    return messages, None, {"rm_type": "unit_test", **record["verifier_metadata"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True, choices=sorted(CATEGORIES))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pool", type=Path, help="question ids, one per line (default: pools/<domain>.txt.gz)")
    p.add_argument("--blend", type=Path, help="local copy of the blend's train.jsonl (default: download it)")
    args = p.parse_args()

    pool_path = args.pool or POOLS / f"{args.domain}.txt.gz"
    opener = gzip.open if pool_path.suffix == ".gz" else open
    with opener(pool_path, "rt") as stream:
        pool = [line.strip() for line in stream if line.strip()]
    wanted = set(pool)
    if args.blend is None:
        from huggingface_hub import hf_hub_download
        args.blend = Path(hf_hub_download(BLEND[0], "train.jsonl", revision=BLEND[1], repo_type="dataset"))

    math_rows, found = MathRows(), {}
    with args.blend.open("rb") as stream:
        for index, line in enumerate(stream):
            qid = question_id(index, line)
            if qid not in wanted:
                continue
            record = json.loads(line)
            if record["dataset"] not in CATEGORIES[args.domain]:
                raise ValueError(f"{qid} is a {record['dataset']} record, not {args.domain}")
            messages, label, metadata = prepare(args.domain, record, math_rows)
            found[qid] = dict(question_id=qid, messages=messages, label=label, metadata=metadata)
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"{len(missing)} pool ids are not in the blend, e.g. {sorted(missing)[:3]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for qid in pool:
            out.write(json.dumps(found[qid], ensure_ascii=False) + "\n")
    print(json.dumps({"domain": args.domain, "questions": len(pool), "output": str(args.output)}))


if __name__ == "__main__":
    main()
