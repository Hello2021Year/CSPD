"""Rebuild the GPQA-Diamond evaluation prompts from the gated original dataset.

GPQA asks that its questions not be redistributed in plain text, so the release only stores
``inputs/gpqa_diamond/layout.jsonl``: for each question, SHA-256 hashes of the question
text, of each answer option in A-D order, and of the final prompt. Accept the dataset terms
at https://huggingface.co/datasets/Idavidrein/gpqa, log in (``hf auth login``) and run

  python experiments/m2rl/eval/build_gpqa.py

This writes ``inputs/gpqa_diamond/questions.jsonl`` with exactly the prompts used in the
paper (the M2RL prompt template, options in the evaluated order).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

INSTRUCTION = ("Answer the following multiple choice question. The last line of your response should be of the "
               "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
               "Think step by step before answering.")
TEMPLATE = "{Instruction}\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
ANSWERS = ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parent / "inputs")
    args = p.parse_args()
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Idavidrein/gpqa", "gpqa_diamond.csv", repo_type="dataset",
                           revision="83022cefff930aea54f654c0b282e74b9eeda5c6")
    with open(path, newline="", encoding="utf-8") as stream:
        by_question = {sha(row["Question"].strip()): row for row in csv.DictReader(stream)}
    directory = args.inputs / "gpqa_diamond"
    layout = [json.loads(line) for line in (directory / "layout.jsonl").open()]
    with (directory / "questions.jsonl").open("w") as out:
        for item in layout:
            row = by_question[item["question_sha256"]]
            options = {sha(row[name].strip()): row[name].strip() for name in ANSWERS}
            ordered = [options[h] for h in item["option_sha256"]]
            prompt = TEMPLATE.format(Instruction=INSTRUCTION, Question=row["Question"].strip(),
                                     A=ordered[0], B=ordered[1], C=ordered[2], D=ordered[3])
            if sha(prompt) != item["prompt_sha256"]:
                raise ValueError(f"rebuilt prompt differs for {item['question_id']}")
            out.write(json.dumps({"question_id": item["question_id"],
                                  "messages": [{"role": "user", "content": prompt}]}, ensure_ascii=False) + "\n")
    print(f"wrote {len(layout)} GPQA-Diamond prompts to {directory / 'questions.jsonl'}")


if __name__ == "__main__":
    main()
