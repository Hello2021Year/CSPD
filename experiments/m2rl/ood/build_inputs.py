"""Build the out-of-domain evaluation inputs of Table 3 from the public datasets.

  python experiments/m2rl/ood/build_inputs.py --output experiments/m2rl/ood/inputs

Writes ``<bench>/questions.jsonl`` and ``<bench>/references.jsonl`` for TriviaQA, TruthfulQA
(MC1), XSTest, BBEH and FinQA, in the format read by eval/generate.py. The selection and
prompts follow the paper: a fixed ``numpy.random.default_rng(20261109)`` order, TriviaQA closed
book (2,000 questions), TruthfulQA MC1 with shuffled options (817), all 450 XSTest prompts,
the official BBEH mini subset, and all 1,147 FinQA test questions. BFCL is run separately
(see ood/README in the main README).
"""
from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path

import numpy as np

SEED = 20261109
SOURCES = {
    "triviaqa": ("mandarjoshi/trivia_qa", "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f", "rc.nocontext/validation-00000-of-00001.parquet"),
    "truthfulqa": ("truthfulqa/truthful_qa", "741b8276f2d1982aa3d5b832d3ee81ed3b896490", "multiple_choice/validation-00000-of-00001.parquet"),
    "xstest": ("Paul/XSTest", "f600c994b256f12867dfa5b3eb3d545a3e62f8b5", "xstest_prompts.csv"),
    "bbeh": ("BBEH/bbeh", "08e07a803851822c04399782ece3c4a07ce419f9", "bbeh_data.json"),
}
FINQA_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/0f16e2867befa6840783e58be38c9efb9229d742/dataset/test.json"

TQA_INSTR = ("Answer the following trivia question concisely. Put only your final answer (a short name, word, "
             "phrase or number) inside \\boxed{}.\n\nQuestion: ")
TFQA_LETTERS = "ABCDEFGHIJKLM"
TFQA_HEAD = ("The following is a multiple choice question. Think step by step and then finish your answer with "
             '"The answer is (X)" where X is the correct letter choice.\n\n')
BOX_TAIL = "\n\nThink step by step, then put your final answer inside \\boxed{}."


def fetch(bench):
    from huggingface_hub import hf_hub_download
    repo, revision, filename = SOURCES[bench]
    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


def ordered_by_stratum(rows):
    """rows: (stratum, qid, content, reference) in a fixed order; shuffle within each stratum."""
    rng = np.random.default_rng(SEED)
    by = {}
    for row in rows:
        by.setdefault(row[0], []).append(row)
    out = []
    for stratum in sorted(by):
        items = by[stratum]
        out += [items[int(i)] for i in rng.permutation(len(items))]
    return [(qid, content, dict(ref, stratum=stratum)) for stratum, qid, content, ref in out]


def triviaqa():
    import pandas as pd
    df = pd.read_parquet(fetch("triviaqa"))
    df = df.drop_duplicates("question_id", keep="first").sort_values("question_id").reset_index(drop=True)
    order = np.random.default_rng(SEED).permutation(len(df))[:2000]
    for i in order:
        r = df.iloc[int(i)]
        a = r.answer
        aliases = sorted({str(x) for x in list(a["aliases"]) + list(a["normalized_aliases"]) + [a["value"], a["normalized_value"]] if str(x).strip()})
        yield f"triviaqa:{r.question_id}", TQA_INSTR + r.question.strip(), dict(fmt="alias", gold=str(a["value"]), aliases=aliases)


def truthfulqa():
    import pandas as pd
    df = pd.read_parquet(fetch("truthfulqa"))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(df))
    perms = [rng.permutation(len(df.iloc[i].mc1_targets["choices"])) for i in range(len(df))]
    for i in order:
        r = df.iloc[int(i)]
        choices, labels = [str(c) for c in r.mc1_targets["choices"]], [int(x) for x in r.mc1_targets["labels"]]
        shuffled = [choices[j] for j in perms[i]]
        gold = TFQA_LETTERS[int(list(perms[i]).index(labels.index(1)))]
        text = TFQA_HEAD + "Question:\n" + r.question.strip() + "\nOptions:\n" + "".join(
            f"{TFQA_LETTERS[j]}. {o}\n" for j, o in enumerate(shuffled)).rstrip("\n")
        yield f"truthfulqa_mc1:{int(i):03d}", text, dict(fmt="mc", gold=gold, n_options=len(choices))


def xstest():
    with open(fetch("xstest"), newline="", encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: int(r["id"]))
    for r in rows:
        yield f"xstest:{int(r['id']):03d}", r["prompt"], dict(fmt="refusal", gold="comply" if r["label"] == "safe" else "refuse",
                                                             label=r["label"], type=r["type"])


def bbeh():
    data = [x for x in json.load(open(fetch("bbeh"))) if x.get("mini")]
    rows = [(x["task"].replace(" ", "_"), f"bbeh:{x['task'].replace(' ', '_')}:{i:04d}", x["input"].strip() + BOX_TAIL,
             dict(fmt="exact", gold=str(x["target"]).strip())) for i, x in enumerate(data)]
    yield from ordered_by_stratum(rows)


def finqa():
    with urllib.request.urlopen(FINQA_URL, timeout=120) as response:
        data = json.load(response)
    rows = []
    for x in data:
        table = "\n".join(" | ".join(str(c) for c in row) for row in x["table"])
        gold = x["qa"]["exe_ans"]
        if isinstance(gold, str) and gold.strip().lower() in ("yes", "no"):
            ref = dict(fmt="label", gold=gold.strip().lower(), labels=["yes", "no"])
        else:
            ref = dict(fmt="number", gold=float(gold))
        text = ("Answer the question about the financial report below. The answer is a single number "
                "(or yes/no). Give only the number in the box, without units; write percentages as a number with a % sign.\n\n"
                "Text before the table:\n" + " ".join(x["pre_text"]) + "\n\nTable:\n" + table + "\n\nText after the table:\n"
                + " ".join(x["post_text"]) + "\n\nQuestion: " + x["qa"]["question"] + BOX_TAIL)
        rows.append(("all", f"finqa:{x['id']}", text, ref))
    yield from ordered_by_stratum(rows)


BUILDERS = {"triviaqa": triviaqa, "truthfulqa": truthfulqa, "xstest": xstest, "bbeh": bbeh, "finqa": finqa}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "inputs")
    p.add_argument("--benches", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    args = p.parse_args()
    for bench in args.benches:
        directory = args.output / bench
        directory.mkdir(parents=True, exist_ok=True)
        n = 0
        with (directory / "questions.jsonl").open("w") as q, (directory / "references.jsonl").open("w") as r:
            for qid, content, ref in BUILDERS[bench]():
                q.write(json.dumps({"question_id": qid, "messages": [{"role": "user", "content": content}]}, ensure_ascii=False) + "\n")
                r.write(json.dumps({"question_id": qid, **ref}, ensure_ascii=False) + "\n")
                n += 1
        print(f"{bench}: {n} questions -> {directory}")


if __name__ == "__main__":
    main()
