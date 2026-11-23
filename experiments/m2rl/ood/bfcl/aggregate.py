"""Aggregate per-case BFCL v3 results into the Table 3 score.

BFCL is run with the official harness (gorilla, commit cd9429ccf3d4d04156affe883c495b3b047e6b64,
`berkeley-function-call-leaderboard`, handler `QwenFCHandler` for `Qwen/Qwen3-4B-FC`, at most 32,768
new tokens per turn and 20 steps per turn), on the 4,041 cases in cases.json: all BFCL v3
categories except multi_turn_miss_func and multi_turn_miss_param. Each case scores 0 or 1.

The aggregate uses the official BFCL v3 category weights; the two remaining multi-turn categories
are rescaled so that multi-turn contributes one third.

  python experiments/m2rl/ood/bfcl/aggregate.py runs/bfcl/MCSI.jsonl   # output of run.py
"""
import json
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

WEIGHTS = {  # official BFCL v3 overall weights
    "simple": "1/45", "java": "1/45", "javascript": "1/45", "irrelevance": "1/15", "parallel": "1/15",
    "multiple": "1/15", "parallel_multiple": "1/15", "live_simple": "86/2251", "live_multiple": "351/2251",
    "live_parallel": "16/6753", "live_parallel_multiple": "8/2251", "live_irrelevance": "294/2251",
    "live_relevance": "6/2251", "multi_turn_base": "1/12", "multi_turn_long_context": "1/12",
}


def weights():
    w = {k: Fraction(v) for k, v in WEIGHTS.items()}
    multi = [k for k in w if k.startswith("multi_turn_")]
    factor = Fraction(1, 3) / sum(w[k] for k in multi)
    for k in multi:
        w[k] *= factor
    assert sum(w.values()) == 1
    return w


def main():
    cases = {c["question_id"]: c["category"] for c in json.load(open(Path(__file__).with_name("cases.json")))}
    scores = {json.loads(l)["question_id"]: float(json.loads(l)["score"]) for l in open(sys.argv[1])}
    groups = defaultdict(list)
    for qid, category in cases.items():
        groups[category].append(scores.get(qid, 0.0))  # a missing case counts as 0
    means = {c: sum(v) / len(v) for c, v in groups.items()}
    total = sum(float(w) * means[c] for c, w in weights().items())
    print(json.dumps({"cases": len(cases), "missing": len(set(cases) - set(scores)), "score": round(100 * total, 2),
                      "categories": {c: round(100 * m, 2) for c, m in sorted(means.items())}}))


if __name__ == "__main__":
    main()
