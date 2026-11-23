"""CPU checks of the M2RL teacher-data helpers (no model, no network)."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

TEACHER = Path(__file__).resolve().parents[1] / "experiments" / "m2rl" / "teacher"
sys.path.insert(0, str(TEACHER))

import build_questions as bq  # noqa: E402
import sample_teacher as st  # noqa: E402


def test_question_id_hashes_the_raw_line():
    line = b'{"dataset": "x"}\n'
    assert bq.question_id(3, line) == "blend:0000003:" + hashlib.sha256(line).hexdigest()[:16]


def test_pools_are_sorted_unique_blend_ids():
    sizes = {"math": 18205, "code": 10661, "science": 19636, "if": 16569}
    for domain, size in sizes.items():
        with gzip.open(TEACHER / "pools" / f"{domain}.txt.gz", "rt") as stream:
            ids = stream.read().split()
        assert len(ids) == len(set(ids)) == size and ids == sorted(ids)
        assert all(i.startswith("blend:") and len(i) == 30 for i in ids)


def test_math_placeholder_restore():
    question = f"{bq.DAPO_PREFIX}\n\nWhat is 1+1?\n\n{bq.DAPO_SUFFIX}"
    rows = lambda category, index: {"prompt": [{"content": question}], "reward_model": {"ground_truth": '["2"]'}}
    record = {"dataset": "nano_v3_sft_profiled_dapo17k",
              "_hf_placeholder": {"row": 0, "question_template": {"prefix": "Q: ", "suffix": None}}}
    messages, label, metadata = bq.prepare("math", record, rows)
    assert messages == [{"role": "user", "content": bq.MATH_PREFIX + "Q: \n\nWhat is 1+1?"}]
    assert label == "2" and metadata == {"rm_type": "deepscaler"}


def test_science_prefix_and_letters():
    text = "Answer the following multiple choice question. Which?\n\nA) one\nB) two"
    record = {"responses_create_params": {"input": [{"role": "user", "content": text}]},
              "options": [{"A": "one"}, {"B": "two"}], "expected_answer": "B"}
    messages, label, metadata = bq.prepare("science", record, None)
    assert messages[0]["content"].startswith(bq.SCIENCE_NEW + " Which?")
    assert label == "B" and metadata == {"rm_type": "gpqa", "valid_letters": ["A", "B"]}
    record["options"] = [{"A": "one"}, {"C": "two"}]
    try:
        bq.prepare("science", record, None)
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned options must be rejected")


def test_verdicts():
    good = dict(reward=1.0, finish_reason="stop", output_ids=[1, 2, 151645], text="<think>x</think>answer")
    assert st.verdict(good, 10, set()) is None
    assert st.verdict(dict(good, reward=0.0), 10, set()) == "incorrect"
    assert st.verdict(dict(good, finish_reason="length"), 10, set()) == "truncated_or_aborted"
    assert st.verdict(dict(good, text="<think>a</think>b</think>"), 10, set()) == "malformed_thinking"
    assert st.verdict(good, 65534 - 256 - 2, set()) == "does_not_fit_with_256_SP_tokens"
    assert st.verdict(good, 10, {hashlib.sha256(good["text"].encode()).hexdigest()}) == "duplicate_response"


def test_code_test_seed_is_stable():
    import rewards
    seed = rewards.code_test_seed("blend:0000004:fa3299bf7491aa09")
    assert seed == 1683001624  # the seed recorded with the original code grades
    assert json.dumps(seed)
