"""Sample, verify and select the teacher responses that train one domain prompt.

  python experiments/m2rl/teacher/sample_teacher.py --domain math \
      --questions runs/teacher/math/questions.jsonl --output runs/teacher/math
  python experiments/m2rl/pack_data.py runs/teacher/math data/m2rl/math

Protocol, matching the paper's data:

* **Teacher.** Each domain uses its M2RL RL expert at a pinned revision, run with SGLang in bf16 with
  a 65,536-token context. Prompts are rendered with the teacher's Qwen3 chat template, thinking on.
* **Sampling.** Temperature 1.0, top-p 1.0, no top-k, at most 32,768 new tokens. The engine seed is 42
  and requests have no seed of their own, so reruns do not reproduce the released responses token for token.
* **Acceptance.** A response is accepted when all of these hold:
  - the M2RL training reward is 1 (``rewards.py``);
  - generation stopped on an end token;
  - the thinking block is closed exactly once;
  - prompt + response + 256 soft-prompt rows fit in 65,534 tokens;
  - its text differs from the responses already accepted for that question.
* **Attempts.** Samples are drawn one at a time per question, and sampling stops once enough are accepted:
  - math, science and IF: at most 2 samples, 1 accepted response;
  - code: at most 4 samples, up to 2 distinct accepted responses.

Writes ``responses.jsonl`` (every sample with its reward and verdict; the run resumes from it) and
``candidates/shard-00.jsonl``, the input of ``pack_data.py``. ``--limit`` takes the first questions only,
for a smoke test. Code rewards execute generated programs: run code inside a container or sandbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rewards  # noqa: E402

TEACHERS = {
    "math": ("Jackwang111/M2RL-RL_Math", "56da007b8bd471bed79077d2c3f26f48a1e595dd"),
    "code": ("Jackwang111/M2RL-RL_Coding", "22df4d65195cd0b2b47f1e46c0602452272a00d4"),
    "science": ("Jackwang111/M2RL-RL_Science", "7f8747ba81e10c35343de90fd358be9867897cff"),
    "if": ("Jackwang111/M2RL-RL_IF", "9b14959ec7bd0d4c4c97f2bee06eb453ca658f94"),
}
ATTEMPTS = {"math": (2, 1), "science": (2, 1), "if": (2, 1), "code": (4, 2)}  # (max samples, max accepted)
STOP_IDS = (151645, 151643)
CONTEXT, USABLE_CONTEXT, SP_ROWS, MAX_PROMPT, MAX_NEW = 65536, 65534, 256, 2048, 32768


def malformed_thinking(text):
    openings, closings = text.count("<think>"), text.count("</think>")
    well_formed = closings == 1 and openings <= 1 and ("<think>" not in text or text.index("<think>") < text.index("</think>"))
    return not well_formed and bool(openings or closings)


def verdict(sample, prompt_len, accepted_texts):
    if sample["reward"] != 1:
        return "incorrect"
    if sample["finish_reason"] != "stop" or not sample["output_ids"] or sample["output_ids"][-1] not in STOP_IDS:
        return "truncated_or_aborted"
    if malformed_thinking(sample["text"]):
        return "malformed_thinking"
    if prompt_len + len(sample["output_ids"]) + SP_ROWS > USABLE_CONTEXT:
        return "does_not_fit_with_256_SP_tokens"
    if hashlib.sha256(sample["text"].encode()).hexdigest() in accepted_texts:
        return "duplicate_response"
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True, choices=sorted(TEACHERS))
    p.add_argument("--questions", type=Path, required=True, help="output of build_questions.py")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    p.add_argument("--chunk", type=int, default=4096, help="samples generated between checkpoints")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--grading-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    repo, revision = TEACHERS[args.domain]
    max_samples, max_accepted = ATTEMPTS[args.domain]
    model = snapshot_download(repo, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(model)
    questions = [json.loads(line) for line in args.questions.open()][:args.limit]
    prompts = {}
    for q in questions:
        ids = tokenizer.apply_chat_template(q["messages"], tokenize=True, add_generation_prompt=True,
                                            enable_thinking=True, return_dict=False)
        if len(ids) <= MAX_PROMPT:
            prompts[q["question_id"]] = list(ids)
    questions = [q for q in questions if q["question_id"] in prompts]

    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / "responses.jsonl"
    samples = {q["question_id"]: [] for q in questions}
    if log.exists():
        for line in log.open():
            row = json.loads(line)
            samples[row["question_id"]].append(row)

    def accepted(qid):
        return [s for s in samples[qid] if s["verdict"] is None]

    def pending():
        return [q for q in questions
                if len(accepted(q["question_id"])) < max_accepted and len(samples[q["question_id"]]) < max_samples]

    engine = None
    rewards.fetch_sources()
    with ProcessPoolExecutor(max_workers=args.grading_workers, initializer=rewards.init) as pool, log.open("a") as out:
        while todo := pending()[:args.chunk]:
            if engine is None:
                import sglang
                engine = sglang.Engine(model_path=model, dtype="bfloat16", context_length=CONTEXT,
                                       tp_size=args.tensor_parallel_size, random_seed=42, log_level="warning")
            params = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": MAX_NEW,
                      "stop_token_ids": list(STOP_IDS), "no_stop_trim": True, "skip_special_tokens": False,
                      "spaces_between_special_tokens": False}
            results = engine.generate(input_ids=[prompts[q["question_id"]] for q in todo],
                                      sampling_params=[dict(params) for _ in todo])
            rows = []
            for q, result in zip(todo, results):
                ids = list(result["output_ids"])
                trimmed = ids[:]
                while trimmed and trimmed[-1] in STOP_IDS:
                    trimmed.pop()
                text = tokenizer.decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                rows.append(dict(question_id=q["question_id"], sample_index=len(samples[q["question_id"]]),
                                 output_ids=ids, finish_reason=result["meta_info"]["finish_reason"]["type"], text=text))
            jobs = [dict(question_id=q["question_id"], text=r["text"], label=q["label"], metadata=q["metadata"])
                    for q, r in zip(todo, rows)]
            for q, row, reward in zip(todo, rows, pool.map(rewards.score, jobs, chunksize=4)):
                qid = q["question_id"]
                row["reward"] = reward
                texts = {hashlib.sha256(s["text"].encode()).hexdigest() for s in accepted(qid)}
                row["verdict"] = verdict(row, len(prompts[qid]), texts)
                samples[qid].append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            done = sum(len(accepted(q["question_id"])) > 0 for q in questions)
            print(json.dumps({"generated": len(rows), "questions_with_accepted": done, "pending": len(pending())}), flush=True)
    if engine is not None:
        engine.shutdown()

    (args.output / "candidates").mkdir(exist_ok=True)
    counts = {"questions": len(questions), "samples": 0, "accepted": 0, "questions_accepted": 0, "verdicts": {}}
    with (args.output / "candidates" / "shard-00.jsonl").open("w") as out:
        for q in questions:
            qid, prompt = q["question_id"], prompts[q["question_id"]]
            counts["samples"] += len(samples[qid])
            for s in samples[qid]:
                key = s["verdict"] or "accepted"
                counts["verdicts"][key] = counts["verdicts"].get(key, 0) + 1
            chosen = accepted(qid)
            counts["accepted"] += len(chosen)
            counts["questions_accepted"] += bool(chosen)
            for s in chosen:
                request_id = hashlib.sha256(f"{args.domain}:{qid}:{s['sample_index']}".encode()).hexdigest()
                out.write(json.dumps(dict(request_id=request_id, question_id=qid, sample_index=s["sample_index"],
                                          input_ids=prompt + s["output_ids"],
                                          labels=[-100] * len(prompt) + s["output_ids"], prefix_tokens=len(prompt),
                                          supervised_tokens=len(s["output_ids"]), score=1)) + "\n")
    (args.output / "summary.json").write_text(json.dumps(dict(counts, teacher=f"{repo}@{revision}"), indent=2))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
