"""BFCL v3 (Table 3) for M2RL-SFT with or without a composed soft prompt.

  python experiments/m2rl/ood/bfcl/run.py --prompt artifacts/MCSI.safetensors \
      --bfcl-python /path/to/bfcl-env/bin/python --bfcl-root /path/to/gorilla/berkeley-function-call-leaderboard \
      --output runs/bfcl/MCSI.jsonl
  python experiments/m2rl/ood/bfcl/aggregate.py runs/bfcl/MCSI.jsonl

BFCL environment (separate from the SGLang one):

  git clone https://github.com/ShishirPatil/gorilla && git -C gorilla checkout cd9429ccf3d4d04156affe883c495b3b047e6b64
  python -m venv bfcl-env && bfcl-env/bin/pip install -e gorilla/berkeley-function-call-leaderboard soundfile

Each case runs in its own worker.py process with BFCL's official QwenFCHandler (Qwen/Qwen3-4B-FC
format) and checker, from gorilla commit cd9429ccf3d4d04156affe883c495b3b047e6b64. This script
serves the model with SGLang: every formatted prompt is tokenized, the soft-prompt rows are placed
right after the system header (as in all other evaluations), and generation uses temperature 0.6,
top-p 0.95, top-k 20 and at most 32,768 new tokens per step. A step whose prompt leaves no room for
32,768 new tokens in the 65,536-token context ends the case with score 0. Up to 20 steps per user turn
(BFCL's own limit). The 4,041 cases are listed in cases.json; finished cases are skipped on restart.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "eval"))
from generate import STOP_IDS, SYSTEM_CLOSE, SYSTEM_HEADER  # noqa: E402

CONTEXT, MAX_NEW = 65536, 32768


def insert_prompt(ids, header, close, virtual):
    if ids[:len(header)] != header:
        ids = header + close + ids
    return ids[:len(header)] + list(virtual) + ids[len(header):]


async def run_case(case, args, engine, tokenizer, virtual, masked, out, lock):
    header = tokenizer.encode(SYSTEM_HEADER, add_special_tokens=False)
    close = tokenizer.encode(SYSTEM_CLOSE, add_special_tokens=False)
    # Plain Popen with blocking pipe I/O in threads: SGLang installs uvloop, which has no asyncio child watcher.
    process = subprocess.Popen([args.bfcl_python, str(HERE / "worker.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, env={**os.environ, "BFCL_ROOT": str(args.bfcl_root)}, text=True)

    def _send(value):
        process.stdin.write(json.dumps(value) + "\n")
        process.stdin.flush()

    async def send(value):
        await asyncio.to_thread(_send, value)

    await send({"question_id": case["question_id"], "category": case["category"]})
    result = None
    while True:
        line = await asyncio.to_thread(process.stdout.readline)
        if not line:
            break
        message = json.loads(line)
        if message["type"] == "complete":
            result = {k: message[k] for k in ("question_id", "category", "score", "model_queries", "handler_error")}
            if args.save_records:
                result["record"] = message["record"]
            break
        ids = insert_prompt(tokenizer.encode(message["prompt"], add_special_tokens=False), header, close, virtual)
        if len(ids) + MAX_NEW > CONTEXT:
            result = dict(question_id=case["question_id"], category=case["category"], score=0, context_overflow=True)
            break
        seed = int.from_bytes(hashlib.sha256(f"{case['question_id']}:{message['step']}".encode()).digest()[:4], "little") % (2**31)
        response = await engine.async_generate(input_ids=ids, sampling_params={
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": MAX_NEW, "stop_token_ids": list(STOP_IDS),
            "sampling_seed": seed, "skip_special_tokens": False, "spaces_between_special_tokens": False,
            "logit_bias": {int(t): -1e9 for t in masked}})
        output = list(response["output_ids"])
        while output and output[-1] in STOP_IDS:
            output.pop()
        text = tokenizer.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        await send({"type": "response", "step": message["step"], "text": text,
                    "input_tokens": len(ids), "output_tokens": len(response["output_ids"])})
    if process.poll() is None:
        process.kill()
    await asyncio.to_thread(process.wait)
    if result is None:
        result = dict(question_id=case["question_id"], category=case["category"], score=0, worker_failed=True)
    async with lock:
        out.write(json.dumps(result) + "\n")
        out.flush()


async def main_async(args):
    import sglang
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    source = Path(snapshot_download(args.model, revision=args.revision))
    tokenizer = AutoTokenizer.from_pretrained(source)
    model_dir, virtual = source, []
    if args.prompt:
        from safetensors.torch import load_file
        from derived_model import build_derived_model
        model_dir = Path(tempfile.mkdtemp(prefix="cspd-derived-")) / "model"
        virtual = build_derived_model(source, model_dir, load_file(args.prompt)["soft_prompt"].float())
    masked = list(range(len(tokenizer), len(tokenizer) + 256)) + list(virtual)
    engine = sglang.Engine(model_path=str(model_dir), tokenizer_path=str(source), dtype="bfloat16", context_length=CONTEXT,
                           random_seed=42, sampling_backend="pytorch",
                           attention_backend="triton", mem_fraction_static=0.8, log_level="warning")
    cases = json.load(open(HERE / "cases.json"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["question_id"] for l in args.output.open()} if args.output.exists() else set()
    todo = [c for c in cases if c["question_id"] not in done]
    if args.limit:  # smoke test: the first N cases of every category
        per = {}
        todo = [c for c in todo if per.setdefault(c["category"], []).append(c) or len(per[c["category"]]) <= args.limit]
    from concurrent.futures import ThreadPoolExecutor
    # each running case blocks one thread on its worker's stdout
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=2 * args.concurrency + 8))
    lock, gate = asyncio.Lock(), asyncio.Semaphore(args.concurrency)
    with args.output.open("a") as out:
        async def guarded(case):
            async with gate:
                await run_case(case, args, engine, tokenizer, virtual, masked, out, lock)
        await asyncio.gather(*(guarded(c) for c in todo))
    engine.shutdown()
    print(json.dumps({"completed": len(todo), "output": str(args.output)}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Jackwang111/M2RL-SFT")
    p.add_argument("--revision", default="aa49d49a86b15ac920cf356c359f299edd0c1eb3")
    p.add_argument("--prompt", type=Path, help="composed soft prompt from compose.py (omit for the student)")
    p.add_argument("--bfcl-python", required=True, help="python of an environment with BFCL installed")
    p.add_argument("--bfcl-root", type=Path, required=True, help="gorilla/berkeley-function-call-leaderboard at cd9429c")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--save-records", action="store_true", help="keep the full BFCL record (model outputs, decoded calls)")
    p.add_argument("--limit", type=int, help="smoke test: run only the first N cases of each category")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
