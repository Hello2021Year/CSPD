"""Generate K samples per question for the M2RL benchmarks, with or without soft prompts.

  python experiments/m2rl/eval/generate.py --bench aime24 --k 64 \
      --prompt artifacts/m2rl/MCSI.safetensors --output runs/eval/MCSI/aime24.jsonl

Inputs are ``inputs/<bench>/questions.jsonl`` next to this script (chat messages; build the GPQA
file with build_gpqa.py). Prompts are rendered with the Qwen3 chat template and thinking enabled. An empty
system turn is added when absent, and the soft-prompt rows are placed right after the
system header, as in training. Decoding follows the paper: temperature 0.6, top-p 0.95,
top-k 20, at most 32,768 new tokens.

The default engine is SGLang, the engine used for the paper's numbers (environment:
scripts/setup_blackwell_runtime.sh). ``--engine vllm`` (vLLM 0.11, experiments/m2rl/requirements.txt)
runs the same protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

PAPER_K = {"aime24": 64, "lcb_v5": 8, "gpqa_diamond": 16, "ifbench": 8,
           # out-of-domain (Table 3), inputs from ood/build_inputs.py: --inputs experiments/m2rl/ood/inputs
           "triviaqa": 1, "truthfulqa": 4, "xstest": 4, "bbeh": 1, "finqa": 1}
STOP_IDS = (151645, 151643)
SYSTEM_HEADER = "<|im_start|>system\n"
SYSTEM_CLOSE = "<|im_end|>\n"


def render(tokenizer, messages, virtual_ids):
    """Token ids with ``virtual_ids`` inserted right after the system header."""
    header = tokenizer.encode(SYSTEM_HEADER, add_special_tokens=False)
    close = tokenizer.encode(SYSTEM_CLOSE, add_special_tokens=False)
    ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                             enable_thinking=True, return_dict=False))
    if ids[:len(header)] != header:
        if ids[:3] != [151644, 872, 198]:  # <|im_start|>user\n
            raise ValueError("expected a system or user turn first")
        ids = header + close + ids
    return ids[:len(header)] + list(virtual_ids) + ids[len(header):]


def generate_vllm(args, model_dir, source, tokenizer, inputs, k):
    # The vLLM engine subprocess exits silently on the derived (extended-vocabulary) checkpoint
    # in vLLM 0.11; running the engine in-process avoids it and does not change the outputs.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams
    # Only tokenizer tokens may be generated: this masks the unused embedding rows and every
    # appended prompt row (vLLM requires allowed ids to lie inside the tokenizer vocabulary).
    params = SamplingParams(n=k, temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768, seed=args.seed,
                            stop_token_ids=list(STOP_IDS), skip_special_tokens=False,
                            allowed_token_ids=list(range(len(tokenizer))))
    llm = LLM(model=str(model_dir), tokenizer=str(source), dtype="bfloat16", max_model_len=65536,
              tensor_parallel_size=args.tensor_parallel_size, seed=args.seed)
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in inputs], params)
    return [[(o.token_ids, o.finish_reason) for o in out.outputs] for out in outputs]


def generate_sglang(args, model_dir, source, tokenizer, inputs, virtual, k):
    """SGLang with the paper's settings: PyTorch sampling, per-request
    seeds, and an output mask over the 256 unused tokenizer rows and the appended prompt rows.

    The mask must be effectively hard: Qwen3 ties input and output embeddings, so the appended
    prompt rows get very large logits and a bias such as -100 does not keep them from being sampled."""
    import hashlib
    import sglang
    masked = list(range(len(tokenizer), len(tokenizer) + 256)) + list(virtual)
    engine = sglang.Engine(model_path=str(model_dir), tokenizer_path=str(source), dtype="bfloat16",
                           context_length=65536, tp_size=args.tensor_parallel_size, random_seed=args.seed,
                           sampling_backend="pytorch",
                           attention_backend="triton", mem_fraction_static=0.8, log_level="warning")
    requests, params = [], []
    for qi, ids in enumerate(inputs):
        for sample in range(k):
            seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{qi}:{sample}".encode()).digest()[:4], "little") % (2**31)
            requests.append(ids)
            params.append({"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 32768,
                           "stop_token_ids": list(STOP_IDS), "sampling_seed": seed, "no_stop_trim": True,
                           "skip_special_tokens": False, "spaces_between_special_tokens": False,
                           "logit_bias": {int(t): -1e9 for t in masked}})
    results = engine.generate(input_ids=requests, sampling_params=params)
    engine.shutdown()
    flat = [(r["output_ids"], r["meta_info"]["finish_reason"]["type"]) for r in results]
    return [flat[i * k:(i + 1) * k] for i in range(len(inputs))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True, choices=sorted(PAPER_K))
    p.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parent / "inputs")
    p.add_argument("--model", default="Jackwang111/M2RL-SFT")
    p.add_argument("--revision", default="aa49d49a86b15ac920cf356c359f299edd0c1eb3")
    p.add_argument("--prompt", type=Path, help="composed soft prompt (.safetensors from compose.py)")
    p.add_argument("--k", type=int, help="samples per question (default: the paper's K)")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--engine", choices=["sglang", "vllm"], default="sglang",
                   help="sglang (default) is the engine used for the paper's numbers")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    source = Path(snapshot_download(args.model, revision=args.revision))
    tokenizer = AutoTokenizer.from_pretrained(source)
    model_dir, virtual = source, []
    if args.prompt:
        from safetensors.torch import load_file
        from derived_model import build_derived_model
        prompt = load_file(args.prompt)["soft_prompt"].float()
        model_dir = Path(tempfile.mkdtemp(prefix="cspd-derived-")) / "model"
        virtual = build_derived_model(source, model_dir, prompt)

    k = args.k or PAPER_K[args.bench]
    questions = [json.loads(line) for line in (args.inputs / args.bench / "questions.jsonl").open()]
    inputs = [render(tokenizer, q["messages"], virtual) for q in questions]
    samples = generate_sglang(args, model_dir, source, tokenizer, inputs, virtual, k) if args.engine == "sglang" \
        else generate_vllm(args, model_dir, source, tokenizer, inputs, k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for question, outs in zip(questions, samples):
            for index, (ids, finish_reason) in enumerate(outs):
                ids = list(ids)
                while ids and ids[-1] in STOP_IDS:
                    ids.pop()
                text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                stream.write(json.dumps({"question_id": question["question_id"], "sample_index": index, "text": text,
                                         "finish_reason": finish_reason}, ensure_ascii=False) + "\n")
    print(json.dumps({"bench": args.bench, "questions": len(questions), "k": k, "output": str(args.output)}))


if __name__ == "__main__":
    main()
