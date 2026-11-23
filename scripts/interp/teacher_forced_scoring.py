#!/usr/bin/env python3
"""Teacher-forced per-token scoring of traces under teacher / student / student+SP arms.

Step 2 of the token-level analysis (Appendix G).  The teacher
runs on one GPU and the student on another; for every response token we keep
scalar statistics only (log-prob of the realised token, entropy, rank, argmax,
KL from the teacher) plus top-5 lists for T, S and the reference SP arm.  The
soft prompt is spliced into ``inputs_embeds`` exactly as in ``training.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from prompt_variants import VARIANTS, apply_variant

CHUNK = 2048
TOPK = 5
# per-position log-probs of a few decision tokens, recorded for every arm
WATCH_TOKENS = ("<|end|>", "Wait", " Wait", " But", " So", " Let", " Therefore", " Thus")


def load_model(path: Path, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        str(path), dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="flex_attention",
    ).to(device).eval()
    return model


def load_prompt(path: Path) -> torch.Tensor:
    with safe_open(str(path), framework="pt") as handle:
        return handle.get_tensor("prompt_embeddings")


@torch.no_grad()
def response_hidden(model, prompt_ids, response_ids, insert_at=None, soft=None) -> torch.Tensor:
    """Final-norm hidden states at the positions that predict each response token."""

    device = model.device
    ids = torch.tensor([list(prompt_ids) + list(response_ids)], device=device)
    embed = model.get_input_embeddings()(ids)
    if soft is not None:
        soft = soft.to(device=device, dtype=embed.dtype)[None]
        embed = torch.cat([embed[:, :insert_at], soft, embed[:, insert_at:]], dim=1)
    out = model.model(inputs_embeds=embed, attention_mask=torch.ones(embed.shape[:2], device=device,
                      dtype=torch.long), use_cache=False)
    hidden = out.last_hidden_state[0]
    start = len(prompt_ids) + (soft.size(1) if soft is not None else 0) - 1
    return hidden[start:start + len(response_ids)]


@torch.no_grad()
def chunk_stats(logp: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    tgt = logp.gather(1, target[:, None])[:, 0]
    return {
        "logp": tgt,
        "entropy": -(logp.exp() * logp).sum(-1),
        "rank": (logp > tgt[:, None]).sum(-1),
        "argmax": logp.argmax(-1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--student-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="*", default=[], choices=VARIANTS,
                        help="natural-language prompt arms, scored as S+<variant>")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="name=path pairs, e.g. u0=... u40=... u180=...")
    parser.add_argument("--reference-arm", default="u180")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--teacher-device", default="cuda:0")
    parser.add_argument("--student-device", default="cuda:1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--teacher-hidden-cache", type=Path, default=None,
                        help="directory of cached teacher final hidden states; lets one GPU "
                             "hold only the student")
    parser.add_argument("--cache-teacher-only", action="store_true",
                        help="only fill --teacher-hidden-cache with the teacher, then exit")
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    traces = []
    for path in args.traces:
        traces += [json.loads(line) for line in path.open()]
    traces = traces[args.shard::args.num_shards]
    if args.limit:
        traces = traces[: args.limit]
    soft = {k: load_prompt(Path(v)) for k, v in (item.split("=", 1) for item in args.checkpoints)}
    arms = ["T", "S"] + [f"S+{k}" for k in soft] + [f"S+{v}" for v in args.variants]
    ref = f"S+{args.reference_arm}"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.student_dir))
    watch = {}
    for text in WATCH_TOKENS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            watch[text] = ids[0]
    cache = args.teacher_hidden_cache

    def cache_path(tr: dict) -> Path:
        return cache / f"{tr['source']}__{tr['question_id'].replace('/', '_')}__{tr['sample']}.pt"

    if args.cache_teacher_only:
        cache.mkdir(parents=True, exist_ok=True)
        teacher = load_model(args.teacher_dir, args.teacher_device)
        for n, tr in enumerate(traces):
            path = cache_path(tr)
            if not path.exists():
                h = response_hidden(teacher, tr["prompt_token_ids"], tr["response_token_ids"])
                torch.save(h.cpu(), path)
            if n % 50 == 0:
                print(f"[cache {n + 1}/{len(traces)}]", flush=True)
        print(f"cached teacher hidden states for {len(traces)} traces in {cache}")
        return

    dev = args.student_device
    student = load_model(args.student_dir, dev)
    if cache is None:
        teacher = load_model(args.teacher_dir, args.teacher_device)
        teacher_head = teacher.lm_head.weight.to(dev)
    else:
        teacher = None
        index = json.loads((args.teacher_dir / "model.safetensors.index.json").read_text())
        with safe_open(str(args.teacher_dir / index["weight_map"]["lm_head.weight"]),
                       framework="pt") as handle:
            teacher_head = handle.get_tensor("lm_head.weight").to(dev, torch.bfloat16)
    W = {"T": teacher_head, "S": student.lm_head.weight}

    store: dict[str, list[np.ndarray]] = {}
    meta = []
    offset = 0
    t0 = time.time()
    for n, tr in enumerate(traces):
        p_ids, r_ids, ins = tr["prompt_token_ids"], tr["response_token_ids"], tr["soft_prompt_insert_at"]
        h_t = (torch.load(cache_path(tr)) if teacher is None
               else response_hidden(teacher, p_ids, r_ids))
        hidden = {"T": h_t.to(dev), "S": response_hidden(student, p_ids, r_ids)}
        for k, sp in soft.items():
            hidden[f"S+{k}"] = response_hidden(student, p_ids, r_ids, ins, sp)
        for v in args.variants:
            hidden[f"S+{v}"] = response_hidden(
                student, apply_variant(p_ids, ins, v, tokenizer), r_ids)
        target = torch.tensor(r_ids, device=dev)
        rows: dict[str, list[torch.Tensor]] = {}
        for s in range(0, len(r_ids), CHUNK):
            sl = slice(s, s + CHUNK)
            tgt = target[sl]
            lp_t = F.log_softmax((hidden["T"][sl] @ W["T"].T).float(), dim=-1)
            p_t = lp_t.exp()
            for arm in arms:
                lp = lp_t if arm == "T" else F.log_softmax(
                    (hidden[arm][sl] @ W["S"].T).float(), dim=-1)
                for key, val in chunk_stats(lp, tgt).items():
                    rows.setdefault(f"{key}:{arm}", []).append(val)
                if arm != "T":
                    rows.setdefault(f"kl_T:{arm}", []).append((p_t * (lp_t - lp)).sum(-1))
                rows.setdefault(f"watch_logp:{arm}", []).append(lp[:, list(watch.values())])
                if arm in ("T", "S", ref):
                    tv, ti = lp.topk(TOPK, dim=-1)
                    rows.setdefault(f"top_ids:{arm}", []).append(ti)
                    rows.setdefault(f"top_logp:{arm}", []).append(tv)
                del lp
        for key, parts in rows.items():
            store.setdefault(key, []).append(torch.cat(parts).cpu().numpy())
        store.setdefault("token", []).append(np.asarray(r_ids, dtype=np.int64))
        meta.append({k: tr[k] for k in ("source", "dataset", "question_id", "sample", "correct",
                                        "truncated", "level")} | {"offset": offset,
                                                                    "length": len(r_ids)})
        offset += len(r_ids)
        if n % 10 == 0:
            rate = offset / (time.time() - t0)
            print(f"[{n + 1}/{len(traces)}] tokens {offset} ({rate:.0f} tok/s)", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, parts in store.items():
        arr = np.concatenate(parts)
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        elif key.startswith(("rank", "argmax", "top_ids", "token")):
            arr = arr.astype(np.int32)
        arrays[key.replace(":", "__").replace("+", "_")] = arr
    np.savez(args.output, **arrays)
    args.output.with_suffix(".meta.json").write_text(json.dumps({
        "arms": arms, "reference_arm": ref, "traces": meta,
        "checkpoints": dict(item.split("=", 1) for item in args.checkpoints),
        "watch_tokens": watch,
        "variants": list(args.variants),
    }, ensure_ascii=False))
    print(f"wrote {args.output} ({offset} tokens, {len(meta)} traces)")


if __name__ == "__main__":
    main()
