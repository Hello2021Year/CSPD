"""Train the K/V LoRA baseline of Table 4 (GPT-OSS-Swallow-20B, rank 5, alpha 5).

The LoRA arm uses the same correct-only responses, global batches and sequence-mean
response loss as the soft-prompt arm; only the trainable module differs (rank-5 adapters
on every k_proj and v_proj, FP32 masters, BF16 compute). This driver is a reconstruction:
the adapter module, export and evaluation code (gptoss_lora_core.py, gptoss_lora_evaluate.py)
are the originals, but the original training script was not preserved, so results need not
match Table 4 exactly.

Training data (``--data``) is a TokenData directory: ``tokens.i32`` (prompt and response
tokens back to back), ``index.npy`` (rows of offset, prompt length, response length) and
``identity.json``.

  torchrun --nproc-per-node 4 scripts/gptoss_lora/gptoss_lora_train.py \\
      --model tokyotech-llm/GPT-OSS-Swallow-20B-SFT-v0.1 --data runs/gptoss/correct-only-tokens \\
      --output runs/gptoss/lora-kv
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gptoss_lora_core import TokenData, adapter_state, collate, install_lora, microbatches, response_loss  # noqa: E402


def batch_indices(n, batch, seed, update):
    """Same deterministic drop-last schedule as soft_prompt_distillation.training."""
    steps = n // batch
    epoch, position = (update - 1) // steps, (update - 1) % steps
    order = list(range(n))
    random.Random(seed + epoch).shuffle(order)
    return order[position * batch:(position + 1) * batch]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rank", type=int, default=5)
    p.add_argument("--alpha", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=20)
    args = p.parse_args(argv)
    from transformers import AutoModelForCausalLM

    distributed = "RANK" in os.environ
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=dtype).to(device)
    model.config.use_cache = False
    params = install_lora(model, args.rank, args.alpha, args.seed)
    optimizer = torch.optim.AdamW(params.values(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.0)
    if device.type == "cuda":
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        loss_fn = LigerFusedLinearCrossEntropyLoss(reduction="mean")
    else:
        def loss_fn(weight, hidden, targets, bias=None):
            return torch.nn.functional.cross_entropy(torch.nn.functional.linear(hidden, weight, bias).float(), targets)
    data = TokenData(args.data)
    pad = model.config.eos_token_id if isinstance(model.config.eos_token_id, int) else 0
    (args.output / "adapters").mkdir(parents=True, exist_ok=True)
    for update in range(1, args.updates + 1):
        indices = batch_indices(len(data), args.batch_size, args.seed, update)[rank::world]
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for micro in microbatches(indices, data, args.micro_batch):
            for i in micro:  # sequence-mean loss: each response weighs 1 / global batch
                loss, _ = response_loss(model, collate([data[i]], pad, device), loss_fn)
                (loss / args.batch_size).backward()
                total += float(loss.detach())
        grads = [q.grad for q in params.values()]
        if distributed:
            for g in grads:
                dist.all_reduce(g)
            value = torch.tensor([total], device=device)
            dist.all_reduce(value)
            total = float(value)
        torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
        optimizer.step()
        if rank == 0:
            print(json.dumps({"update": update, "loss": total / args.batch_size}), flush=True)
            if update % args.save_every == 0 or update == args.updates:
                from safetensors.torch import save_file
                save_file(adapter_state(model), args.output / "adapters" / f"update_{update:08d}.safetensors")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
