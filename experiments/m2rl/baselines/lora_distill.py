"""LoRA (or full fine-tuning) distillation baseline on the soft-prompt training data (Table 2).

Uses exactly the prompt runs' packed teacher responses and global batches: response-only CE with
sequence-mean loss, batch_indices_for_update(seed 42), 200 updates of 64 responses balanced over the
ranks, gradient clipping 1.0, cosine LR from LR to LR/10. The empty system turn that packing adds is
removed. Paper setting: LoRA rank 16, alpha 32, LR 2e-4 on all attention and MLP projections.
The merged model is written to <output>/model.

  torchrun --nproc-per-node 4 experiments/m2rl/baselines/lora_distill.py \
      --data data/m2rl/if --output runs/lora/if
"""
import argparse
import json, math, os, time
import numpy as np, torch, torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from soft_prompt_distillation.training import batch_indices_for_update

from huggingface_hub import snapshot_download

_p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument('--data', required=True, help='packed domain directory (index.json, tokens.uint32)')
_p.add_argument('--output', required=True)
_p.add_argument('--mode', choices=['lora', 'full'], default='lora')
_p.add_argument('--lr', type=float, default=2e-4)
_p.add_argument('--lora-rank', type=int, default=16)
_p.add_argument('--lora-alpha', type=int, default=32)
_a = _p.parse_args()
MODEL = snapshot_download('Jackwang111/M2RL-SFT', revision='aa49d49a86b15ac920cf356c359f299edd0c1eb3')
MODE, LR, OUT, DATA = _a.mode, _a.lr, _a.output, _a.data
UPDATES, GB = 200, 64

def balanced(indices, entries, world):
    groups, loads, cap = [[] for _ in range(world)], [0] * world, len(indices) // world
    for i in sorted(indices, key=lambda i: (-entries[i]['length'], i)):
        r = min((r for r in range(world) if len(groups[r]) < cap), key=lambda r: (loads[r], r))
        groups[r].append(i); loads[r] += entries[i]['length']
    return groups

def main():
    dist.init_process_group('nccl'); rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK'])); dev = torch.device('cuda')
    entries = json.load(open(DATA + '/index.json')); tokens = np.memmap(DATA + '/tokens.uint32', dtype='<u4', mode='r')
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation='sdpa').to(dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False)); model.config.use_cache = False
    if MODE == 'lora':
        from peft import LoraConfig, get_peft_model
        r = _a.lora_rank
        model = get_peft_model(model, LoraConfig(r=r, lora_alpha=_a.lora_alpha, lora_dropout=0.0, bias='none',
                               target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
        model.enable_input_require_grads()
    model.train()  # HF only applies gradient checkpointing in training mode
    params = [p for p in model.parameters() if p.requires_grad]
    if MODE == 'lora':
        for p in params: p.data = p.data.float()  # fp32 adapters on a bf16 frozen backbone (bf16 q/k/v keeps the flash kernel)
    for p in params:  # identical start on every rank (LoRA A is randomly initialised)
        dist.broadcast(p.data, 0)
    ntrain = sum(p.numel() for p in params)
    # full: bf16 working weights + fp32 master copies for AdamW
    master = [p.detach().float().clone() for p in params] if MODE == 'full' else params
    opt = torch.optim.AdamW(master, lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    if rank == 0:
        os.makedirs(OUT, exist_ok=True)
        print(f'[lf] mode={MODE} data={DATA} lr={LR} trainable={ntrain} records={len(entries)}', flush=True)
    log = open(OUT + '/train-log.jsonl', 'a') if rank == 0 else None
    t0 = time.time()
    for upd in range(1, UPDATES + 1):
        g = batch_indices_for_update(dataset_size=len(entries), effective_batch_size=GB, batch_seed=42, update=upd)
        mine = balanced(g, entries, world)[rank]
        lr = LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (upd - 1) / (UPDATES - 1))))
        for pg in opt.param_groups: pg['lr'] = lr
        opt.zero_grad(set_to_none=True); loss_sum = 0.0
        for i in mine:
            e = entries[i]; ids = np.asarray(tokens[e['offset']:e['offset'] + e['length']], dtype=np.int64)
            assert list(ids[:5]) == [151644, 8948, 198, 151645, 198] and e['insert_at'] == 3, 'expected the added empty system turn'
            ids = torch.tensor(ids[5:], device=dev)[None]; prefix = e['prefix_tokens'] - 5
            base = model.get_base_model() if MODE == 'lora' else model
            from torch.nn.attention import sdpa_kernel, SDPBackend  # as the SP trainer: force flash (math fallback is O(L^2) memory)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                h = base.model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0, prefix - 1:-1]
                tgt = ids[0, prefix:]; l = 0.0
                # chunked, checkpointed LM head + CE: never materialise the full [L, 151936] fp32 logits (OOM for 30k-token code)
                for c in range(0, h.size(0), 2048):
                    l = l + torch.utils.checkpoint.checkpoint(
                        lambda hh, tt: torch.nn.functional.cross_entropy(base.lm_head(hh).float(), tt, reduction='sum'),
                        h[c:c + 2048], tgt[c:c + 2048], use_reentrant=False)
                l = l / tgt.numel()
            (l / GB).backward(); loss_sum += l.item()
        for p in params:
            if p.grad is None: p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)
        if MODE == 'full':
            for m, p in zip(master, params): m.grad = p.grad.float(); p.grad = None
        gn = torch.nn.utils.clip_grad_norm_(master, 1.0).item()
        opt.step()
        if MODE == 'full':
            for m, p in zip(master, params): p.data.copy_(m.data); m.grad = None
        tot = torch.tensor([loss_sum], device=dev); dist.all_reduce(tot)
        if rank == 0:
            rec = dict(update=upd, loss=tot.item() / GB, grad_norm=gn, lr=lr, elapsed=time.time() - t0)
            log.write(json.dumps(rec) + '\n'); log.flush()
            print('[lf]', json.dumps(rec), flush=True)
    if rank == 0:
        m = model.merge_and_unload() if MODE == 'lora' else model
        m = m.to(torch.bfloat16); m.config.use_cache = True
        m.save_pretrained(OUT + '/model', safe_serialization=True)
        AutoTokenizer.from_pretrained(MODEL).save_pretrained(OUT + '/model')
        for f in ('generation_config.json',):
            if os.path.exists(os.path.join(MODEL, f)):
                import shutil; shutil.copy(os.path.join(MODEL, f), OUT + '/model/' + f)
        print('[lf] saved', OUT + '/model', flush=True)
    dist.barrier(); dist.destroy_process_group()

if __name__ == '__main__':
    main()
