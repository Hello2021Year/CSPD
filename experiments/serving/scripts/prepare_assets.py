"""Independent nonzero adapters, actual text token IDs, reproducible manifests."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def adapter_seed(arm, index):
    if arm == 'r4':
        return 402611140000 + index
    return 202611140000 + index * 3 + ['sp', 'r1', 'r16'].index(arm)


def adapter_shapes(cfg):
    h, inter = cfg['hidden_size'], cfg['intermediate_size']
    hd = cfg.get('head_dim', h // cfg['num_attention_heads'])
    shapes = {'self_attn.q_proj': (h, cfg['num_attention_heads'] * hd),
              'self_attn.k_proj': (h, cfg['num_key_value_heads'] * hd),
              'self_attn.v_proj': (h, cfg['num_key_value_heads'] * hd),
              'self_attn.o_proj': (cfg['num_attention_heads'] * hd, h),
              'mlp.gate_proj': (h, inter), 'mlp.up_proj': (h, inter),
              'mlp.down_proj': (inter, h)}
    if cfg.get('num_local_experts', 0):
        for module in ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
            dims = shapes.pop(module)
            for expert in range(cfg['num_local_experts']):
                shapes[f'mlp.experts.{expert}.{module.split(".")[-1]}'] = dims
    return shapes


def make_adapter(cfg, arm, index):
    """Identical seeded weights for offline and arrival-time preparation."""
    import torch
    g = torch.Generator().manual_seed(adapter_seed(arm, index))
    if arm == 'sp':
        return {'soft_prompt': (torch.randn(256, cfg['hidden_size'], generator=g)
                               * cfg.get('initializer_range', .02)).to(torch.bfloat16)}
    rank, tensors = int(arm[1:]), {}
    for layer in range(cfg['num_hidden_layers']):
        for module, (n_in, n_out) in adapter_shapes(cfg).items():
            key = f'base_model.model.model.layers.{layer}.{module}'
            tensors[key + '.lora_A.weight'] = (torch.randn(rank, n_in, generator=g) / math.sqrt(n_in)).to(torch.bfloat16)
            tensors[key + '.lora_B.weight'] = (torch.randn(n_out, rank, generator=g) * .01).to(torch.bfloat16)
    return tensors


def adapter_config(experiment, cfg, arm):
    rank = int(arm[1:])
    return dict(base_model_name_or_path=experiment['model_repo'], peft_type='LORA',
                task_type='CAUSAL_LM', r=rank, lora_alpha=2*rank,
                lora_dropout=0., bias='none',
                target_modules=sorted({s.split('.')[-1] for s in adapter_shapes(cfg)}))


def main():
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out')
    ap.add_argument('--count', type=int, default=16)
    ap.add_argument('--arms', nargs='+', default=['sp', 'r1', 'r16'])
    a = ap.parse_args()
    experiment = json.loads(Path('experiment.json').read_text())
    root = Path(a.out or experiment['assets'])
    root.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((Path(a.model) / 'config.json').read_text())
    h, inter = cfg['hidden_size'], cfg['intermediate_size']
    hd = cfg.get('head_dim', h // cfg['num_attention_heads'])
    shapes = {'self_attn.q_proj': (h, cfg['num_attention_heads'] * hd),
              'self_attn.k_proj': (h, cfg['num_key_value_heads'] * hd),
              'self_attn.v_proj': (h, cfg['num_key_value_heads'] * hd),
              'self_attn.o_proj': (cfg['num_attention_heads'] * hd, h),
              'mlp.gate_proj': (h, inter), 'mlp.up_proj': (h, inter),
              'mlp.down_proj': (inter, h)}
    experts = cfg.get('num_local_experts', 0)
    if experts:
        for module in ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
            dims = shapes.pop(module)
            for expert in range(experts):
                shapes[f'mlp.experts.{expert}.{module.split(".")[-1]}'] = dims
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    paragraph = ('A serving system receives independent requests over time. Each request '
                 'contains text to process and produces a fixed number of output tokens. '
                 'We record arrival times, queueing, memory transfers, and generation times. '
                 'Comparing identical workloads helps distinguish execution overhead from '
                 'differences in the text that a language model chooses to generate. ')
    for i in range(a.count):
        ip = root / f'input-{i:06d}.json'
        if not ip.exists():
            ids = tokenizer.encode(f'Request {i}. ' + paragraph * 60, add_special_tokens=False)[:1024]
            assert len(ids) == 1024 and experiment['placeholder_id'] not in ids
            ip.write_text(json.dumps(ids))
    for arm in a.arms:
        manifest = []
        for i in range(a.count):
            seed = adapter_seed(arm, i)
            path = root / arm / f'{i:06d}'
            path.mkdir(parents=True, exist_ok=True)
            weights = path / ('prompt.safetensors' if arm == 'sp' else 'adapter_model.safetensors')
            if not weights.exists():
                tensors = make_adapter(cfg, arm, i)
                if arm != 'sp':
                    (path / 'adapter_config.json').write_text(json.dumps(adapter_config(experiment, cfg, arm)))
                assert all(torch.count_nonzero(t).item() > 0 for t in tensors.values())
                save_file(tensors, str(weights) + '.tmp')
                Path(str(weights) + '.tmp').replace(weights)
            manifest.append(dict(adapter_id=i, seed=seed, arm=arm,
                                 synthetic=True, serving_only=True,
                                 path=str(weights.resolve()), sha256=digest(weights),
                                 bytes=weights.stat().st_size,
                                 expert_count=experts, expert_lora_format=experiment['expert_lora_format'],
                                 model_revision=json.loads(Path('model-provenance.json').read_text())['revision'],
                                 input_sha256=digest(root / f'input-{i:06d}.json')))
            if i % 32 == 0:
                print(arm, i, a.count, flush=True)
        assert len({r['sha256'] for r in manifest}) == len(manifest)
        manifest_tmp = root / f'{arm}-manifest.json.tmp'
        manifest_tmp.write_text(json.dumps(manifest, indent=2))
        manifest_tmp.replace(root / f'{arm}-manifest.json')


if __name__ == '__main__':
    main()
