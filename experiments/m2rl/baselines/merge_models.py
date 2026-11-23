"""Data-free merging baselines (Table 2, Appendix C): Task Arithmetic and TIES.

Task vector of model d: tau_d = theta_d - theta_student (computed in FP32).
  ta   : theta = student + LAM * sum_d tau_d          (paper: LAM 1.0 for the RL experts, 0.5 for LoRA)
  ties : trim each tau_d to its top-DENSITY magnitudes, elect a sign per entry, disjoint mean of the
         agreeing entries; theta = student + LAM * merged   (paper: DENSITY 0.2, LAM 1.0)
  avg, dare_ta, dare_ties are also available.

  python experiments/m2rl/baselines/merge_models.py --method ties --output runs/merge/ties \
      --models Jackwang111/M2RL-RL_Math Jackwang111/M2RL-RL_Coding Jackwang111/M2RL-RL_Science Jackwang111/M2RL-RL_IF

MOPD is the released Jackwang111/M2RL-MT_OPD checkpoint and needs no merging.
"""
import argparse
import json, os, shutil
import torch
from safetensors.torch import load_file, save_file

from huggingface_hub import snapshot_download

_p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument('--method', required=True, choices=['avg', 'ta', 'ties', 'dare_ta', 'dare_ties'])
_p.add_argument('--models', nargs='+', required=True, help='local directories or Hub ids')
_p.add_argument('--output', required=True)
_p.add_argument('--student', default='Jackwang111/M2RL-SFT')
_p.add_argument('--student-revision', default='aa49d49a86b15ac920cf356c359f299edd0c1eb3')
_p.add_argument('--lam', type=float)
_p.add_argument('--density', type=float, default=0.2)
_p.add_argument('--drop', type=float, default=0.8)
_p.add_argument('--seed', type=int, default=0)
_a = _p.parse_args()
_local = lambda m, rev=None: m if os.path.isdir(m) else snapshot_download(m, revision=rev)
STUDENT = _local(_a.student, _a.student_revision)
METHOD, MODELS, OUT = _a.method, [_local(m) for m in _a.models], _a.output
LAM = _a.lam if _a.lam is not None else (0.5 if METHOD.endswith('ta') else 1.0)
DENSITY, DROP = _a.density, _a.drop
g = torch.Generator().manual_seed(_a.seed)

def load_all(d):
    idx = json.load(open(os.path.join(d, 'model.safetensors.index.json')))['weight_map'] if os.path.exists(os.path.join(d, 'model.safetensors.index.json')) else None
    files = sorted(set(idx.values())) if idx else [f for f in os.listdir(d) if f.endswith('.safetensors')]
    sd = {}
    for f in files: sd.update(load_file(os.path.join(d, f)))
    return sd

base = load_all(STUDENT); models = [load_all(m) for m in MODELS]
out = {}
for k, w0 in base.items():
    if not w0.is_floating_point() or all(k not in m for m in models):
        out[k] = w0; continue
    taus = [(m[k].float() - w0.float()) for m in models]
    if METHOD.startswith('dare'):
        taus = [t * (torch.rand(t.shape, generator=g) >= DROP).float() / (1 - DROP) for t in taus]
    m_ = METHOD.replace('dare_', '')
    if m_ == 'avg':
        delta = sum(taus) / len(taus)
    elif m_ == 'ta':
        delta = LAM * sum(taus)
    elif m_ == 'ties':
        T = torch.stack(taus)
        if not METHOD.startswith('dare'):  # trim (DARE already sparsifies)
            kk = max(1, int(DENSITY * T[0].numel()))
            for i in range(T.shape[0]):
                th = T[i].abs().flatten().kthvalue(T[i].numel() - kk + 1).values
                T[i] = T[i] * (T[i].abs() >= th)
        sign = torch.sign(T.sum(0))
        agree = (torch.sign(T) == sign) & (T != 0)
        delta = LAM * (T * agree).sum(0) / agree.sum(0).clamp(min=1)
    else:
        raise ValueError(METHOD)
    out[k] = (w0.float() + delta).to(w0.dtype)
os.makedirs(OUT, exist_ok=True)
save_file(out, os.path.join(OUT, 'model.safetensors'), metadata={'format': 'pt'})
for f in os.listdir(STUDENT):
    if not f.endswith('.safetensors') and f != 'model.safetensors.index.json':
        shutil.copy(os.path.join(STUDENT, f), os.path.join(OUT, f))
json.dump(dict(method=METHOD, models=MODELS, lam=LAM, density=DENSITY, drop=DROP), open(os.path.join(OUT, 'merge.json'), 'w'), indent=1)
print('[merge]', METHOD, 'lam', LAM, 'models', len(MODELS), '->', OUT, flush=True)
