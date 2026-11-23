# Instructions for coding agents

Use this file when reproducing results from this repository, *Composable Soft-Prompt Distillation of RL Experts* ([arXiv:2611.27376](https://arxiv.org/abs/2611.27376)). The [README](README.md) has the paper's numbers and the full command reference.

## Ground rules

- **Do not change protocol code.** Never edit graders, decoding parameters, prompts or model revisions to move a score. If a script fails, fix the environment, or report the error and the fix you propose.
- **Sandbox executed code.** LiveCodeBench scoring and the code teacher reward (`experiments/m2rl/teacher/rewards.py`) run model-written programs. Run them inside a container or sandbox, never on a machine holding credentials.
- **Ask before you download.** Model weights and benchmark data come from Hugging Face. GPQA (`Idavidrein/gpqa`) is gated: the user must accept its terms and provide `HF_TOKEN`. Ask before starting downloads larger than about 50 GB.
- **Check the GPUs first.** Run `nvidia-smi`, then plan for one GPU per evaluation job. Training (`train_sp.py`) uses 4 GPUs with `torchrun`. The release was tested on NVIDIA B200 (CUDA 12.8).
- **Keep outputs out of Git.** Write results under `runs/` and `artifacts/`.

## Environments

| Purpose | Setup | Used by |
| --- | --- | --- |
| Generation (SGLang 0.5.18) | `bash scripts/setup_blackwell_runtime.sh .venv-sgl` | `teacher/sample_teacher.py`, `eval/generate.py`, `ood/bfcl/run.py` |
| Training and scoring (vLLM 0.11) | `python3.11 -m venv .venv-train && .venv-train/bin/pip install -r experiments/m2rl/requirements.txt -e '.[data,test]'` | `train_sp.py`, `pack_data.py`, `eval/score.py`, `ood/score.py`, `baselines/` |
| CPU tests | either environment | `python -m pytest` (2 tests are skipped by design) |

Run `python -m pytest` once after setup. If it fails, report the failure and stop.

## Tasks

Paths below are relative to the repository root. Every scorer prints one JSON line containing `"score"`.

**A. Table 2, CSPD (MCSI), from the released prompts.** This is the main check: about 4 GPU-jobs, a few hours each on one B200.

```bash
python experiments/m2rl/compose.py --paper-order MCSI --output artifacts/MCSI.safetensors
python experiments/m2rl/eval/build_gpqa.py                       # needs HF_TOKEN with GPQA access
for bench in aime24 lcb_v5 gpqa_diamond ifbench; do  # run the four benchmarks as parallel GPU jobs if possible
  python experiments/m2rl/eval/generate.py --bench $bench --prompt artifacts/MCSI.safetensors --output runs/MCSI/$bench.jsonl
  python experiments/m2rl/eval/score.py --bench $bench --responses runs/MCSI/$bench.jsonl
done
```

Run `generate.py` in the SGLang environment and `score.py` in the training environment. The paper reports AIME24 80.47, LCB v5 64.67, GPQA-D 57.77 and IFBench 68.42, average 67.83.

Variants:
- **Reverse order:** `--paper-order ISCM`; the paper reports 79.90 / 65.72 / 57.92 / 67.33, average 67.72.
- **Student:** omit `--prompt`; the paper reports 55.94 / 58.61 / 44.38 / 42.71.
- **Continual addition and leave-one-out:** `--paper-order I`, `SI`, `CSI`, or drop one letter.

**B. Retrain one domain prompt from the released teacher data.** 4 GPUs.

```bash
hf download ucsbai/cspd --repo-type dataset --include 'data/*' --local-dir .
torchrun --nproc-per-node 4 experiments/m2rl/train_sp.py --data data/m2rl/math --output runs/m2rl/math
python experiments/m2rl/compose.py --module M=runs/m2rl/math/checkpoints/update-0200/state.pt --output artifacts/M.safetensors
```

Evaluate `artifacts/M.safetensors` as in task A. Domains are `math`, `code`, `science` and `if`, with module letters M, C, S and I. To compose several prompts, pass `--module` once per prompt in the concatenation order. The released prompts are in the same dataset (`--include 'prompts/m2rl/*'` gives `prompts/m2rl/{M,C,S,I}.pt`), so a retrained prompt can be mixed with them, for example:

```bash
python experiments/m2rl/compose.py --module M=prompts/m2rl/M.pt --module C=prompts/m2rl/C.pt \
    --module S=runs/m2rl/science/checkpoints/update-0200/state.pt --module I=prompts/m2rl/I.pt \
    --output artifacts/MCSI-mine.safetensors
```

**C. Regenerate the teacher data.** One GPU per domain. Code grading must run in a sandbox.

```bash
python experiments/m2rl/teacher/build_questions.py --domain science --output runs/teacher/science/questions.jsonl
python experiments/m2rl/teacher/sample_teacher.py --domain science --questions runs/teacher/science/questions.jsonl --output runs/teacher/science
python experiments/m2rl/pack_data.py runs/teacher/science data/m2rl/science       # add --max-per-question 2 for code
```

The run resumes from `responses.jsonl`. For a quick check, add `--limit 48` first. Sampling is not seeded per request, so counts land near the released ones (math 11,508, code 7,968, science 12,642, IF 14,179), not exactly on them.

**D. Out of domain (Table 3), single expert (Table 1), serving, analysis.** Follow the matching README section. For Table 1, start with `bash scripts/run_pipeline.sh configs/smoke.yaml runs/smoke`.

## Comparing with the paper

- **Expected spread.** Sampling at temperature 0.6 is not bit-reproducible across GPUs or engine versions. On our B200 rerun, per-benchmark scores fell within about 1.5 points of the paper and the four-benchmark average within about 0.5. A larger gap usually points to an environment problem, such as a wrong revision, missing grader dependencies or truncated generations; investigate before concluding.
- **Report format.** Give a table with columns benchmark, your score, the paper's score and the difference. Add the commands you ran, the GPU type, the `sglang`/`vllm` versions, and every step you skipped or changed.
