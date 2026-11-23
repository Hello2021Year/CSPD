# Multi-expert composition on M2RL (Table 2, Figures 3–4, Table 3)

The frozen student is [M2RL-SFT](https://huggingface.co/Jackwang111/M2RL-SFT) (Qwen3-4B, revision `aa49d49a`). The teachers are the M2RL RL experts for math (M), code (C), science (S) and instruction following (I). Each domain prompt is 256 × 2,560 and sits right after the system header; an empty system turn is added when the input has none.

| Model | Role | Revision |
| --- | --- | --- |
| [M2RL-SFT](https://huggingface.co/Jackwang111/M2RL-SFT) | frozen student | `aa49d49a86b15ac920cf356c359f299edd0c1eb3` |
| [M2RL-RL_Math](https://huggingface.co/Jackwang111/M2RL-RL_Math) | math teacher (M) | `56da007b8bd471bed79077d2c3f26f48a1e595dd` |
| [M2RL-RL_Coding](https://huggingface.co/Jackwang111/M2RL-RL_Coding) | code teacher (C) | `22df4d65195cd0b2b47f1e46c0602452272a00d4` |
| [M2RL-RL_Science](https://huggingface.co/Jackwang111/M2RL-RL_Science) | science teacher (S) | `7f8747ba81e10c35343de90fd358be9867897cff` |
| [M2RL-RL_IF](https://huggingface.co/Jackwang111/M2RL-RL_IF) | instruction-following teacher (I) | `9b14959ec7bd0d4c4c97f2bee06eb453ca658f94` |
| [M2RL-MT_OPD](https://huggingface.co/Jackwang111/M2RL-MT_OPD) | MOPD baseline | `91004dd53f205ffa70ac2aa1a9a58ca654891233` |

| Step | Script |
| --- | --- |
| Build the training questions from the RL blend | `teacher/build_questions.py` (question pools in `teacher/pools/`) |
| Sample and verify teacher responses | `teacher/sample_teacher.py` (M2RL training rewards in `teacher/rewards.py`) |
| Pack verified teacher responses | `pack_data.py` |
| Train one domain prompt | `train_sp.py` (prefix-robust training, Muon, cosine 0.03 → 0.003, 200 updates, batch 64) |
| Compose prompts in an order | `compose.py` (`--paper-order MCSI` downloads the released prompts) |
| Generate and score AIME24 / LCB v5 / GPQA-Diamond / IFBench | `eval/generate.py`, `eval/score.py`, `eval/build_gpqa.py` |
| Out-of-domain benchmarks | `ood/build_inputs.py`, `eval/generate.py --inputs ood/inputs`, `ood/score.py` |
| Baselines | `baselines/merge_models.py` (TA, TIES), `baselines/lora_distill.py` |

Runtime versions: generation uses SGLang 0.5.18 (`scripts/setup_blackwell_runtime.sh`); training, baselines and scoring use [requirements.txt](requirements.txt). Commands are in the [top-level README](../../README.md#multi-expert-composition-table-2-figure-3).

**Training recipe.** `train_sp.py` reproduces the released prompts bit for bit on the same data (checked against the original training code on CPU). Variants used in the paper:

- without prefix-robust training (Table 2 and Table 5 use AdamW for this row): `--no-prefix-robust --optimizer adamw`
- AdamW instead of Muon (Table 5): `--optimizer adamw` (cosine 0.01 → 0.001)
- progressive training (Figure 4): `--frozen-prompts I.pt` for S, then `--frozen-prompts S.pt I.pt` for C, and so on; the new prompt is placed in front of the frozen ones.

**Teacher data.** Each expert is distilled on the questions of its own RL training data, from `nvidia/Nemotron-3-Nano-RL-Training-Blend@74d18444` with M2RL's prompt format.
- **Question pools.** `teacher/pools/` lists the paper's pools: math 18,205, code 10,661, science 19,636 and IF 16,569 questions. They keep the first copy of each question, remove overlaps with the evaluation benchmarks and drop prompts over 2,048 tokens.
- **Sampling.** The teacher samples at temperature 1.0 with up to 32,768 new tokens. A response is kept only if the M2RL training reward is 1 and the response ends with a closed thinking block. Math, science and IF keep one response from at most two samples per question; code keeps up to two distinct responses from at most four.
- **Determinism.** Sampling uses the engine seed only, so a rerun yields different responses with the same protocol. The released packed data (`ucsbai/cspd`) are the paper's exact training set.

**Evaluation.** `eval/generate.py` appends the composed prompt to the embedding table as new token rows (`eval/derived_model.py`) and inserts their ids after the system header, so any engine that accepts token ids runs the composition. Generated tokens are restricted to the tokenizer vocabulary. Decoding: temperature 0.6, top-p 0.95, top-k 20, 32,768 new tokens; K = 64/8/16/8. The benchmark questions in `eval/inputs/` are the exact token-for-token inputs used in the paper; GPQA is rebuilt locally from the gated dataset because its questions should not be redistributed. Scoring uses the M2RL reward functions (AIME, GPQA), the official IFBench library (strict prompt accuracy) and the official LiveCodeBench runner.

**Out of domain.** `ood/build_inputs.py` builds TriviaQA (2,000 closed-book questions), TruthfulQA MC1 (817, K = 4), XSTest (450, K = 4, judged by Qwen3Guard-Gen-8B), the BBEH mini set and FinQA (1,147) from the public sources. BFCL v3 (15 categories, 4,041 cases in `ood/bfcl/cases.json`) runs with `ood/bfcl/run.py`: each case uses BFCL's official `QwenFCHandler` and checker (gorilla commit `cd9429c`) in a worker process, while generation goes through SGLang with the soft prompt inserted after the system header; `ood/bfcl/aggregate.py` computes the weighted score.

**Versions.** Every dataset and grader revision is listed in [BENCHMARKS.md](BENCHMARKS.md).
