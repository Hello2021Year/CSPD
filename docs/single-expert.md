# Single-expert distillation (Table 1)

Each config in `configs/math/` and `configs/code/` reproduces one row of Table 1. A config pins the student and teacher revisions, the training data, the chat template and the soft-prompt position, the sampling and training settings, and the evaluation benchmarks and sample counts.

```bash
bash scripts/run_pipeline.sh configs/math/eurus-2-7b.yaml runs/eurus-2-7b
```

`run_pipeline.sh` runs `soft-prompt-distill evaluate` on the student and teacher, `sample` (teacher responses, verified correct and capped per question), `train` (256-vector prompt, frozen backbone) and `evaluate` on the saved checkpoints. Student+SP in Table 1 is the best evaluated checkpoint, and recovery is 100 × (Student+SP − Student) / (Teacher − Student). For avg@K, average the K samples of each question before averaging questions.

| Row | Config | Notes |
| --- | --- | --- |
| MiMo-7B math | `math/mimo-7b.yaml` | DeepScaleR prompts as proxy data |
| GPT-OSS-Swallow-20B math | `math/gpt-oss-swallow-20b.yaml`, sampling `math/gpt-oss-swallow-20b-sampling.yaml` | correct-only arm of Table 4 (best at update 140); the same configs build the mixed arm |
| Qwen3-Swallow-8B / 30B-A3B | `math/qwen3-swallow-8b.yaml`, `math/qwen3-swallow-30b-a3b.yaml` | |
| OLMo-3-7B math / code | `math/olmo3-7b.yaml`, `code/olmo3-7b.yaml` | code on LiveCodeBench v3 |
| Nemotron-Cascade-8B math / code | `math/nemotron-cascade-8b.yaml`, `code/nemotron-cascade-8b.yaml` | code with Klear code data as proxy |
| Eurus-2-7B, INTELLECT-MATH | `math/eurus-2-7b.yaml`, `math/intellect-math-7b.yaml` | AIME24 + AMC + MATH + Olympiad |
| Poro-2-8B, Klear-Reasoner-8B | `math/poro-2-8b.yaml`, `math/klear-reasoner-8b.yaml`, `code/klear-reasoner-8b.yaml` | |
| MiMo-7B code | `code/mimo-7b.yaml` | AReaL code data as proxy |
| AReaL-boba-2 8B / 14B | `code/areal-boba2-8b.yaml`, `code/areal-boba2-14b.yaml` | LCB Oct 2024 – Feb 2025 |
| Mellum2-12B-A2.5B | `code/mellum2-12b-a2p5b.yaml` | LCB test6, Klear code data as proxy |

Ablations: `configs/ablations/gpt-oss-prompt-length-{2,8,32,128,256,512}.yaml` (prompt length) and `configs/ablations/gpt-oss-training-curve.yaml` (training curve). Code is graded by executing tests; see [CODE_VERIFICATION.md](CODE_VERIFICATION.md) for the sandboxed grader. GPT-OSS needs the Blackwell runtime from `scripts/setup_blackwell_runtime.sh` and a sink-aware attention backend.
