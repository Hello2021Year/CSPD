# What does a soft prompt learn? (Figure 6, Appendix G)

The analysis uses the GPT-OSS-Swallow-20B math prompt (student `GPT-OSS-Swallow-20B-SFT-v0.1`, teacher `GPT-OSS-Swallow-20B-RL-v0.1`). The prompt checkpoints (updates 0–226) are in [ucsbai/cspd](https://huggingface.co/datasets/ucsbai/cspd/tree/main/trajectories) under `trajectories/math-gptoss/`; the other five pairs used by `plot_cross_model.py` are next to it. Pass a pair folder as `--run-dir` to `nn_token_analysis.py`. All scripts are in `scripts/interp/` and run from the repository root.

| Analysis | Scripts | Output |
| --- | --- | --- |
| Traces on held-out math problems (teacher, student, student+SP) | `generate_traces.py prepare / build-sp / generate` | sampled traces |
| Share of the teacher–student KL gap closed, token-level gains | `teacher_forced_scoring.py`, `analyze_token_gains.py`, `plot_token_gains.py` | Figure 6a, Figure 11 |
| Natural-language instructions and reasoning effort in place of the prompt | `prompt_variants.py`, `compare_prompt_variants.py`, `plot_prompt_variants.py` | Figure 12a |
| Attention pathways: which tokens may read the prompt | `sp_pathway.py`, `analyze_sp_pathway.py`, `plot_sp_pathway.py`; `check_sp_pathway.py` tests the knockouts on a tiny model | Figure 6c, Figure 12b |
| Embedding geometry and nearest vocabulary tokens | `nn_token_analysis.py`, `plot_nn_figures.py`, `plot_cross_model.py` | Appendix G |

A typical run:

```bash
python scripts/interp/generate_traces.py prepare --config configs/ablations/gpt-oss-prompt-length-256.yaml --output runs/interp/problems.json
python scripts/interp/generate_traces.py build-sp --student-dir $STUDENT --checkpoint $PROMPT/state.pt --output runs/interp/student-sp
python scripts/interp/generate_traces.py generate --config configs/ablations/gpt-oss-prompt-length-256.yaml --source teacher \
    --model-dir $TEACHER --teacher-dir $TEACHER --student-dir $STUDENT --problems runs/interp/problems.json \
    --output runs/interp/traces-teacher.jsonl     # repeat with --source student and student_sp
python scripts/interp/teacher_forced_scoring.py --traces runs/interp/traces-*.jsonl \
    --teacher-dir $TEACHER --student-dir $STUDENT --checkpoints u0=... u180=... --output runs/interp/scores
python scripts/interp/analyze_token_gains.py --scores runs/interp/scores/*.jsonl --tokenizer-dir $STUDENT --output-dir runs/interp/token-gains
```

Each script documents its options (`--help`). The teacher-forced and pathway scripts need two GPUs with at least 80 GB each. The [figure scripts](../paper-figures/README.md) redraw the published panels from the reported numbers on CPU; `figures/` holds the full-resolution analysis plots.
