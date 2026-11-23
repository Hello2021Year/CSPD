# Agent experiments (Table 1, SWE-bench Verified and Terminal-Bench 2.1)

Agent prompts are trained on complete teacher episodes with the same frozen-backbone objective. Multi-turn episodes are rendered with [`soft_prompt_distillation.multiturn`](../../src/soft_prompt_distillation/multiturn.py), which labels only assistant tokens ([tests](../../tests/test_multiturn.py), [notes](MULTITURN_SFT_ENGINEERING.md)).

| Setting | Code here | Evaluation harness |
| --- | --- | --- |
| Nemotron-Cascade-8B (SWE-bench Verified, avg@4) | `nemotron-swe/`: `prepare_prompts.py` (8,192 Nemotron-Cascade-RL-SWE issues), `sample_teacher.py`, `train_sp.py`, `export_sp.py` | Agentless-style localization and repair with the teacher's bug locations |
| SkyRL-Agent-7B (SWE-bench Verified) | `multiturn/train_sp.py skyrl` | OpenHands CodeActAgent |
| TMax-9B / 27B (Terminal-Bench 2.1, avg@3) | `multiturn/train_sp.py tmax` | Harbor |
| DeepSWE, Lego-RL | not included | R2E-Gym; Harbor with OpenHands SDK |

The scripts read their inputs and outputs from environment variables documented at the top of each file and use the library in `src/`. The evaluation harnesses are the official ones of each benchmark and are not vendored here.
