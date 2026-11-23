# Benchmark versions

Scores move with dataset revisions and grader versions, so every benchmark below is pinned.

| Benchmark | Questions × K | Source (revision) | Grader (revision) | Inputs in this repo |
| --- | --- | --- | --- | --- |
| AIME 2024 | 30 × 64 | `math-ai/aime24`, M2RL prompt (`Mosi-AI/M2RL@c94ce97`, `data/download_aime.py`) | M2RL `async_rm` + `deepscaler` reward (`M2RL@c94ce97`, hash-checked) | `eval/inputs/aime24` |
| LiveCodeBench v5 | 167 × 8 | `livecodebench/code_generation_lite@0fe84c3`, `release_v5` subset, official generic prompt | `LiveCodeBench/LiveCodeBench@28fef95` runner, 6 s per test | `eval/inputs/lcb_v5` (hidden tests located by byte offset and SHA-256) |
| GPQA-Diamond | 198 × 16 | `Idavidrein/gpqa@83022ce` (gated), M2RL prompt with the evaluated option order | M2RL `async_rm` + `gpqa` reward (`M2RL@c94ce97`) | `eval/inputs/gpqa_diamond` (hashes; rebuild with `eval/build_gpqa.py`) |
| IFBench | 300 × 8 | `allenai/IFBench_test@2e8a48d` | `allenai/IFBench@1c40f0c` `evaluation_lib`, strict prompt accuracy | `eval/inputs/ifbench` |
| TriviaQA | 2,000 × 1 | `mandarjoshi/trivia_qa@0f7faf3`, `rc.nocontext` validation | SQuAD-normalized exact match (`ood/score.py`) | built by `ood/build_inputs.py` |
| TruthfulQA MC1 | 817 × 4 | `truthfulqa/truthful_qa@741b827`, multiple_choice validation | letter extraction (`ood/score.py`) | built |
| XSTest | 450 × 4 | `Paul/XSTest@f600c99` | `Qwen/Qwen3Guard-Gen-8B@4505cb1` refusal judgment | built |
| BBEH (mini) | 460 × 1 | `BBEH/bbeh@08e07a8` | exact match, macro over 23 tasks | built |
| FinQA | 1,147 × 1 | `czyssrs/FinQA@0f16e28`, `dataset/test.json` | 1% relative tolerance | built |
| BFCL v3 | 4,041 × 1 | `ShishirPatil/gorilla@cd9429c` (`berkeley-function-call-leaderboard`), 15 categories in `ood/bfcl/cases.json` | official BFCL checker, `QwenFCHandler` (`Qwen/Qwen3-4B-FC`), runner `ood/bfcl/run.py`, aggregation `ood/bfcl/aggregate.py` | case ids |

Decoding for all generations: temperature 0.6, top-p 0.95, top-k 20, up to 32,768 new tokens, Qwen3 chat template with thinking enabled (`Jackwang111/M2RL-SFT@aa49d49` tokenizer). The in-domain question files reproduce the paper's evaluation token ids exactly; the prompts are inserted after the system header as in training.
