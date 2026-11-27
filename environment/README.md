# CSPD 复现环境

在仓库根目录运行以下命令。脚本使用 `uv` 创建 Python 3.11 环境。GPU 实验使用 Linux 和 NVIDIA GPU，上游在 B200 和 CUDA 12.8 上测试。在执行 GPU 任务的节点上用 `nvidia-smi` 查看 GPU 和驱动。 [上游环境信息](../AGENTS.md#environments)

```bash
bash environment/setup.sh all
```

| 安装目标 | Python 路径 | 用途 |
| --- | --- | --- |
| `generation` | `environment/.venv/sglang/bin/python` | SGLang 0.5.18，teacher 采样和论文推理 |
| `training` | `environment/.venv/train/bin/python` | M2RL 训练、评分、vLLM 0.11 推理 |
| `cpu` | `environment/.venv/cpu/bin/python` | 单元测试及发布 prompt 的组合，支持 macOS 和 Linux |

生成环境调用 [上游安装脚本](../scripts/setup_blackwell_runtime.sh)，训练环境使用 [上游 requirements](../experiments/m2rl/requirements.txt)。安装下载 Python 依赖。模型、数据和 prompt 在下文相应命令执行时按需下载。虚拟环境和实验产物使用 Git 忽略的目录。

对上游生成依赖做 Linux x86-64 解析时，SGLang 0.5.18 的 wheel 要求 glibc 2.34 或更新版本，Torch 2.13.0 解析到 CUDA 13 系列运行库。按 CUDA 13 的基础兼容要求，检查 NVIDIA 驱动至少为 580 系列。 [NVIDIA CUDA 兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)

## CPU 检查

```bash
bash environment/setup.sh cpu
environment/.venv/cpu/bin/python -m pytest -ra
environment/.venv/cpu/bin/python experiments/m2rl/compose.py \
  --paper-order MCSI --output artifacts/repro/MCSI.safetensors
```

组合命令验证四份发布 checkpoint 的 SHA-256，并输出 `[1024, 2560]` 的组合张量。安装 GPU 环境后，也可以用 `environment/.venv/train/bin/python -m pytest -ra` 检查仓库。缺少双 CUDA 设备时，对应的模型分片测试会跳过。

## 用发布 prompt 复现 Table 2

已有 GPU Docker 环境时，可以用 [M2RL Docker 批量评测](M2RL_DOCKER.md) 做单卡链路试跑，或复现 Table 2 和 Table 3 的已发布模型与软提示。每项任务保存日志和评分，校验脚本将结果与论文对照。

完成 GPU 环境安装后，先验证 AIME24 的生成链路。以下试跑设为每题 1 次。

```bash
CSPD_GEN=environment/.venv/sglang/bin/python
CSPD_TRAIN=environment/.venv/train/bin/python

"$CSPD_TRAIN" experiments/m2rl/compose.py \
  --paper-order MCSI --output artifacts/repro/MCSI.safetensors
"$CSPD_GEN" experiments/m2rl/eval/generate.py \
  --bench aime24 --k 1 --prompt artifacts/repro/MCSI.safetensors \
  --output artifacts/repro/smoke/aime24.jsonl
```

完整评测使用脚本内的采样默认值。GPQA-Diamond 需要有权访问 Hugging Face 上的 `Idavidrein/gpqa`，在本机配置 HF 凭据后先执行下方的题目构建。LiveCodeBench 的评分会执行生成代码，使用上游要求的隔离环境运行该评分步骤。 [Benchmark 说明](../experiments/m2rl/BENCHMARKS.md) · [代码执行隔离说明](../docs/CODE_VERIFICATION.md)

```bash
"$CSPD_GEN" experiments/m2rl/eval/build_gpqa.py

for bench in aime24 lcb_v5 gpqa_diamond ifbench; do
  "$CSPD_GEN" experiments/m2rl/eval/generate.py \
    --bench "$bench" --prompt artifacts/repro/MCSI.safetensors \
    --output "artifacts/repro/MCSI/$bench.jsonl"
done

for bench in aime24 gpqa_diamond ifbench; do
  "$CSPD_TRAIN" experiments/m2rl/eval/score.py \
    --bench "$bench" --responses "artifacts/repro/MCSI/$bench.jsonl"
done
```

在隔离环境内进入仓库根目录，用该环境中的训练与评分 Python 执行 LiveCodeBench 评分。

```bash
environment/.venv/train/bin/python experiments/m2rl/eval/score.py \
  --bench lcb_v5 --responses artifacts/repro/MCSI/lcb_v5.jsonl
```

## 从发布的训练数据重训数学 prompt

使用 4 张 GPU。输出的 `update-0200/state.pt` 可以重新组合，也可以替换四个发布模块中的数学模块。 [上游 M2RL 训练说明](../experiments/m2rl/README.md)

```bash
environment/.venv/train/bin/hf download ucsbai/cspd --repo-type dataset \
  --include 'data/m2rl/math/*' --local-dir artifacts/repro

environment/.venv/train/bin/torchrun --nproc-per-node 4 \
  experiments/m2rl/train_sp.py --data artifacts/repro/data/m2rl/math \
  --output artifacts/repro/train/math

environment/.venv/train/bin/python experiments/m2rl/compose.py \
  --module M=artifacts/repro/train/math/checkpoints/update-0200/state.pt \
  --output artifacts/repro/M.safetensors
```
