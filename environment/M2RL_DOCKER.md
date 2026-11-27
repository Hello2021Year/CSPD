# M2RL Docker 批量评测

## 从试跑到对照论文

在有 NVIDIA GPU 和 Docker 的主机上，从仓库根目录执行。单机模式优先使用本地镜像 `qwen-jev:x2-3-jev`，可用 `CSPD_BASE` 显式指定镜像。以下例子沿用已有的 Hugging Face 缓存，按实际路径修改 `CSPD_HF_CACHE`。

```bash
export CSPD_OUT="$PWD/artifacts/paper-check"
export CSPD_HF_CACHE="$PWD/artifacts/docker-repro/hf"
export CSPD_HF_OFFLINE=1

CSPD_SUITE=smoke bash environment/run_m2rl_docker.sh --plan all
CSPD_SUITE=smoke bash environment/run_m2rl_docker.sh all
```

`smoke` 只跑 MCSI 的 AIME24，每题生成一次，用来检查模型、软提示、生成和评分能否串起来。正式评测使用论文的采样数。下列子集在单机模式中使用可用的空闲卡，每卡同一时间运行一项任务。

| `CSPD_SUITE` | 任务数 | 内容 |
| --- | ---: | --- |
| `smoke` | 1 | MCSI AIME24，K=1 |
| `mcsi` | 4 | Table 2 的 MCSI，K=64/8/16/8 |
| `table2` | 24 | Table 2 中可直接用发布产物跑的六个方法 |
| `table3` | 18 | Table 3 的 Student、MOPD、MCSI |
| `full` | 66 | 默认评测集，包含域内、域外和模块增删实验 |

先跑论文的 MCSI 主结果，再用校验脚本生成分数对照表。GPQA-Diamond 需要已授权的数据缓存，或者在联网模式下配置有数据集权限的 `HF_TOKEN`。LiveCodeBench 评分由启动器放进断网的隔离容器。

```bash
CSPD_SUITE=mcsi bash environment/run_m2rl_docker.sh all
commit=$(git rev-parse --short=12 HEAD)
python3 environment/verify_m2rl.py "$CSPD_OUT/$commit/mcsi/node-all" --table 2 --method MCSI
```

要验证其他已发布方法，可改用 `CSPD_SUITE=table2` 或 `CSPD_SUITE=table3`，将校验命令的目录和 `--table` 一起改成对应实验，并省略 `--method`。双机结果放在同一运行目录的 `node-0/` 和 `node-1/` 下时，向校验脚本传它们的父目录。

校验脚本检查题数、采样数和评分文件，输出实测分、论文分和差值。缺项、重复或计数不符时返回非零状态，且该方法不计算均分。上游 B200 复跑经验是单项约 ±1.5 分、四项均分约 ±0.5 分，可用于排查明显偏差。 [论文分数](../README.md#composing-experts-table-2) · [上游复现说明](../AGENTS.md#comparing-with-the-paper)

## 单机八卡（x2-4）

从 CSPD 仓库根目录执行。单机模式优先使用本地镜像 `qwen-jev:x2-3-jev`，该镜像不在本地时回退到 `docker.m.daocloud.io/lmsysorg/sglang:qwen38-27b-cu129`。可通过 `CSPD_BASE` 和 `CSPD_GEN_RUNTIME` 覆盖。默认完整评测要求八张卡各有至少 95% 空闲显存，每张卡独立处理一项评测。

```bash
cd /mnt_upfs/spl/gitlab/CSPD
nvidia-smi
bash environment/run_m2rl_docker.sh --plan all

export CSPD_OUT="$PWD/artifacts/x2-4-full"
export CSPD_HF_CACHE="$PWD/artifacts/docker-repro/hf"  # 有现成缓存时，改成实际包含 hub/ 的目录
export CSPD_HF_OFFLINE=1
mkdir -p "$CSPD_OUT"
nohup bash environment/run_m2rl_docker.sh all > "$CSPD_OUT/launcher.log" 2>&1 < /dev/null &
echo "PID=$!"
```

镜像里的 CUDA、Torch 和 SGLang 用于生成，评分和 BFCL 则在独立环境运行。构建上下文、日志和结果放在 `CSPD_OUT`，模型缓存放在 `CSPD_HF_CACHE`；镜像层仍由 Docker 管理。首次构建镜像仍需能访问依赖和公共 LCB 数据；`CSPD_HF_OFFLINE` 只控制运行时 Hugging Face 客户端。启动器会先跑测试并检查固定版本的 M2RL-SFT 可用性。`CSPD_HF_CACHE` 指向的是包含 `hub/models--Jackwang111--M2RL-SFT/` 的上一级 HF_HOME（保留完整的 Hub 缓存结构，包括 `blobs` 和 `snapshots`），不是模型目录。如果已有的只是空缓存，或者固定版本未缓存，默认不会下载模型。

如果确认机器可访问 Hugging Face、磁盘容量足够，且允许多模型下载，再设 `export CSPD_HF_OFFLINE=0` 后启动。启动器显式覆盖基础镜像的 `HF_HUB_OFFLINE` 等变量。GPQA 需要数据集授权：联网运行时在启动前交互式设置 `HF_TOKEN`（见下文），或复用已授权下载并按固定版本缓存的数据。缺失的 GPQA 或域外输入会使对应任务失败，其他任务继续。监控用 `tail -f "$CSPD_OUT/launcher.log"`，汇总位于 `$CSPD_OUT/<commit>/node-all/scores.tsv`。不要同时启动两个写入同一输出目录的实例；若换镜像或评测环境，使用新的输出目录。

## 双机分工

在两台 Linux GPU 主机上用同一个 Git 提交作为评测源码。默认 Docker 基础镜像是本机已有的 `lmsysorg/sglang:latest`。脚本在基础镜像上构建仓库指定的 SGLang 0.5.18、评分环境和 BFCL 环境。Docker 必须能访问 GPU。首次构建和运行需要能访问 PyPI、GitHub 和 Hugging Face，并预留模型与 Docker 镜像所需的磁盘空间。

```bash
git clone https://github.com/Hello2021Year/CSPD.git
cd CSPD
nvidia-smi
bash environment/run_m2rl_docker.sh --plan
```

两台主机分别配置已获准访问 `Idavidrein/gpqa` 的 Hugging Face token。可设置 `CSPD_OUT` 和 `CSPD_HF_CACHE` 指向容量更大的目录，默认使用仓库的 `artifacts/docker-repro/`。共享 HF 缓存需要两台机器都能访问该目录。 [GPQA 数据集](https://huggingface.co/datasets/Idavidrein/gpqa)

```bash
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
# 在第一台机器启动
nohup bash environment/run_m2rl_docker.sh 0 > "$HOME/cspd-node-0.log" 2>&1 < /dev/null &
unset HF_TOKEN
```

```bash
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
# 在第二台机器启动
nohup bash environment/run_m2rl_docker.sh 1 > "$HOME/cspd-node-1.log" 2>&1 < /dev/null &
unset HF_TOKEN
```

共 66 项任务，两个节点各 33 项。一个节点里的每张 GPU 同时执行一项任务。Table 2 覆盖 Student、Oracle、MOPD、单域 SP、MCSI、ISCM 及模块增删组合。Table 3 覆盖 Student、MOPD、MCSI 的全部六项域外评测。TA/TIES、LoRA、前缀消融、Table 1 和 serving 实验使用其他模型产物或环境。 [评测流程与数据版本](../experiments/m2rl/BENCHMARKS.md)

| 内容 | 默认位置 |
| --- | --- |
| 创建镜像与任务进度 | `~/cspd-node-0.log` 或 `~/cspd-node-1.log` |
| 逐项生成日志 | `artifacts/docker-repro/<commit>/node-<n>/results/<method>/<benchmark>/run.log` |
| 逐项评分日志 | 同目录下的 `grader.log` |
| 每台机器汇总 | `artifacts/docker-repro/<commit>/node-<n>/scores.tsv` |

同一目录重复启动时，脚本复用已完成的评分和条数完整的生成结果。LiveCodeBench 在断网、只读根文件系统的 CPU 容器里评分，挂载当前任务的回答文件。主日志、评分、模型回答、GPQA 输入和模型缓存应留在运行机器上。提交问题时可摘录去掉凭据和样本内容的错误摘要。
