# M2RL Docker 批量评测

## 单机八卡（x2-4）

在 x2-4 从 CSPD 仓库根目录执行。单机模式默认使用本地镜像 `docker.m.daocloud.io/lmsysorg/sglang:qwen38-27b-cu129` 的生成环境；可通过 `CSPD_BASE` 和 `CSPD_GEN_RUNTIME` 覆盖。启动器要求八张卡各有至少 95% 空闲显存；每张卡独立处理一项评测，不使用八卡张量并行。

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
