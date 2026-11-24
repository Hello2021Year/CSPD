# M2RL Docker 批量评测

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
