# M2RL Docker 验证记录

验证日期为 2026-11-24。评测源码的上游版本为 `2fa3eac`，环境配置来自 `abc80d5`。

| 检查 | 结果 |
| --- | --- |
| `bash -n environment/run_m2rl_docker.sh` | 通过 |
| `bash environment/run_m2rl_docker.sh --plan` | 66 项唯一任务，两个节点各 33 项 |
| 模拟 Docker 与 NVIDIA 命令执行启动器 | 两个模拟节点各两张 GPU。验证了任务分配、LCB 评分的断网与只读参数、从混合标准输出提取评分、单项失败后继续及重跑复用 |
| `environment/.venv/cpu/bin/python -m pytest` | `428 passed, 2 skipped` |
| 使用发布的 checkpoint 组合 `MCSI`、`ISCM` | SHA-256 校验通过，两种顺序均生成 `[1024, 2560]` 张量 |
| Python 3.11、Linux x86-64 glibc 2.34 依赖解析 | 生成环境解析出 222 个包，评分环境解析出 184 个包 |
| 两台目标机器的只读 SSH 探测 | 两台机器均在 SSH 握手时关闭连接 |
| 本机 Docker | Docker 客户端可用，无法连接 Docker daemon |

Docker 镜像构建、NVIDIA 容器运行时、两台服务器的实际 GPU 推理和评分需要在服务器上验证。本记录使用模拟任务来检查运行逻辑，记录中无真实 benchmark 分数。
