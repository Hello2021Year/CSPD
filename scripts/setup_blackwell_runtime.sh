#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 1 ]]; then
  echo "usage: bash scripts/setup_blackwell_runtime.sh [ENV_DIR]" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required; install it with your site package manager or pipx" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${1:-${repo_dir}/.runtime-blackwell}"

uv venv --python 3.11 "${env_dir}"
(
  cd "${repo_dir}"
  uv pip install --python "${env_dir}/bin/python" \
    --prerelease=allow \
    -e '.[data,test,train]' \
    "torch==2.13.0" \
    "transformers==5.12.1" \
    "sglang==0.5.18" \
    "sglang-kernel==0.4.6.post1" \
    nltk langdetect absl-py immutabledict pylatexenc sympy  # M2RL rewards (experiments/m2rl/teacher)
)

"${env_dir}/bin/python" "${repo_dir}/scripts/patch_sglang_runtime.py"

"${env_dir}/bin/python" -c \
  'from importlib.metadata import version; import sgl_kernel, sglang, torch, transformers, triton; from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss; assert tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 9); print(f"Blackwell runtime ready: torch={torch.__version__} triton={triton.__version__} transformers={transformers.__version__} sglang={sglang.__version__} sglang_kernel={version('"'"'sglang-kernel'"'"')} liger_fused_ce=yes")'

printf '\nUse this runtime with:\nexport SOFT_PROMPT_PYTHON=%q\nexport PATH=%q:$PATH\n' \
  "${env_dir}/bin/python" "${env_dir}/bin"
