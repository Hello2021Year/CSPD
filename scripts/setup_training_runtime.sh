#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 1 ]]; then
  echo "usage: bash scripts/setup_training_runtime.sh [ENV_DIR]" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required; install it with your site package manager or pipx" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${1:-${repo_dir}/.runtime-train}"

uv venv --python 3.12 "${env_dir}"
# Keep training separate from the SGLang runtime: SGLang currently resolves an
# older Torch/Triton pair whose compiler cannot target Blackwell FlexAttention.
uv pip install --python "${env_dir}/bin/python" \
  "torch==2.13.0" "transformers==4.57.1"
(
  cd "${repo_dir}"
  uv pip install --python "${env_dir}/bin/python" -e '.[train,data,test]' \
    "torch==2.13.0" "transformers==4.57.1"
)

"${env_dir}/bin/python" -c \
  'import torch, transformers; from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss; from torch.nn.attention.flex_attention import flex_attention; assert tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 9); print(f"training runtime ready: torch={torch.__version__} triton={__import__('"'"'triton'"'"').__version__} transformers={transformers.__version__} flex_attention=yes liger_fused_ce=yes")'

printf '\nUse this runtime with:\nexport SOFT_PROMPT_TRAIN_PYTHON=%q\n' \
  "${env_dir}/bin/python"
