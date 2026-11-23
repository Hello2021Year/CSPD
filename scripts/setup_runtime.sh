#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 1 ]]; then
  echo "usage: bash scripts/setup_runtime.sh [ENV_DIR]" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required; install it with your site package manager or pipx" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${1:-${repo_dir}/.runtime}"

uv venv --python 3.11 "${env_dir}"
(
  cd "${repo_dir}"
  uv pip install --python "${env_dir}/bin/python" \
    --prerelease=allow \
    -e '.[train,data,sglang,test]'
)
# FlashAttention builds against the already-installed torch/CUDA runtime, so
# it must be installed without an isolated build environment.
MAX_JOBS="${MAX_JOBS:-4}" uv pip install --python "${env_dir}/bin/python" \
  --no-build-isolation -e '.[flash_attention]'

if [[ ! -x "${env_dir}/bin/ninja" ]]; then
  echo "runtime is missing ninja: ${env_dir}/bin/ninja" >&2
  exit 1
fi
"${env_dir}/bin/ninja" --version

"${env_dir}/bin/python" -c \
  'import flash_attn.flash_attn_interface, sglang, torch; from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss; print(f"python runtime ready: torch={torch.__version__} sglang={sglang.__version__} flash_attention_2=yes liger_fused_ce=yes")'

printf '\nUse this runtime with:\nexport SOFT_PROMPT_PYTHON=%q\nexport PATH=%q:$PATH\n' \
  "${env_dir}/bin/python" "${env_dir}/bin"
