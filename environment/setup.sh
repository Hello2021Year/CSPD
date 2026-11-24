#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash environment/setup.sh [all|generation|training|cpu]

  all         Install the generation and training/scoring environments (default).
  generation  Install the paper's SGLang generation environment.
  training    Install the M2RL training/scoring and vLLM environment.
  cpu         Install the environment for CPU tests and prompt composition.
EOF
}

if [[ "$#" -gt 1 ]]; then
    usage >&2
    exit 2
fi

target="${1:-all}"
case "$target" in
    -h|--help) usage; exit 0 ;;
    all|generation|training|cpu) ;;
    *) usage >&2; exit 2 ;;
esac

if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' 'Install uv before running this setup script.' >&2
    exit 1
fi

if [[ "$target" != cpu && "$(uname -s)" != Linux ]]; then
    printf '%s\n' 'The GPU environments require Linux. Use the cpu target on this host.' >&2
    exit 1
fi

environment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$environment_dir")"
venv_root="$environment_dir/.venv"
cd "$repo_dir"
mkdir -p "$venv_root"

if [[ "$target" == all || "$target" == generation ]]; then
    bash scripts/setup_blackwell_runtime.sh "$venv_root/sglang"
fi

if [[ "$target" == all || "$target" == training ]]; then
    train_python="$venv_root/train/bin/python"
    uv venv --allow-existing --python 3.11 "$venv_root/train"
    uv pip install --python "$train_python" -r experiments/m2rl/requirements.txt -e '.[data,test]'
    uv pip check --python "$train_python"
fi

if [[ "$target" == cpu ]]; then
    cpu_python="$venv_root/cpu/bin/python"
    uv venv --allow-existing --python 3.11 "$venv_root/cpu"
    uv pip install --python "$cpu_python" -e '.[data,test]' \
        'torch==2.8.0' 'transformers==4.57.1' 'safetensors>=0.5' matplotlib
    "$cpu_python" -c 'import matplotlib, pyarrow, pytest, safetensors, torch, transformers; print(f"CPU runtime ready: torch={torch.__version__} transformers={transformers.__version__}")'
fi

printf 'Environments are under %s\n' "$venv_root"
