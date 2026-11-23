#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: bash scripts/watch_and_evaluate_checkpoints.sh CONFIG RUN_DIR" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="$(realpath "$1")"
run_dir="$(realpath "$2")"
python_bin="${SOFT_PROMPT_PYTHON:-}"
cache_dir="${HF_HOME:-${repo_dir}/.cache_runtime/huggingface}"
poll_seconds="${SOFT_PROMPT_CHECKPOINT_POLL_SECONDS:-20}"

if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
  echo "set SOFT_PROMPT_PYTHON to the evaluation runtime's executable Python" >&2
  exit 2
fi
if [[ ! "${poll_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SOFT_PROMPT_CHECKPOINT_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

cd "${repo_dir}"
export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${cache_dir}"
export TOKENIZERS_PARALLELISM=false

train_dir="${run_dir}/train"
output_dir="${run_dir}/evaluate_student_soft_prompt"
temporary_args=()
model_args=()
sglang_args=(
  --sglang-tp-size "${SOFT_PROMPT_SGLANG_TP_SIZE:-1}"
  --sglang-dp-size "${SOFT_PROMPT_SGLANG_DP_SIZE:-1}"
  --sglang-batch-size "${SOFT_PROMPT_SGLANG_BATCH_SIZE:-8192}"
  --sglang-log-level "${SOFT_PROMPT_SGLANG_LOG_LEVEL:-error}"
)
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  temporary_args+=(--temporary-dir "${SLURM_TMPDIR}")
fi
if [[ -n "${SOFT_PROMPT_STUDENT_MODEL_PATH:-}" ]]; then
  model_args+=(--student-model-path "${SOFT_PROMPT_STUDENT_MODEL_PATH}")
fi
if [[ -n "${SOFT_PROMPT_TEACHER_MODEL_PATH:-}" ]]; then
  model_args+=(--teacher-model-path "${SOFT_PROMPT_TEACHER_MODEL_PATH}")
fi

evaluate_one() {
  local selector="$1"
  "${python_bin}" -m soft_prompt_distillation evaluate "${config}" \
    --target soft-prompt --train-dir "${train_dir}" --checkpoint "${selector}" \
    --output "${output_dir}" --cache-dir "${cache_dir}" \
    "${temporary_args[@]}" "${model_args[@]}" "${sglang_args[@]}"
}

shopt -s nullglob
while true; do
  for checkpoint in "${train_dir}"/checkpoints/update_*.safetensors; do
    stem="$(basename "${checkpoint}" .safetensors)"
    digits="${stem#update_}"
    selector="$((10#${digits}))"
    if [[ ! -f "${output_dir}/${stem}/manifest.json" ]]; then
      evaluate_one "${selector}"
    fi
  done
  if [[ -f "${train_dir}/soft_prompt.safetensors" && -f "${train_dir}/manifest.json" ]]; then
    # The training scheduler always publishes the last update in checkpoints/;
    # soft_prompt.safetensors is the same prompt kept as a compatibility alias.
    touch "${output_dir}/.watcher_complete"
    exit 0
  fi
  sleep "${poll_seconds}"
done
