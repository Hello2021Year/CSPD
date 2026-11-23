#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "usage: bash scripts/run_pipeline.sh CONFIG [RUN_DIR]" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="$1"
run_dir="${2:-}"
python_bin="${SOFT_PROMPT_PYTHON:-}"
train_python_bin="${SOFT_PROMPT_TRAIN_PYTHON:-${python_bin}}"
cache_dir="${HF_HOME:-${repo_dir}/.cache_runtime/huggingface}"

if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
  echo "set SOFT_PROMPT_PYTHON to the runtime's executable Python" >&2
  exit 2
fi
if [[ -z "${train_python_bin}" || ! -x "${train_python_bin}" ]]; then
  echo "SOFT_PROMPT_TRAIN_PYTHON is not an executable Python" >&2
  exit 2
fi
# Canonicalize the containing directory, but keep the virtualenv's Python
# launcher path. Resolving the launcher symlink itself would bypass the
# virtualenv and run uv's base interpreter without the runtime dependencies.
runtime_bin="$(cd "$(dirname "${python_bin}")" && pwd -P)"
python_bin="${runtime_bin}/$(basename "${python_bin}")"
train_runtime_bin="$(cd "$(dirname "${train_python_bin}")" && pwd -P)"
train_python_bin="${train_runtime_bin}/$(basename "${train_python_bin}")"
if [[ ! -f "${config}" ]]; then
  echo "config does not exist: ${config}" >&2
  exit 2
fi
config="$(realpath "${config}")"

sglang_tp_size="${SOFT_PROMPT_SGLANG_TP_SIZE:-1}"
sglang_dp_size="${SOFT_PROMPT_SGLANG_DP_SIZE:-1}"
sglang_batch_size="${SOFT_PROMPT_SGLANG_BATCH_SIZE:-8192}"
sglang_log_level="${SOFT_PROMPT_SGLANG_LOG_LEVEL:-error}"
train_workers="${SOFT_PROMPT_TRAIN_WORKERS:-1}"
train_placement="${SOFT_PROMPT_TRAIN_MODEL_PLACEMENT:-replicated}"
stop_after_baselines="${SOFT_PROMPT_PIPELINE_STOP_AFTER_BASELINES:-0}"

for value in "${sglang_tp_size}" "${sglang_dp_size}" "${train_workers}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TP size, DP size, and train workers must be positive integers" >&2
    exit 2
  fi
done
if [[ "${train_placement}" != "replicated" && "${train_placement}" != "balanced" ]]; then
  echo "SOFT_PROMPT_TRAIN_MODEL_PLACEMENT must be replicated or balanced" >&2
  exit 2
fi
if [[ "${train_placement}" == "balanced" && "${train_workers}" != "1" ]]; then
  echo "balanced placement requires SOFT_PROMPT_TRAIN_WORKERS=1" >&2
  exit 2
fi
if [[ "${stop_after_baselines}" != "0" && "${stop_after_baselines}" != "1" ]]; then
  echo "SOFT_PROMPT_PIPELINE_STOP_AFTER_BASELINES must be 0 or 1" >&2
  exit 2
fi

cd "${repo_dir}"
export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${cache_dir}"
export TOKENIZERS_PARALLELISM=false

log_progress() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

if [[ -z "${run_dir}" ]]; then
  run_name="$(
    "${python_bin}" -c \
      "import sys; from soft_prompt_distillation.config import load_config; print(load_config(sys.argv[1]).name)" \
      "${config}"
  )"
  if [[ ! "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "config name is not a safe run-directory name: ${run_name}" >&2
    exit 2
  fi
  run_dir="${repo_dir}/experiments/runs/${run_name}"
fi

echo "resolved_run_dir=${run_dir}"
mkdir -p "${run_dir}" "${cache_dir}"

config_snapshot="${run_dir}/config.yaml"
if [[ -e "${config_snapshot}" ]]; then
  if ! cmp -s -- "${config}" "${config_snapshot}"; then
    echo "run directory contains a different config snapshot: ${config_snapshot}" >&2
    echo "use a new run directory for the changed config" >&2
    exit 2
  fi
else
  temporary_config_snapshot="${config_snapshot}.tmp.$$"
  cp -- "${config}" "${temporary_config_snapshot}"
  mv -- "${temporary_config_snapshot}" "${config_snapshot}"
fi
echo "config_snapshot=${config_snapshot}"
# All stages consume the immutable run-local copy. This prevents an edit to the
# shared checkout after launch from changing later stages of an active run.
config="${config_snapshot}"

model_args=()
if [[ -n "${SOFT_PROMPT_STUDENT_MODEL_PATH:-}" ]]; then
  model_args+=(--student-model-path "${SOFT_PROMPT_STUDENT_MODEL_PATH}")
fi
if [[ -n "${SOFT_PROMPT_TEACHER_MODEL_PATH:-}" ]]; then
  model_args+=(--teacher-model-path "${SOFT_PROMPT_TEACHER_MODEL_PATH}")
fi
sglang_args=(
  --sglang-tp-size "${sglang_tp_size}"
  --sglang-dp-size "${sglang_dp_size}"
  --sglang-batch-size "${sglang_batch_size}"
  --sglang-log-level "${sglang_log_level}"
)

stage_complete() {
  local directory="$1"
  shift
  if [[ ! -e "${directory}" ]]; then
    return 1
  fi
  local required
  for required in "$@"; do
    if [[ ! -f "${directory}/${required}" ]]; then
      if [[ "${directory}" == "${student_eval_dir}" || "${directory}" == "${teacher_eval_dir}" ]]; then
        local evidence_suffix="${SLURM_JOB_ID:-$(date -u '+%Y%m%dT%H%M%SZ')}"
        local evidence_directory="${directory}_incomplete_${evidence_suffix}"
        if [[ -e "${evidence_directory}" ]]; then
          evidence_directory="${evidence_directory}_$$"
        fi
        echo "archive incomplete evaluation stage: ${directory} -> ${evidence_directory}" >&2
        mv -- "${directory}" "${evidence_directory}"
        return 1
      fi
      if [[ "${directory}" == "${train_dir}" && -f "${directory}/training_state.pt" ]]; then
        echo "resume incomplete training stage: ${directory}"
        return 1
      fi
      echo "incomplete stage directory: ${directory}" >&2
      echo "move it aside after inspection, then rerun this script" >&2
      exit 2
    fi
  done
  echo "skip complete stage: ${directory}"
  return 0
}

sample_dir="${run_dir}/sample"
train_dir="${run_dir}/train"
student_eval_dir="${run_dir}/evaluate_student"
teacher_eval_dir="${run_dir}/evaluate_teacher"
soft_prompt_eval_dir="${run_dir}/evaluate_student_soft_prompt"

if ! stage_complete "${student_eval_dir}" results.jsonl summary.json manifest.json; then
  log_progress "stage=evaluate_student event=start run_dir=${run_dir}"
  "${python_bin}" -m soft_prompt_distillation evaluate "${config}" \
    --target student --output "${student_eval_dir}" --cache-dir "${cache_dir}" \
    ${model_args[@]+"${model_args[@]}"} "${sglang_args[@]}"
  log_progress "stage=evaluate_student event=complete run_dir=${run_dir}"
fi

if ! stage_complete "${teacher_eval_dir}" results.jsonl summary.json manifest.json; then
  log_progress "stage=evaluate_teacher event=start run_dir=${run_dir}"
  "${python_bin}" -m soft_prompt_distillation evaluate "${config}" \
    --target teacher --output "${teacher_eval_dir}" --cache-dir "${cache_dir}" \
    ${model_args[@]+"${model_args[@]}"} "${sglang_args[@]}"
  log_progress "stage=evaluate_teacher event=complete run_dir=${run_dir}"
fi

if [[ "${stop_after_baselines}" == "1" ]]; then
  log_progress "stage=pipeline event=paused_after_baselines run_dir=${run_dir}"
  printf 'baseline reports:\n  %s\n  %s\n' \
    "${student_eval_dir}/summary.json" \
    "${teacher_eval_dir}/summary.json"
  exit 0
fi

sample_complete=false
if [[ -e "${sample_dir}" ]]; then
  if [[ -f "${sample_dir}/teacher_trajectories.jsonl" && -f "${sample_dir}/manifest.json" ]]; then
    echo "skip complete stage: ${sample_dir}"
    sample_complete=true
  else
    echo "resume incomplete sample stage: ${sample_dir}"
  fi
fi
if [[ "${sample_complete}" != true ]]; then
  log_progress "stage=sample event=start run_dir=${run_dir}"
  "${python_bin}" -m soft_prompt_distillation sample "${config}" \
    --output "${sample_dir}" --cache-dir "${cache_dir}" \
    ${model_args[@]+"${model_args[@]}"} "${sglang_args[@]}"
  log_progress "stage=sample event=complete run_dir=${run_dir}"
fi

if ! stage_complete "${train_dir}" soft_prompt.safetensors training_metrics.jsonl manifest.json; then
  log_progress "stage=train event=start run_dir=${run_dir}"
  train_args=(
    -m soft_prompt_distillation train "${config}"
    --snapshot "${sample_dir}/teacher_trajectories.jsonl"
    --output "${train_dir}"
    --cache-dir "${cache_dir}"
    --train-model-placement "${train_placement}"
    ${model_args[@]+"${model_args[@]}"}
  )
  if [[ -f "${train_dir}/training_state.pt" ]]; then
    echo "resume incomplete training stage: ${train_dir}"
    train_args+=(--resume)
  fi
  if [[ "${train_placement}" == "replicated" && "${train_workers}" -gt 1 ]]; then
    "${train_python_bin}" -m torch.distributed.run --standalone \
      --nproc_per_node="${train_workers}" "${train_args[@]}"
  else
    "${train_python_bin}" "${train_args[@]}"
  fi
  log_progress "stage=train event=complete run_dir=${run_dir}"
fi

soft_prompt_temporary_args=()
soft_prompt_summary="${soft_prompt_eval_dir}/checkpoint_summary.json"
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  soft_prompt_temporary_args+=(--temporary-dir "${SLURM_TMPDIR}")
fi
log_progress "stage=evaluate_student_soft_prompt event=start run_dir=${run_dir}"
"${python_bin}" -m soft_prompt_distillation evaluate "${config}" \
  --target soft-prompt --train-dir "${train_dir}" --checkpoint all \
  --output "${soft_prompt_eval_dir}" --cache-dir "${cache_dir}" \
  "${soft_prompt_temporary_args[@]}" \
  ${model_args[@]+"${model_args[@]}"} "${sglang_args[@]}"
log_progress "stage=evaluate_student_soft_prompt event=complete run_dir=${run_dir}"

log_progress "stage=pipeline event=complete run_dir=${run_dir}"
printf 'reports:\n  %s\n  %s\n  %s\n  %s\n  %s\n' \
  "${student_eval_dir}/summary.json" \
  "${teacher_eval_dir}/summary.json" \
  "${soft_prompt_summary}" \
  "${soft_prompt_eval_dir}/recovery_curve.svg" \
  "${soft_prompt_eval_dir}/recovery_curve_overall.svg"
