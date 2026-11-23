#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 REQUESTS_JSONL RESPONSES_JSONL" >&2
  exit 2
fi

request_path=$(realpath "$1")
response_path=$(realpath -m "$2")
io_dir=$(dirname "$request_path")
if [[ $(dirname "$response_path") != "$io_dir" ]]; then
  echo "request and response must share one temporary directory" >&2
  exit 2
fi
if [[ ! -f "$request_path" ]]; then
  echo "request file does not exist: $request_path" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(realpath "$script_dir/..")
image=${SOFT_PROMPT_CODE_VERIFIER_IMAGE:?set SOFT_PROMPT_CODE_VERIFIER_IMAGE to an immutable .sqsh or digest-pinned image}

if [[ -f "$image" ]]; then
  image=$(realpath "$image")
  image_identity="sha256:$(sha256sum "$image" | awk '{print $1}')"
elif [[ "$image" == *@sha256:* ]]; then
  image_identity=${image##*@}
else
  echo "verifier image must be a local immutable file or use an @sha256 digest" >&2
  exit 2
fi

export SOFT_PROMPT_VERIFIER_ISOLATION_BACKEND=slurm_pyxis
export SOFT_PROMPT_VERIFIER_IMAGE_IDENTITY="$image_identity"
export PYTHONPATH=/workspace/src

worker_args=(
  "/io/$(basename "$request_path")"
  "/io/$(basename "$response_path")"
)
payload_path="${io_dir}/payloads.jsonl"
if [[ -f "$payload_path" ]]; then
  worker_args+=("/io/payloads.jsonl")
fi

srun \
  --nodes=1 \
  --ntasks=1 \
  --cpu-bind=none \
  --cpus-per-task="${SOFT_PROMPT_CODE_VERIFIER_CPUS:-32}" \
  --mem="${SOFT_PROMPT_CODE_VERIFIER_MEMORY:-64G}" \
  --overlap \
  --container-image="$image" \
  --container-mounts="$repo_dir:/workspace:ro,$io_dir:/io:rw" \
  --container-workdir=/workspace \
  --container-readonly \
  --no-container-mount-home \
  --container-env=PYTHONPATH,SOFT_PROMPT_VERIFIER_ISOLATION_BACKEND,SOFT_PROMPT_VERIFIER_IMAGE_IDENTITY \
  python -m soft_prompt_distillation.code_verifier_worker \
    "${worker_args[@]}"
