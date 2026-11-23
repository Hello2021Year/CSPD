#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="${1:?Usage: run_b200.sh {base|sp|r4} {low|c2|high|c32}}"
group="${2:?Specify a concurrency group}"
case "$arm" in
    base|sp) kernel=triton ;;
    r4) kernel=csgmv ;;
    *) echo "Unknown arm: $arm" >&2; exit 2 ;;
esac
# Match the original request indices. The high-budget C=16 run precedes C=24
# in its source job, so its 9*16 request slots are skipped here.
case "$group" in
    low) kv=524288; levels=(1 4 8 16); offset=0; run=1500 ;;
    c2) kv=524288; levels=(2); offset=0; run=1590 ;;
    high) kv=1048576; levels=(24); offset=261; run=1554 ;;
    c32) kv=1048576; levels=(32); offset=477; run=1570 ;;
    *) echo "Unknown concurrency group: $group" >&2; exit 2 ;;
esac

: "${SLURM_JOB_ID:?Run inside a single-B200 Slurm allocation}"
export PYTHONPATH="$PWD/engine/python${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor"
python -c 'import torch; assert torch.cuda.device_count() == 1 and "B200" in torch.cuda.get_device_name()'

python scripts/run_one.py --arm "$arm" --kernel "$kernel" --development-tp-size 1 \
    --attention-backend triton --slots 32 --max-total-tokens "$kv" \
    --swa-full-tokens-ratio .5 --mem-fraction-static .85 --adapter-preparation registered \
    --registered-assets assets-registered-r4-long --input-assets assets-text-v2 \
    --text-input-tokens 20480 --output-tokens 512 --sp-concurrent-updates 1 --sp-prefill-graph 1 \
    --registered-concurrency "${levels[@]}" --warmup 0 --duration 0 --timeout 300 \
    --asset-offset "$offset" --run-index "$run" --out "results/$SLURM_JOB_ID/$group/$arm"
