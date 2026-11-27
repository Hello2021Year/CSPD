#!/usr/bin/env bash
set -euo pipefail
umask 022

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
m=experiments/m2rl
gen=/opt/gen/bin/python
score=/opt/score/bin/python
in_benches=(aime24 lcb_v5 gpqa_diamond ifbench)
ood_benches=(triviaqa truthfulqa xstest bbeh finqa bfcl)
in_methods=(MCSI student ISCM single_sp oracle mopd I SI CSI MSI MCI MCS)
prompt_orders=(MCSI ISCM I SI CSI MSI MCI MCS M C S)
suite=${CSPD_SUITE:-full}
run_in_domain=1
run_ood=1
case "$suite" in
    full) ;;
    table2)
        in_methods=(MCSI student ISCM single_sp oracle mopd)
        run_ood=0
        prompt_orders=(MCSI ISCM M C S I) ;;
    table3)
        in_methods=(MCSI student mopd)
        run_in_domain=0
        prompt_orders=(MCSI) ;;
    mcsi|smoke)
        in_methods=(MCSI)
        run_ood=0
        prompt_orders=(MCSI)
        if [[ "$suite" == smoke ]]; then in_benches=(aime24); fi ;;
    *) echo 'CSPD_SUITE must be full, table2, table3, mcsi, or smoke.' >&2; exit 2 ;;
esac
jobs=()
for method in "${in_methods[@]}"; do
    if (( run_in_domain )); then
        for bench in "${in_benches[@]}"; do jobs+=("$method/$bench"); done
    fi
    if (( run_ood )); then
        case "$method" in
            MCSI|student|mopd)
                for bench in "${ood_benches[@]}"; do jobs+=("$method/$bench"); done ;;
        esac
    fi
done

base_fallback=
case "${1:-}" in
    --plan)
        if [[ "${2:-}" == all && $# == 2 ]]; then
            for i in "${!jobs[@]}"; do printf 'GPU-slot-%s\t%s\n' "$((i % 8))" "${jobs[i]}"; done
        elif (( $# == 1 )); then
            for i in "${!jobs[@]}"; do printf 'node-%s\t%s\n' "$((i % 2))" "${jobs[i]}"; done
        else
            exit 2
        fi
        exit 0 ;;
    0|1)
        node=$1; nodes=2; offline_default=0
        base_default=lmsysorg/sglang:latest; runtime_default=blackwell ;;
    all)
        node=0; nodes=1; offline_default=1
        base_default=qwen-jev:x2-3-jev
        base_fallback=docker.m.daocloud.io/lmsysorg/sglang:qwen38-27b-cu129
        runtime_default=base ;;
    *) printf 'Usage: CSPD_SUITE={full|table2|table3|mcsi|smoke} bash environment/run_m2rl_docker.sh {0|1|all|--plan [all]}\n' >&2; exit 2 ;;
esac
if (( $# != 1 )); then exit 2; fi

base=${CSPD_BASE:-$base_default}
runtime=${CSPD_GEN_RUNTIME:-$runtime_default}
case "$runtime" in
    blackwell|base) ;;
    *) echo 'CSPD_GEN_RUNTIME must be blackwell or base.' >&2; exit 2 ;;
esac
offline=${CSPD_HF_OFFLINE:-$offline_default}
case "$offline" in
    0|1) ;;
    *) echo 'CSPD_HF_OFFLINE must be 0 or 1.' >&2; exit 2 ;;
esac
revision=$(git rev-parse HEAD)
commit=$(git rev-parse --short=12 "$revision")
image="cspd-m2rl-repro:$commit-$runtime"
root=${CSPD_OUT:-$PWD/artifacts/docker-repro}
mkdir -p "$root"
root=$(cd "$root" && pwd -P)
run_root="$root/$commit"
if [[ "$suite" != full ]]; then run_root="$run_root/$suite"; fi
out="$run_root/node-$1"
hf=${CSPD_HF_CACHE:-$root/hf}
mkdir -p "$out" "$hf"
hf=$(cd "$hf" && pwd -P)
gpu_rows=$(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits)
if [[ -z "$gpu_rows" ]]; then echo 'No NVIDIA GPUs detected.' >&2; exit 1; fi
gpus=()
while IFS=, read -r gpu total free; do
    if [[ ! "$gpu" =~ ^[0-9]+$ || ! "${total// /}" =~ ^[0-9]+$ || ! "${free// /}" =~ ^[0-9]+$ ]]; then
        echo "Unexpected nvidia-smi row: $gpu,$total,$free" >&2
        exit 1
    fi
    if (( nodes == 1 && free * 100 < total * 95 )); then
        echo "GPU $gpu has less than 95% free VRAM." >&2
        if [[ "$suite" == full ]]; then exit 1; fi
        continue
    fi
    gpus+=("$gpu")
done <<< "$gpu_rows"
if [[ -z "${gpus[0]:-}" ]]; then echo 'No eligible NVIDIA GPUs.' >&2; exit 1; fi
if [[ "$suite" == full ]] && (( nodes == 1 && ${#gpus[@]} != 8 )); then
    echo "Expected 8 GPUs in all mode, found ${#gpus[@]}." >&2
    exit 1
fi
nvidia-smi -L
if [[ -z "${CSPD_BASE:-}" && -n "$base_fallback" ]] &&
    ! docker image inspect "$base" >/dev/null 2>&1 &&
    docker image inspect "$base_fallback" >/dev/null 2>&1; then
    base=$base_fallback
fi
docker image inspect "$base" >/dev/null
printf 'Using Docker base image %s with %s runtime\n' "$base" "$runtime"

context=$(mktemp -d "$root/.docker-context.XXXXXX")
trap 'rm -rf "$context"' EXIT
git init --quiet --template= "$context"
git -C "$context" -c protocol.file.allow=always fetch --quiet --depth=1 --no-tags --no-write-fetch-head "$repo" "$revision"
git -C "$context" -c core.logAllRefUpdates=false checkout --quiet --detach "$revision"
cp containers/m2rl-repro.Dockerfile "$context/Dockerfile"
docker build --build-arg "CSPD_BASE=$base" --build-arg "CSPD_GEN_RUNTIME=$runtime" --tag "$image" "$context"

dock() {
    local gpu=$1
    shift
    local -a docker_args=(--rm --pull=never --init --shm-size 8g)
    if [[ "$gpu" != cpu ]]; then
        docker_args+=(--gpus "device=$gpu" -e CUDA_VISIBLE_DEVICES=0)
    else
        docker_args+=(-e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=)
    fi
    docker run "${docker_args[@]}" \
        --user "$(id -u):$(id -g)" -e HOME=/tmp/cspd-home -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub \
        -e "HF_HUB_OFFLINE=$offline" -e "TRANSFORMERS_OFFLINE=$offline" \
        -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
        -v "$out:/out" -v "$hf:/hf" --entrypoint /bin/bash "$image" \
        -c 'mkdir -p "$HOME"; exec "$@"' _ "$@"
}

dock "${gpus[0]}" "$gen" -c \
    'import torch, sglang, transformers; assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0), "CUDA", torch.version.cuda, "SGLang", sglang.__version__, "Transformers", transformers.__version__)'
if (( nodes == 1 )); then
    dock cpu "$score" -m pytest -p no:cacheprovider
    if ! dock cpu "$gen" -c '
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

model = Path(snapshot_download("Jackwang111/M2RL-SFT", revision="aa49d49a86b15ac920cf356c359f299edd0c1eb3"))
index = json.loads((model / "model.safetensors.index.json").read_text())
for shard in set(index["weight_map"].values()):
    if not (model / shard).is_file():
        raise FileNotFoundError(model / shard)
AutoTokenizer.from_pretrained(model)
print(f"M2RL-SFT ready: {model}")
'; then
        echo "M2RL-SFT preflight failed (CSPD_HF_CACHE=$hf, CSPD_HF_OFFLINE=$offline)." >&2
        echo 'Mount the complete pinned Hub cache, or explicitly enable online downloads with CSPD_HF_OFFLINE=0.' >&2
        exit 1
    fi
fi
dock cpu bash -c 'mkdir -p /out/in; cp -r experiments/m2rl/eval/inputs/. /out/in/'
gpqa="$out/in/gpqa_diamond/questions.jsonl"
if [[ "$suite" == full || "$suite" == table2 || "$suite" == mcsi ]] &&
    { [[ ! -s "$gpqa" ]] || [[ $(wc -l < "$gpqa") -ne 198 ]]; }; then
    if ! docker run --rm --pull=never --user "$(id -u):$(id -g)" \
        -e NVIDIA_VISIBLE_DEVICES=void -e HF_TOKEN -e HF_HOME=/hf -e HF_HUB_CACHE=/hf/hub \
        -e "HF_HUB_OFFLINE=$offline" -e "TRANSFORMERS_OFFLINE=$offline" -v "$out:/out" -v "$hf:/hf" \
        --entrypoint "$gen" "$image" "$m/eval/build_gpqa.py" --inputs /out/in; then
        echo 'GPQA preparation failed. Check the gated dataset permission and HF_TOKEN.' >&2
    fi
fi
unset HF_TOKEN

for order in "${prompt_orders[@]}"; do
    file="$out/prompts/$order.safetensors"
    if [[ -s "$file" ]]; then continue; fi
    dock cpu "$gen" "$m/compose.py" --paper-order "$order" --output "/out/prompts/$order.safetensors.part"
    mv "$file.part" "$file"
done
if (( run_ood )); then
    for bench in "${ood_benches[@]}"; do
        if [[ "$bench" == bfcl ]]; then continue; fi
        if [[ -s "$out/ood/$bench/questions.jsonl" && -s "$out/ood/$bench/references.jsonl" ]]; then continue; fi
        dock cpu "$gen" "$m/ood/build_inputs.py" --output /out/ood --benches "$bench" ||
            echo "Could not prepare $bench; continuing with the other benchmarks." >&2
    done
fi

score_lcb() {
    docker run --rm --pull=never --init --network none --read-only \
        --cap-drop ALL --security-opt no-new-privileges --pids-limit 512 \
        --tmpfs /tmp:rw,nosuid,nodev,mode=1777 --user 65534:65534 \
        -e NVIDIA_VISIBLE_DEVICES=void -e TMPDIR=/tmp -e HOME=/opt/lcb-cache -e HF_HOME=/opt/lcb-cache/hf \
        -e HF_HUB_CACHE=/opt/lcb-cache/hf/hub -e HF_HUB_OFFLINE=1 \
        -e PYTHONDONTWRITEBYTECODE=1 -v "$1:/responses.jsonl:ro" \
        --entrypoint "$score" "$image" "$m/eval/score.py" --bench lcb_v5 \
        --responses /responses.jsonl --workers 4 --output /tmp/sample-scores.jsonl
}

run_job() {
    local gpu=$1 job=$2 method=${2%/*} bench=${2#*/}
    local result="$out/results/$job" dir="/out/results/$job" inputs=/out/in grader="$m/eval/score.py"
    local -a model=(--model Jackwang111/M2RL-SFT --revision aa49d49a86b15ac920cf356c359f299edd0c1eb3)
    local scoring_gpu=cpu k
    if [[ -s "$result/score.json" ]]; then return 0; fi

    case "$method" in
        student) ;;
        mopd) model=(--model Jackwang111/M2RL-MT_OPD --revision 91004dd53f205ffa70ac2aa1a9a58ca654891233) ;;
        single_sp)
            case "$bench" in
                aime24) model=(--prompt /out/prompts/M.safetensors) ;;
                lcb_v5) model=(--prompt /out/prompts/C.safetensors) ;;
                gpqa_diamond) model=(--prompt /out/prompts/S.safetensors) ;;
                ifbench) model=(--prompt /out/prompts/I.safetensors) ;;
            esac ;;
        oracle)
            case "$bench" in
                aime24) model=(--model Jackwang111/M2RL-RL_Math --revision 56da007b8bd471bed79077d2c3f26f48a1e595dd) ;;
                lcb_v5) model=(--model Jackwang111/M2RL-RL_Coding --revision 22df4d65195cd0b2b47f1e46c0602452272a00d4) ;;
                gpqa_diamond) model=(--model Jackwang111/M2RL-RL_Science --revision 7f8747ba81e10c35343de90fd358be9867897cff) ;;
                ifbench) model=(--model Jackwang111/M2RL-RL_IF --revision 9b14959ec7bd0d4c4c97f2bee06eb453ca658f94) ;;
            esac ;;
        *) model=(--prompt "/out/prompts/$method.safetensors") ;;
    esac

    if [[ "$bench" == bfcl ]]; then
        dock "$gpu" "$gen" "$m/ood/bfcl/run.py" "${model[@]}" \
            --bfcl-python /opt/bfcl/bin/python --bfcl-root /opt/gorilla/berkeley-function-call-leaderboard \
            --output "$dir/responses.jsonl" || return 1
        dock cpu "$gen" "$m/ood/bfcl/aggregate.py" "$dir/responses.jsonl" > "$result/grader.log" || return 1
    else
        case "$bench" in
            aime24) k=64 ;;
            lcb_v5|ifbench) k=8 ;;
            gpqa_diamond) k=16 ;;
            *) inputs=/out/ood; grader="$m/ood/score.py"
               case "$bench" in truthfulqa|xstest) k=4 ;; *) k=1 ;; esac ;;
        esac
        if [[ "$suite" == smoke ]]; then k=1; fi
        local questions="$out${inputs#/out}/$bench/questions.jsonl"
        local references="$out${inputs#/out}/$bench/references.jsonl"
        if [[ ! -s "$questions" || ! -s "$references" ]]; then echo "Missing inputs for $bench"; return 1; fi
        local count expected responses="$result/responses.jsonl"
        count=$(wc -l < "$questions") || return 1
        if (( count != $(wc -l < "$references") )); then echo "Incomplete inputs for $bench"; return 1; fi
        expected=$((count * k))
        if [[ ! -s "$responses" ]] || (( $(wc -l < "$responses") != expected )); then
            dock "$gpu" "$gen" "$m/eval/generate.py" --bench "$bench" --inputs "$inputs" \
                "${model[@]}" --k "$k" --output "$dir/responses.jsonl.part" || return 1
            if (( $(wc -l < "$responses.part") != expected )); then echo 'Incomplete generation'; return 1; fi
            mv "$responses.part" "$responses" || return 1
        fi
        if [[ "$bench" == lcb_v5 ]]; then
            score_lcb "$responses" > "$result/grader.log" || return 1
        else
            if [[ "$bench" == xstest ]]; then scoring_gpu=$gpu; fi
            local -a grader_args=("$grader" --bench "$bench" --inputs "$inputs" --responses "$dir/responses.jsonl")
            if [[ "$inputs" == /out/in ]]; then grader_args+=(--workers 4); fi
            dock "$scoring_gpu" "$score" "${grader_args[@]}" > "$result/grader.log" || return 1
        fi
    fi

    python3 - "$result/grader.log" "$result/score.json.part" <<'PY' || return 1
import json
from pathlib import Path
import sys

for line in reversed(Path(sys.argv[1]).read_text().splitlines()):
    try:
        score = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(score, dict) and "score" in score:
        if score.get("missing", 0):
            raise SystemExit("BFCL has missing cases")
        Path(sys.argv[2]).write_text(json.dumps(score) + "\n")
        break
else:
    raise SystemExit("Grader did not emit a score")
PY
    mv "$result/score.json.part" "$result/score.json"
}

worker() {
    local slot=$1 gpu=$2 i job dir failed=0
    for i in "${!jobs[@]}"; do
        (( i % nodes == node && (i / nodes) % ${#gpus[@]} == slot )) || continue
        job=${jobs[i]}
        dir="$out/results/$job"
        mkdir -p "$dir" || return 1
        if [[ -s "$dir/score.json" ]]; then printf 'GPU %s SKIP %s\n' "$gpu" "$job"; continue; fi
        printf 'GPU %s START %s\n' "$gpu" "$job"
        if run_job "$gpu" "$job" >> "$dir/run.log" 2>&1; then
            printf 'GPU %s DONE %s\n' "$gpu" "$job"
        else
            printf 'GPU %s FAIL %s (see %s)\n' "$gpu" "$job" "$dir/run.log" >&2
            failed=1
        fi
    done
    return "$failed"
}

pids=()
for slot in "${!gpus[@]}"; do worker "$slot" "${gpus[slot]}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done

python3 - "$out" <<'PY' | tee "$out/scores.tsv"
import json
from pathlib import Path
import sys

print("method\tbenchmark\tscore")
for path in sorted((Path(sys.argv[1]) / "results").glob("*/*/score.json")):
    print(path.parent.parent.name, path.parent.name, json.loads(path.read_text())["score"], sep="\t")
PY
printf 'Logs and scores: %s\n' "$out"
exit "$status"
