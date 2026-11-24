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
jobs=()
for method in "${in_methods[@]}"; do
    for bench in "${in_benches[@]}"; do jobs+=("$method/$bench"); done
    case "$method" in
        MCSI|student|mopd)
            for bench in "${ood_benches[@]}"; do jobs+=("$method/$bench"); done ;;
    esac
done

case "${1:-}" in
    --plan)
        for i in "${!jobs[@]}"; do printf 'node-%s\t%s\n' "$((i % 2))" "${jobs[i]}"; done
        exit 0 ;;
    0|1) node=$1 ;;
    *) printf 'Usage: bash environment/run_m2rl_docker.sh {0|1|--plan}\n' >&2; exit 2 ;;
esac
if (( $# != 1 )); then exit 2; fi

base=${CSPD_BASE:-lmsysorg/sglang:latest}
commit=$(git rev-parse --short=12 HEAD)
image="cspd-m2rl-repro:$commit"
root=${CSPD_OUT:-$PWD/artifacts/docker-repro}
mkdir -p "$root"
root=$(cd "$root" && pwd -P)
out="$root/$commit/node-$node"
hf=${CSPD_HF_CACHE:-$root/hf}
mkdir -p "$out" "$hf"
hf=$(cd "$hf" && pwd -P)
gpu_indexes=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
if [[ -z "$gpu_indexes" ]]; then echo 'No NVIDIA GPUs detected.' >&2; exit 1; fi
gpus=()
while IFS= read -r gpu; do gpus+=("$gpu"); done <<< "$gpu_indexes"
nvidia-smi -L
docker image inspect "$base" >/dev/null

context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
git archive HEAD | tar -xf - -C "$context"
cp containers/m2rl-repro.Dockerfile "$context/Dockerfile"
docker build --build-arg "CSPD_BASE=$base" --tag "$image" "$context"

dock() {
    local gpu=$1
    shift
    local -a docker_args=(--rm --pull=never --init --shm-size 8g)
    if [[ "$gpu" != cpu ]]; then docker_args+=(--gpus "device=$gpu"); fi
    docker run "${docker_args[@]}" \
        --user "$(id -u):$(id -g)" -e HOME=/tmp/cspd-home -e HF_HOME=/hf \
        -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
        -v "$out:/out" -v "$hf:/hf" --entrypoint /bin/bash "$image" \
        -c 'mkdir -p "$HOME"; exec "$@"' _ "$@"
}

dock "${gpus[0]}" "$gen" -c \
    'import torch, sglang; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), sglang.__version__)'
dock cpu bash -c 'mkdir -p /out/in; cp -r experiments/m2rl/eval/inputs/. /out/in/'
gpqa="$out/in/gpqa_diamond/questions.jsonl"
if [[ ! -s "$gpqa" ]] || [[ $(wc -l < "$gpqa") -ne 198 ]]; then
    if ! docker run --rm --pull=never --user "$(id -u):$(id -g)" \
        -e HF_TOKEN -e HF_HOME=/hf -v "$out:/out" -v "$hf:/hf" \
        --entrypoint "$gen" "$image" "$m/eval/build_gpqa.py" --inputs /out/in; then
        echo 'GPQA preparation failed. Check the gated dataset permission and HF_TOKEN.' >&2
    fi
fi
unset HF_TOKEN

for order in MCSI ISCM I SI CSI MSI MCI MCS M C S; do
    file="$out/prompts/$order.safetensors"
    if [[ -s "$file" ]]; then continue; fi
    dock cpu "$gen" "$m/compose.py" --paper-order "$order" --output "/out/prompts/$order.safetensors.part"
    mv "$file.part" "$file"
done
for bench in "${ood_benches[@]}"; do
    if [[ "$bench" == bfcl ]]; then continue; fi
    if [[ -s "$out/ood/$bench/questions.jsonl" && -s "$out/ood/$bench/references.jsonl" ]]; then continue; fi
    dock cpu "$gen" "$m/ood/build_inputs.py" --output /out/ood --benches "$bench" ||
        echo "Could not prepare $bench; continuing with the other benchmarks." >&2
done

score_lcb() {
    docker run --rm --pull=never --init --network none --read-only \
        --cap-drop ALL --security-opt no-new-privileges --pids-limit 512 \
        --tmpfs /tmp:rw,nosuid,nodev,mode=1777 --user 65534:65534 \
        -e HOME=/opt/lcb-cache -e HF_HOME=/opt/lcb-cache/hf -e HF_HUB_OFFLINE=1 \
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
        local questions="$out${inputs#/out}/$bench/questions.jsonl"
        local references="$out${inputs#/out}/$bench/references.jsonl"
        if [[ ! -s "$questions" || ! -s "$references" ]]; then echo "Missing inputs for $bench"; return 1; fi
        local count expected responses="$result/responses.jsonl"
        count=$(wc -l < "$questions") || return 1
        if (( count != $(wc -l < "$references") )); then echo "Incomplete inputs for $bench"; return 1; fi
        expected=$((count * k))
        if [[ ! -s "$responses" ]] || (( $(wc -l < "$responses") != expected )); then
            dock "$gpu" "$gen" "$m/eval/generate.py" --bench "$bench" --inputs "$inputs" \
                "${model[@]}" --output "$dir/responses.jsonl.part" || return 1
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
        (( i % 2 == node && (i / 2) % ${#gpus[@]} == slot )) || continue
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
