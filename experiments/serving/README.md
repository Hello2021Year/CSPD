# B200 serving benchmark

One B200 serves BF16 GPT-OSS-20B with three arms: backbone, SP256, and unmerged rank-4 LoRA using CSGMV. Each arm receives matched 20,480-token text inputs and generates 512 output tokens. At concurrency C, the client runs a warm-up burst of C requests, followed by a measured closed loop of 8C requests. Each measured request uses a distinct synthetic adapter in the adapted arms.

## Render Figure 4 data

```bash
python -m pip install matplotlib
python scripts/plot_b200.py --output b200-paper.pdf
```

Run these commands from this directory. [Archived rows](data/b200-paper.csv) include an extra concurrency-16 measurement at the larger KV budget. The plot follows the manuscript's budget at each concurrency.

## Repeat generation on B200

Use an NVIDIA GPU environment compatible with the versions in [the recorded package list](environment-lock.txt). Create a dedicated environment for this benchmark. The engine is upstream SGLang at the commit in [engine-revision.json](data/engine-revision.json) plus [engine.patch](engine.patch), which adds multi-tenant soft prompts and per-request adapters.

```bash
git clone https://github.com/sgl-project/sglang.git engine
git -C engine checkout fd4b75c6960002d341cbb7bb127de7631770c87a
git -C engine apply ../engine.patch
python -m pip install -e engine/python
python scripts/fetch_model.py
hf download zai-org/LongBench-v2 data.json --repo-type dataset \
    --revision 2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9 --local-dir data/longbench-v2
python scripts/prepare_text_inputs.py
python scripts/prepare_assets.py --model "$(cat model-path.txt)" \
    --out assets-registered-r4-long --arms sp r4 --count 765
```

Asset generation writes one SP and one all-projection LoRA per index and needs substantial disk space. Regenerate manifests in the same directory from which the run will read their absolute adapter paths.

Inside a single-B200 Slurm allocation, run the three arms sequentially for each group. The launcher uses the original request index offsets. Use fresh output directories for reruns.

```bash
for group in low c2 high c32; do
    for arm in base sp r4; do
        bash scripts/run_b200.sh "$arm" "$group" || exit
    done
done
python scripts/summarize_b200.py \
    --low "results/$SLURM_JOB_ID/low" --c2 "results/$SLURM_JOB_ID/c2" \
    --high "results/$SLURM_JOB_ID/high" --c32 "results/$SLURM_JOB_ID/c32" \
    --output results/summary.csv
python scripts/plot_b200.py --csv results/summary.csv --output b200-rerun.pdf
```

Throughput is all measured output tokens divided by elapsed time from the first request submission through the last completion. P95 TTFT measures submission to receipt of the first token. The audit checks matched input hashes, unique adapters, in-flight concurrency, token counts and absence of retractions.
