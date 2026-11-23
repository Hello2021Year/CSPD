# Executable-code verification

Code generation and code execution are separate stages. The SGLang process only
generates token IDs. It sends extracted code and private test payloads over a
versioned JSONL protocol to an explicit external verifier command; code tasks
fail closed when that command is absent or returns a malformed or non-isolated
receipt.

## Pinned semantics

- LiveCodeBench prompts and last-fenced-block extraction match upstream commit
  `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`.
- LiveCodeBench execution uses that commit's official `run_test` semantics.
  Each candidate runs in a disposable child with a 32 GiB address/data/stack
  ceiling and the exact PID is reaped after completion or global timeout. This
  contains the pinned upstream `check_correctness` memory/zombie failure mode
  without changing its prompt, tests, per-test timeout, or correctness logic.
- The container build pins the Python base image by digest, the upstream archive
  by SHA-256, and every direct Python dependency by version.
- Each result records the grader ID, test outcome metadata, isolation backend,
  and immutable image identity. The host validates response schema, count,
  request order, grader identity, and isolation identity before accepting it.
- Repeated samples for one question reference one sidecar payload instead of
  copying private tests into every candidate request. LiveCodeBench's compressed
  private cases stay compressed until they are inside the verifier container;
  one CPU worker decodes them once, then checks that question's candidate group.

`python_assertions` is available for published training corpora that contain
literal assertions. It runs CPython with isolated/no-site flags and the same
contained timeout contract; it is not silently substituted for LiveCodeBench.

Nemotron-Cascade v5 evaluation pins the original LiveCodeBench v5 release
snapshot (`0687ab61843a90a0cc864a2b67db729861cd0ae5`, 279 dated questions).
Using the later v6 repository snapshot for both labels backfills 44 older
questions and silently changes v5 to 323 questions; v6 remains pinned to the
later 454-question cumulative snapshot. NVIDIA's public Nemotron-Cascade
collection does not contain the approximately 9.8K CodeRL prompt set described
by the paper, so that adapter permits baseline evaluation but fails closed at
sampling instead of substituting RL-SWE, RL-Math, or another code corpus.

X-Coder uses the prompt builder and Python-fence splitter from its published
repository commit `2568105eebc6bd4bb45cc616404218ebbc1259a9`, rather than
silently replacing them with the newer generic LiveCodeBench renderer. Its
Qwen3 training script consumes only synthetic parquet parts 0000--0003; part
0004 exists in the Hub repository but is not declared as a train file. The
adapter pins those four files and the dataset revision explicitly.

AReaL CodeContests evaluation pins the released 165-question
`code_benchmark/code_contest_all/test.jsonl` artifact at dataset revision
`1abdc2c4884eb6c4411c0d5c5e42da5e2be817de`. Its published rows already contain
the exact standard-input test payload used by AReaL's evaluator, so the adapter
keeps the question bytes unchanged, uses AReaL's last-valid-fenced-block
extractor, and sends the tests through the same isolated pinned `run_test`
semantics as the released AReaL verifier. CodeContests is reported as a separate
benchmark label; it is never pooled into the LiveCodeBench score.

## Build and smoke test

Build `containers/code-verifier.Dockerfile` with a container builder, then export
an Enroot/Pyxis `.sqsh` artifact or publish the image by digest. Always compute
and retain the final artifact SHA-256. For example, where Docker and Enroot are
available:

```bash
docker build -f containers/code-verifier.Dockerfile \
  -t soft-prompt-code-verifier:lcb-28fef95e .
enroot import -o code-verifier-lcb-28fef95e.sqsh \
  dockerd://soft-prompt-code-verifier:lcb-28fef95e
sha256sum code-verifier-lcb-28fef95e.sqsh
```

From an existing interactive Slurm allocation with Pyxis enabled:

```bash
tmp_dir=$(mktemp -d)
cp tests/fixtures/code_verifier_smoke_requests.jsonl "$tmp_dir/requests.jsonl"
export SOFT_PROMPT_CODE_VERIFIER_IMAGE=/absolute/path/code-verifier-lcb-28fef95e.sqsh
scripts/run_code_verifier_pyxis.sh \
  "$tmp_dir/requests.jsonl" "$tmp_dir/responses.jsonl"
python -m json.tool --json-lines "$tmp_dir/responses.jsonl"
```

The fixture contains one correct and one incorrect program. The expected
decision sequence is `true, false`, with grader ID
`livecodebench@28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24:code_generation`
and a `slurm_pyxis` receipt. The locally validated 2026-08-31 smoke artifact had
SHA-256 `abda1cde76e1bbddc978613b297a9f6c8583b2a91cb198ac6db928c47002cf6c`;
this identifies that artifact only and is not a substitute for hashing a rebuild.

`tests/fixtures/code_verifier_memory_smoke_requests.jsonl` contains a 40 GiB
allocation attempt. Run it through the same command when validating a new
cluster/runtime combination; it must return `correct=false` with `MemoryError`
metadata while leaving the verifier Slurm step healthy.

Pass the same runner to sampling or evaluation:

```bash
soft-prompt-distill sample CONFIG --output SAMPLE_DIR --cache-dir CACHE_DIR \
  --code-verifier-command scripts/run_code_verifier_pyxis.sh
soft-prompt-distill evaluate CONFIG --target teacher \
  --output EVAL_DIR --cache-dir CACHE_DIR \
  --code-verifier-command scripts/run_code_verifier_pyxis.sh
```

## CPU concurrency and GPU overlap

Verification is CPU-only. The worker reads `SLURM_CPUS_PER_TASK` (32 by default
in the Pyxis runner), reserves two allocated CPUs per concurrent candidate, and
preserves request order. `SOFT_PROMPT_VERIFIER_WORKERS` can reduce the CPU budget.

Sampling and evaluation use a bounded producer/consumer pipeline: while the CPU
worker verifies batch N, the model runtime generates batch N+1. Only one verifier
invocation is in active service at a time and pipeline depth is two, so private
test and completion memory remains bounded. Within a verifier invocation,
question groups run in parallel while the candidates for one question reuse one
decoded test payload. The final verifier tail is drained after the GPU runtime
is shut down.

## Containment boundary

The production runner enters a read-only Pyxis container, omits the host home
mount, exposes the repository read-only, and exposes only a temporary request/
response directory read-write. Slurm supplies the batch-level CPU and memory
cgroup; the worker supplies a disposable process, a 32 GiB per-candidate
memory ceiling, the pinned checker's per-test alarm, and an aggregate timeout.
The worker refuses to run without an isolation and immutable-image receipt.

Pyxis/Enroot is process containment on a shared kernel, not a virtual-machine
security boundary, and this wrapper does not create a private network namespace.
Do not mount credentials, private datasets unrelated to the benchmark, or other
writable host paths. Fully hostile-code execution requires an administrator-
managed stronger sandbox (for example a disposable VM or microVM) and a runner
whose backend is explicitly allow-listed by the host bridge.
