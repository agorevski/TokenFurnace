---
name: benchmark-model
description: Benchmark a Hugging Face model on this repository's four-RTX-8000 host using its target profiles, llama.cpp native tests, and OpenAI-compatible API harness. Use when adding a model target, comparing GPU topology or runtime profiles, measuring prefill/decode/server throughput, or documenting measured performance.
---

# Benchmark a model on 4x RTX 8000

Run commands from the repository root. Never report projected, copied, or
partially observed numbers as measurements. Preserve raw command output and
record failures as well as successes.

## 1. Establish the test question

Define these before changing a profile:

- Hugging Face repository and exact artifact/quantization.
- Runtime: `llama.cpp` for GGUF, or `vllm` for a supported checkpoint.
- Primary metric: native prefill, native decode, single-request API latency, or
  aggregate API throughput.
- Topologies to compare. This host has NVLink pairs 0-1 and 2-3, but traffic
  between the pairs crosses PCIe. Use GPU 3 for single-GPU performance runs
  and the 2-3 pair for NVLink performance runs.
- Prompt length, output length, concurrency, context size, and cache state.

Read the host constraints and a comparable target:

```bash
cat hardware/4x-rtx8000/README.md
cat hardware/4x-rtx8000/profile.env
find targets -maxdepth 3 -type f \( -name 'target.env' -o -path '*/profiles/*.env' \) | sort
cat targets/<similar-target>/README.md
cat targets/<similar-target>/PERFORMANCE.md
cat targets/<similar-target>/benchmark.env
```

Important constraints:

- Four Quadro RTX 8000 GPUs provide 48 GiB each and CUDA capability `sm_75`.
- For performance measurements, use only physical GPU 3 for single-GPU runs
  and physical GPUs 2,3 for NVLink-pair runs. Do not substitute GPUs 0/1 or
  compare results from them as equivalent: their performance is suspected to
  differ. A four-GPU topology test necessarily involves 0/1; run it only
  when explicitly requested, and label it separately from GPU-3 baselines.
  Inspect the effective `CUDA_VISIBLE_DEVICES` after loading every profile;
  changing the caller's environment alone does not override a profile value.
- Turing has no native BF16, TF32, FP8, or FP4 tensor-core execution. Use
  `DTYPE=half` for unquantized vLLM profiles.
- FlashAttention 2 is unavailable. A framework may support CUDA 7.0+ while a
  model-specific attention, MoE, DeltaNet, or quantization kernel still rejects
  `sm_75`; require a real load-and-generate test.
- Prefer a 2-GPU tensor-parallel group inside one NVLink pair. For a four-GPU
  vLLM comparison, compare topology-aware `TP2 x PP2` with `TP4`.
- Do not assume that more GPUs improve single-stream decode. Existing targets
  show model-dependent results.

## 2. Select or create the target and profiles

Reuse an existing `targets/<target>/target.env` when it describes the exact
model family. Otherwise create:

```text
targets/<target>/target.env
targets/<target>/benchmark.env
targets/<target>/benchmark-prompt.txt
targets/<target>/profiles/<profile>.env
targets/<target>/README.md
targets/<target>/PERFORMANCE.md
```

Every model target must have a committed `targets/<target>/benchmark.env`.
Treat a missing baseline as a blocker and backfill it before testing an
existing target. This is the target's runtime-independent baseline contract,
not a runtime tuning profile. Define every comparison dimension explicitly:

```bash
# Served workload limits. Runtime profiles must resolve to these exact values.
BENCHMARK_CONTEXT_SIZE=65536
BENCHMARK_BATCH_SIZE=8192
BENCHMARK_UBATCH_SIZE=2048

# Native prefill/decode sweep.
BENCHMARK_PREFILL_TOKENS_SHORT=512
BENCHMARK_PREFILL_TOKENS_LONG=4096
BENCHMARK_OUTPUT_TOKENS=128
BENCHMARK_REPETITIONS=5

# OpenAI-compatible API workload.
BENCHMARK_API_PROMPT_FILE=targets/<target>/benchmark-prompt.txt
BENCHMARK_API_PROMPT_TOKENS=4096
BENCHMARK_API_OUTPUT_TOKENS=256
BENCHMARK_API_PREFILL_OUTPUT_TOKENS=1
BENCHMARK_API_CONCURRENCY=16
BENCHMARK_API_REQUESTS=64

# One of: cold, warm. Do not mix states in one comparison.
BENCHMARK_CACHE_STATE=cold
```

Do not copy these example values blindly. Choose values that every runtime in
the planned comparison can support, then keep the file unchanged for the
entire comparison series. If a model needs multiple workload classes, create
separate, descriptively named baseline files such as
`benchmark-interactive.env` and `benchmark-throughput.env`; record the selected
file in `BASELINE` and in every run artifact and result table:

```bash
BASELINE=${BASELINE:-targets/$TARGET/benchmark.env}
```

Never edit one baseline between runs and present the results as one comparison.

Commit the exact API prompt alongside the baseline. Generate it with the
target's tokenizer so the complete rendered request, including the chat
template and special tokens, has exactly `BENCHMARK_API_PROMPT_TOKENS`. Reuse
the same bytes for every runtime. Record its SHA-256 and verify each API
response's reported prompt-token count; a mismatch makes that result
ineligible for an apples-to-apples comparison.

Follow nearby targets rather than adding target-specific launch scripts.
`scripts/lib/common.sh` loads the hardware profile, then `target.env`, then the
selected profile. A target normally defines:

```bash
TARGET_NAME="Human-readable name"
BACKEND=llama.cpp
DEFAULT_PROFILE=<profile>
MODEL_ALIAS=<api-name>
MODEL_DIR=/home/algore/models/<target>
HF_REPO=<owner/repository>
SERVER_PORT=<unused-port>
LLAMA_DIR=/home/algore/llama.cpp-dspark-current
BUILD_DIR=/home/algore/llama.cpp-dspark-current/build-peer2048
VLLM_ENV=/home/algore/miniconda3/envs/vllm
```

A llama.cpp profile must identify the GGUF and its GPU layout:

```bash
PROFILE_DESCRIPTION="..."
BACKEND=llama.cpp
HF_REPO=<GGUF-repository>
MODEL_DIR=/home/algore/models/<target>-gguf
MODEL=$MODEL_DIR/<file-or-first-shard>.gguf
DOWNLOAD_PATTERNS="<file-or-directory-glob>"
CUDA_VISIBLE_DEVICES=2,3
TENSOR_SPLIT=1,1
SPLIT_MODE=layer
FLASH_ATTN=on
CTX_SIZE=65536
PARALLEL=1
BATCH_SIZE=8192
UBATCH_SIZE=2048
```

For split GGUFs, point `MODEL` at shard `00001` and include all shards in
`DOWNLOAD_PATTERNS`. Keep the number of `TENSOR_SPLIT` entries equal to the
number of visible devices; the backend intentionally fails fast otherwise.

A vLLM profile must set `BACKEND=vllm`, `MODEL=$MODEL_DIR`, `DTYPE=half`,
`CUDA_VISIBLE_DEVICES`, `TENSOR_PARALLEL_SIZE`, optional
`PIPELINE_PARALLEL_SIZE`, `MAX_MODEL_LEN`, and
`GPU_MEMORY_UTILIZATION`. Copy model-specific parser, language-only, expert
parallel, prefix-cache, or speculative settings only when the runtime and model
require them.

Create separate profiles for each controlled comparison. Change one dimension
at a time and encode it in the profile name; do not edit one profile between
runs and lose the tested configuration.

Before running, source the selected baseline and compare its values with every
profile's resolved settings. Use these cross-runtime mappings:

| Baseline dimension | llama.cpp | vLLM | Other runtimes |
|---|---|---|---|
| Context | `CTX_SIZE` | `MAX_MODEL_LEN` | Served context limit |
| Batch token cap | `BATCH_SIZE` | `MAX_NUM_BATCHED_TOKENS` | Equivalent scheduler/prefill batch token cap |
| Microbatch token cap | `UBATCH_SIZE` | Explicit equivalent, if supported | Equivalent execution microbatch cap |
| API concurrency | `BENCHMARK_API_CONCURRENCY` | Same client workload | Same client workload |
| API requests | `BENCHMARK_API_REQUESTS` | Same client workload | Same client workload |

Context and batch values must be equal, not merely large enough. If a runtime
does not expose an equivalent batch or microbatch control, mark that field
`unmapped` in `PERFORMANCE.md` and do not call the result a strict
apples-to-apples comparison. Do not silently replace an unsupported setting
with a runtime default.

Print and retain a resolved comparison manifest before launching a run. It must
include the baseline file and checksum, model artifact and quantization,
runtime/profile, context, batch, microbatch, native prompt/output lengths,
API prompt path and checksum, API prompt/output lengths, concurrency, request
count, repetitions, cache state, and temperature. Stop if any workload value
differs between profiles; topology and runtime-specific implementation knobs
are the intended independent variables.

## 3. Capture provenance and check prerequisites

Create a run directory. Raw artifacts are evidence; inspect `git status` and do
not commit them automatically.

```bash
TARGET=<target>
PROFILE=<profile>
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACT_DIR="$PWD/.benchmark-runs/$TARGET/$RUN_ID"
mkdir -p "$ARTIFACT_DIR"

git rev-parse HEAD | tee "$ARTIFACT_DIR/repository-commit.txt"
git status --short | tee "$ARTIFACT_DIR/repository-status.txt"
nvidia-smi | tee "$ARTIFACT_DIR/nvidia-smi-before.txt"
nvidia-smi topo -m | tee "$ARTIFACT_DIR/topology.txt"
nvidia-smi nvlink --status | tee "$ARTIFACT_DIR/nvlink.txt"
cp "targets/$TARGET/target.env" "$ARTIFACT_DIR/"
cp "targets/$TARGET/profiles/$PROFILE.env" "$ARTIFACT_DIR/"
BASELINE=${BASELINE:-targets/$TARGET/benchmark.env}
cp "$BASELINE" "$ARTIFACT_DIR/"
set -a
source "$BASELINE"
set +a
cp "$BENCHMARK_API_PROMPT_FILE" "$ARTIFACT_DIR/prompt.txt"
sha256sum "$BASELINE" "$BENCHMARK_API_PROMPT_FILE" \
  | tee "$ARTIFACT_DIR/baseline.sha256"
./scripts/benchmark-config.sh "$TARGET" "$PROFILE" \
  | tee "$ARTIFACT_DIR/comparison-manifest.env"
```

Require an idle or explicitly reserved set of GPUs. Record any unavoidable
co-tenancy. Check temperature, power, clocks, memory use, and throttling before
and after each run; do not compare a cool run with a thermally throttled run.

```bash
nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu \
  --format=csv | tee "$ARTIFACT_DIR/gpu-before.csv"
command -v hf python3 curl nvidia-smi
```

For llama.cpp, build the repository's CUDA/NCCL `sm_75` binaries and preserve
their identity:

```bash
./scripts/build-llama-cpp.sh 2>&1 | tee "$ARTIFACT_DIR/build-llama-cpp.log"
/home/algore/llama.cpp-dspark-current/build-peer2048/bin/llama-server --version \
  | tee "$ARTIFACT_DIR/llama-version.txt"
git -C /home/algore/llama.cpp-dspark-current rev-parse HEAD \
  | tee "$ARTIFACT_DIR/llama-commit.txt"
```

For vLLM, verify the profile's executable before downloading a large model:

```bash
/home/algore/miniconda3/envs/vllm/bin/python --version \
  | tee "$ARTIFACT_DIR/vllm-python.txt"
/home/algore/miniconda3/envs/vllm/bin/vllm --version \
  | tee "$ARTIFACT_DIR/vllm-version.txt"
```

## 4. Download and verify the model

Use the target/profile-aware downloader:

```bash
set -o pipefail
./scripts/download-model.sh "$TARGET" "$PROFILE" 2>&1 \
  | tee "$ARTIFACT_DIR/download.log"
```

Resolve `MODEL` by loading the profile, then record existence, sizes, and
checksums. For a directory checkpoint, inventory the files rather than hashing
terabytes repeatedly when that cost is not justified.

```bash
bash -c 'source scripts/lib/common.sh; load_target "$1" "$2"; printf "%s\n" "$HF_REPO" "$MODEL"' \
  _ "$TARGET" "$PROFILE" | tee "$ARTIFACT_DIR/resolved-model.txt"
find /home/algore/models -maxdepth 3 -type f -newer "$ARTIFACT_DIR/target.env" \
  -printf '%s %p\n' | sort > "$ARTIFACT_DIR/downloaded-files.txt"
```

If files already existed, explicitly inventory the resolved model path with
`find`, `stat`, and, when practical, `sha256sum`. Never infer a complete
download merely from directory existence.

## 5. Run native llama.cpp prefill and decode benchmarks

This step applies only to a supported native profile. The repository harness
loads the selected baseline, validates the profile's context, batching, and
cache state, and uses the baseline's native prompt lengths, output length, and
repetition count.

```bash
set -o pipefail
./scripts/benchmark-native.sh "$TARGET" "$PROFILE" \
  2>&1 | tee "$ARTIFACT_DIR/native.json"
```

The harness converts server-style comma-separated `TENSOR_SPLIT` values to
llama-bench's slash syntax. Do not bypass it with comma-separated `-ts`, which
llama-bench interprets as multiple benchmark variants.

Repeat the complete sweep for each topology profile. Keep model bytes,
quantization, flash attention, batch/ubatch, prompt/decode lengths, repetition
count, and thermal window fixed. Useful llama.cpp comparisons include:

- one GPU versus one NVLink pair;
- 2-GPU `layer` versus `tensor` split, if both load correctly;
- the GPU 2-3 NVLink pair versus all four GPUs, only when explicitly requested;
- all-four-GPU layer split when the model cannot fit on a pair.

Treat load failures, allocation failures, corruption, and unsupported split
modes as results to document, not reasons to silently omit a row.

## 6. Run server and OpenAI-compatible API benchmarks

Start the server in a dedicated terminal and capture its complete log:

```bash
TARGET=<target>
PROFILE=<profile>
RUN_ID=<same-run-id>
ARTIFACT_DIR="$PWD/.benchmark-runs/$TARGET/$RUN_ID"
set -o pipefail
./scripts/serve-model.sh "$TARGET" "$PROFILE" \
  2>&1 | tee "$ARTIFACT_DIR/server.log"
```

In another terminal, use the same run directory and wait for a real health
check:

```bash
./scripts/status.sh "$TARGET" "$PROFILE" \
  | tee "$ARTIFACT_DIR/status-loaded.txt"
```

Do not benchmark until status reports the configured endpoint healthy and the
server log shows that loading completed.

Use the target's committed baseline prompt for every profile, copy it into the
run directory, and record its checksum:

```bash
BASELINE=${BASELINE:-targets/$TARGET/benchmark.env}
set -a
source "$BASELINE"
set +a
cp "$BENCHMARK_API_PROMPT_FILE" "$ARTIFACT_DIR/prompt.txt"
sha256sum "$ARTIFACT_DIR/prompt.txt" | tee "$ARTIFACT_DIR/prompt.sha256"
```

Run at least these API shapes when relevant:

```bash
# Baseline API workload. The harness loads prompt, output, concurrency,
# request-count, temperature, and expected prompt-token values from BASELINE.
./scripts/benchmark-model.sh "$TARGET" "$PROFILE" \
  | tee "$ARTIFACT_DIR/api-decode.json"

# Prefill-oriented request. Use one fixed UTF-8 prompt file for every profile.
./scripts/benchmark-model.sh "$TARGET" "$PROFILE" -- \
  --prompt-file "$ARTIFACT_DIR/prompt.txt" \
  --max-tokens "$BENCHMARK_API_PREFILL_OUTPUT_TOKENS" \
  | tee "$ARTIFACT_DIR/api-prefill.json"

# Aggregate serving throughput.
./scripts/benchmark-model.sh "$TARGET" "$PROFILE" -- \
  --prompt-file "$ARTIFACT_DIR/prompt.txt" \
  --max-tokens "$BENCHMARK_API_OUTPUT_TOKENS" \
  --concurrency "$BENCHMARK_API_CONCURRENCY" \
  --requests "$BENCHMARK_API_REQUESTS" \
  | tee "$ARTIFACT_DIR/api-concurrency.json"
```

The API harness uses temperature zero. Its prompt rate is end-to-end and, for
the prefill test, includes HTTP plus one generated token. Do not label it as a
kernel-only prefill rate. The concurrent result reports aggregate throughput
and mean single-request output rate; keep them distinct.

After every API command, require the reported prompt-token count to equal
`BENCHMARK_API_PROMPT_TOKENS` and the completion-token count to equal the
requested value unless a valid EOS stopped generation. Record EOS-shortened
runs separately; do not mix them with fixed-output throughput results.

Warm up model-specific JIT compilation before timed steady-state requests, but
save and label the first-request result separately. For cache-controlled tests:

- compare cold and warm runs only with byte-identical prompts;
- record whether llama.cpp prompt caching or vLLM APC is enabled;
- create a dedicated no-cache profile when measuring cache-cold behavior;
- do not mix cached and uncached requests in one aggregate.

After each profile:

```bash
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu \
  --format=csv | tee "$ARTIFACT_DIR/gpu-after.csv"
./scripts/status.sh "$TARGET" "$PROFILE" \
  | tee "$ARTIFACT_DIR/status-after.txt"
```

Stop the server cleanly from its terminal before loading another profile.

## 7. Compare topology and interpret results

Compare only runs with identical workload and model artifacts. Calculate
percentage differences from captured values, and state the formula or retain
the calculation. Separate:

- native 512/4096 prefill throughput;
- native 128-token decode throughput;
- API end-to-end latency and output throughput;
- server-reported prompt/decode timings, when present;
- aggregate throughput at a stated concurrency;
- cold-start/JIT time and steady-state time;
- cache-cold and cache-warm behavior;
- VRAM fit, context/KV capacity, and correctness.

Start every comparison by diffing the retained resolved manifests. A row is
comparison-eligible only when model bytes/quantization and every
`BENCHMARK_*` value match. If the experiment intentionally compares
quantizations, identify quantization as the independent variable and still
require all workload values to match.

Interpret results using the topology:

- A 2-GPU result on 0-1 or 2-3 stays within an NVLink pair.
- A four-GPU tensor collective crosses the slower PCIe host bridge.
- `TP2 x PP2` confines vLLM tensor collectives to the two NVLink pairs, while
  `TP4` crosses pairs.
- Dense and sparse/MoE models may respond differently; do not generalize a
  winning split from another target.
- Thermal throttling, compilation, cache hits, context changes, precision
  changes, or different quantization invalidate a simple topology comparison.

Validate generated text, finish reason, requested token count, and server logs.
Throughput from malformed or corrupted output is not a successful benchmark.

## 8. Document without inventing results

Update only the target documentation:

- `targets/$TARGET/PERFORMANCE.md`: measured table, date, hardware, exact model
  artifact and size/hash, runtime/build commit or version, profile settings,
  selected baseline file and checksum, prompt/output/concurrency, context,
  batch/microbatch mapping, repetitions, cache state, failures, thermal
  caveats, and interpretation.
- `targets/$TARGET/README.md`: profile purpose, commands, measured default, and
  known blockers or unsupported paths.

Standardize every `PERFORMANCE.md` for fast comparison. Immediately after the
level-one title, before methodology, caveats, prose, or unmeasured work, place
these two complete measured-result tables:

```markdown
## Prefill

| Runtime / Checkpoint | Native Workload | Tok/s | Std Deviation |
|---|---|---:|---:|
| [runtime and checkpoint](#matching-detail-anchor) | 512-token prefill | 1234.56 | 12.34 |

## Decode

| Runtime / Checkpoint | Batch Size | Input Token Length | Aggregate tok/s | Std Deviation | Mean TFTT | Mean TPOT |
|---|---:|---:|---:|---:|---:|---:|
| [runtime and checkpoint](#matching-detail-anchor) | 1 | 4096 | 45.67 | 0.42 | 123 ms | 21.0 ms |
```

The tables are the above-the-fold index of all valid measured results:

- Put every eligible native prefill measurement in **Prefill**, one row per
  runtime/checkpoint/workload. Keep different prompt lengths in separate rows.
- Put every eligible API/server decode or generation-throughput measurement in
  **Decode**, one row per runtime/checkpoint/batch/input-length combination.
  `Batch Size` means simultaneous requests or sequences for serving results,
  not the runtime's token batch cap.
- Make every `Runtime / Checkpoint` value a Markdown link to an explicit,
  stable detail anchor lower in the same document. The linked section must
  identify the runtime version/build, checkpoint or artifact, hardware/profile,
  workload, cache state, and retained evidence.
- Sort comparable rows by throughput descending so the fastest result is
  visible first. Keep unlike workload lengths and batch sizes explicit; never
  imply that unlike rows are controlled comparisons.
- Use the units shown in the headers. `Mean TFTT` is the measured mean
  time-to-first-token value when the source harness calls it TTFT. Use `—` for
  a metric that was not measured or retained; never derive or invent standard
  deviation, TFTT, TPOT, batch size, or input length.
- Do not place projected, unmeasured, failed, malformed-output, or
  correctness-only results in either table. Keep those statuses in the
  detailed sections below.
- Preserve detailed tables and narrative as evidence. The top tables summarize
  them; they do not replace provenance, comparability caveats, or failure
  records.

Do not modify the repository root `README.md` as part of this workflow unless
the caller explicitly requests it.

Use explicit labels:

- **Measured**: completed on this host with retained output.
- **Projected**: an estimate, never placed in a measured table.
- **Unmeasured**: configured but not run.
- **Failed**: attempted with the captured diagnostic.

Never fabricate missing standard deviations, server timings, token counts,
memory use, or speedups. If raw data was not retained, state that limitation.
Before finishing, inspect the diff and ensure profile defaults are justified by
the measured workload rather than by assumption:

```bash
git diff -- targets/<target> .claude/skills/benchmark-model/SKILL.md
git status --short
```

## 9. Failure handling

On failure, preserve the command, profile snapshot, logs, GPU state, and exact
error before changing anything.

- **Download incomplete:** inventory missing shards and rerun
  `./scripts/download-model.sh TARGET PROFILE`; do not benchmark a partial
  checkpoint.
- **llama.cpp missing model support:** record the build commit and error, update
  with `./scripts/build-llama-cpp.sh`, then rerun the load test. Do not claim a
  speed result from a different build without labeling it.
- **CUDA/NCCL/build failure:** save the CMake/build log and verify CUDA 12.4,
  GCC, `sm_75`, library paths, and NCCL availability against the hardware
  profile. Do not disable required acceleration silently.
- **vLLM kernel rejects Turing:** try only a documented supported fallback
  backend/attention mode in a separate profile. Record the original failure.
- **Out of memory:** reduce context, batching, concurrency, or use a fitting
  quant/profile one variable at a time; never compare the changed workload as
  if it were identical.
- **Server unhealthy:** inspect `server.log`, port use, resolved model path, and
  GPU allocation. Do not run the API harness against another process.
- **Thermal or competing load:** stop the run, cool or reserve the GPUs, and
  repeat the full comparison.
- **Incorrect output:** mark the configuration failed even if throughput is
  high, retain a minimal reproducer, and do not promote it as default.

If no valid run completes, document the blocker and leave performance values
unmeasured.
