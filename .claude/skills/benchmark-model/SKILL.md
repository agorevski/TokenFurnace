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
  between the pairs crosses PCIe.
- Prompt length, output length, concurrency, context size, and cache state.

Read the host constraints and a comparable target:

```bash
cat hardware/4x-rtx8000/README.md
cat hardware/4x-rtx8000/profile.env
find targets -maxdepth 3 -type f \( -name 'target.env' -o -path '*/profiles/*.env' \) | sort
cat targets/<similar-target>/README.md
cat targets/<similar-target>/PERFORMANCE.md
```

Important constraints:

- Four Quadro RTX 8000 GPUs provide 48 GiB each and CUDA capability `sm_75`.
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
targets/<target>/profiles/<profile>.env
targets/<target>/README.md
targets/<target>/PERFORMANCE.md
```

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
CUDA_VISIBLE_DEVICES=0,1
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

This step applies only to a llama.cpp profile. The repository harness runs five
repetitions and emits JSON for 512-token prefill, 4096-token prefill, and
128-token generation:

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
- one NVLink pair versus all four GPUs;
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

Create or copy one representative prompt into the run directory before the
first profile, then reuse the exact file and record its checksum:

```bash
cp <fixed-prompt-file> "$ARTIFACT_DIR/prompt.txt"
sha256sum "$ARTIFACT_DIR/prompt.txt" | tee "$ARTIFACT_DIR/prompt.sha256"
```

Run at least these API shapes when relevant:

```bash
# Interactive single-request decode/end-to-end latency.
./scripts/benchmark-model.sh "$TARGET" "$PROFILE" -- --max-tokens 256 \
  | tee "$ARTIFACT_DIR/api-decode.json"

# Prefill-oriented request. Use one fixed UTF-8 prompt file for every profile.
./scripts/benchmark-model.sh "$TARGET" "$PROFILE" -- \
  --prompt-file "$ARTIFACT_DIR/prompt.txt" --max-tokens 1 \
  | tee "$ARTIFACT_DIR/api-prefill.json"

# Aggregate serving throughput.
./scripts/benchmark-model.sh "$TARGET" "$PROFILE" -- \
  --prompt-file "$ARTIFACT_DIR/prompt.txt" --max-tokens 256 \
  --concurrency 16 --requests 64 \
  | tee "$ARTIFACT_DIR/api-concurrency-16.json"
```

The API harness uses temperature zero. Its prompt rate is end-to-end and, for
the prefill test, includes HTTP plus one generated token. Do not label it as a
kernel-only prefill rate. The concurrent result reports aggregate throughput
and mean single-request output rate; keep them distinct.

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
  prompt/output/concurrency, repetitions, cache state, failures, thermal caveats,
  and interpretation.
- `targets/$TARGET/README.md`: profile purpose, commands, measured default, and
  known blockers or unsupported paths.

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
