# Qwen3.8-27B

Official checkpoint: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).
The optimized profiles use the user-selected
[`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
distribution.

Qwen3.8-27B is a dense multimodal model with 64 language layers: 48 Gated
DeltaNet layers and 16 full-attention layers. It has a 5120-wide hidden state,
17,408-wide FFN, 27B parameters, and 262,144-token native context.

## Profiles

| Profile | GPUs | Goal |
|---|---|---|
| `llama-cpp-q4km-4gpu-tensor-ctx262k` (default) | GPUs 0-3 | Fastest measured decode and prefill, full 262,144-token context |
| `llama-cpp-q4km-4gpu-tensor` | GPUs 0-3 | 8192-token controlled benchmark profile |
| `llama-cpp-q4km-4gpu-layer` | GPUs 0-3 | Layer-split comparison |
| `llama-cpp-q4km-4gpu-tensor-queues4x` / `llama-cpp-q4km-4gpu-layer-queues4x` | GPUs 0-3 | CUDA launch-queue comparisons |
| `llama-cpp-q4km-2gpu-tensor-01-ctx262k` | NVLink pair 0-1 | Fastest measured two-GPU decode fallback, full context |
| `llama-cpp-q4km-2gpu-tensor-01-ctx8k` | NVLink pair 0-1 | Matching-context topology comparison |
| `llama-cpp-q4km-1gpu-0-ctx8k` | GPU 0 | Single-card topology control |
| `llama-cpp-q4km-2gpu-tensor-ctx262k` | NVLink pair 2-3 | Full-context option when GPUs 0-1 are occupied |
| `llama-cpp-q4km-2gpu-tensor-ctx8k` | NVLink pair 2-3 | Matching-context topology and API comparison |
| `llama-cpp-q4km-2gpu-tensor` | NVLink pair 2-3 | Faster-starting 65,536-token context |
| `llama-cpp-q4km-1gpu` | GPU 3 | Lower resource use; leaves GPU 2 free |
| `llama-cpp-q4km-1gpu-ctx262k-prefill` | GPU 3 | Standalone cold-cache 32K-prefill baseline in a 262K context |
| `llama-cpp-q4km-2gpu` | NVLink pair 2-3 | Layer-split baseline |
| `vllm-official-fp16-tp2` | NVLink pair 2-3 | Official unquantized safetensors; BF16 cast to native Turing FP16 |
| `vllm-official-fp16-tp2-mtp1` | NVLink pair 2-3 | Fastest official-weight latency; built-in MTP-1 |
| `vllm-docker-qwen38-fp16-tp2` | NVLink pair 2-3 | Dedicated `vllm/vllm-openai:qwen38` image with official weights cast to FP16 |
| `vllm-docker-qwen38-awq-int4-1gpu-mtp2` | GPU 1 | Dedicated `qwen38` image with W4A16 AWQ (Q4) and MTP-2 |
| `vllm-awq-int4-1gpu` | GPU 1 | W4A16 AWQ continuous-batching baseline |
| `vllm-awq-int4-1gpu-mtp1` | GPU 1 | One-token speculative control |
| `vllm-awq-int4-1gpu-mtp2` | GPU 1 | Highest measured throughput for 2-5 concurrent coding sessions |
| `vllm-awq-int4-1gpu-mtp2-gpu3` | GPU 3 | Same AWQ/MTP-2 configuration on the preferred single-GPU performance card |
| `vllm-awq-int4-1gpu-mtp2-gpu3-prefill` | GPU 3 | Standalone AWQ cold-cache 32K-prefill baseline in a 262K context |
| `ninfer-groupwise-int-1gpu-mtp0` | GPU 0 | Historical container-v2 baseline; faster on the low-acceptance native corpus |
| `ninfer-groupwise-int-1gpu-mtp3` | GPU 0 | Historical container-v2 MTP-3; measured short-request API winner |

The 17,106,773,984-byte Q4_K_M file is only 15.93 GiB and fits comfortably on
one 48 GiB RTX 8000. Four-GPU tensor parallelism measured 55.90 tok/s native
decode and 1625.56 tok/s at 4096-token prefill, versus 42.78 and 1104.21
tok/s on the NVLink pair in the same sweep. The default retains the model's
262,144-token context, loaded on all four GPUs and delivered 51.50 end-to-end
output tok/s on a short request. The four-GPU profiles enable peer access and
use FP16 KV. Despite the slower PCIe crossing between NVLink pairs, they beat
the two-GPU option for this model; four-GPU *layer* split was slower for decode.
See [PERFORMANCE.md](PERFORMANCE.md) for controlled measurements and caveats.

A subsequent matched topology sweep also compared GPU 0 alone and both NVLink
pairs. Four-GPU tensor still led: 54.17 tok/s native decode versus 44.96 on
pair 0-1 and 40.60 on pair 2-3. The 0-1 pair is the fastest measured
two-GPU decode fallback; its full-context profile loaded and generated
correct text. GPU temperatures reached 85-87 C on the single-card and
two-card runs, so small differences between pairs should not be generalized
to other thermal conditions.

The requested Unsloth repository did not expose a separate MTP draft GGUF when
this target was created. The GGUF profiles therefore do not silently combine
artifacts from another publisher. The official checkpoint does include its own
MTP head, exposed by the separate `vllm-official-fp16-tp2-mtp1` profile.

Row split was also tested and failed to load on the earlier llama.cpp build, so no
nonfunctional row profile is shipped.

## KV and prefix caching

All profiles keep the generation KV cache enabled. The llama.cpp profiles
reserve up to 128 GiB of host RAM for cached KV prefixes and preserve idle
slots. This llama.cpp build does not support `cache_reuse` for tensor contexts,
so the four-GPU tensor profiles do not request 256-token partial-prefix reuse.
The vLLM profiles enable automatic prefix
caching with aligned hybrid
Mamba/DeltaNet state and collision-resistant SHA-256 keys.

Neither llama.cpp nor vLLM exposes a time-based KV-cache TTL. Cached prefixes
therefore remain available for the requested 24-hour operating window as long
as the server stays running and cache pressure does not evict them. A server
restart clears the in-memory cache; the 128 GiB llama.cpp limit is intentionally
bounded rather than risking host exhaustion with unlimited caching.

## Setup and run

From the repository root, run the target-specific setup once. It validates the
GPU/CUDA tools, builds llama.cpp for the RTX 8000's `sm_75` architecture, and
downloads the measured Q4_K_M artifact:

```bash
./scripts/setup-qwen3.8-27b.sh
```

Then start the OpenAI-compatible server:

```bash
./scripts/serve-model.sh qwen3.8-27b
```

The default profile uses all four GPUs with equal tensor fractions, peer
access, flash attention, FP16 KV, stock CUDA launch queues, and the full
262,144-token context. It enables reasoning at medium effort and listens at
`http://127.0.0.1:8092/v1` with model alias `qwen3.8-27b`. From another
terminal, verify that loading completed:

```bash
./scripts/status.sh qwen3.8-27b
```

All four GPUs must be available for the default. When only one NVLink pair is
free, select the matching full-context profile explicitly:

```bash
./scripts/serve-model.sh qwen3.8-27b llama-cpp-q4km-2gpu-tensor-01-ctx262k
# Or, if GPUs 0-1 are occupied:
./scripts/serve-model.sh qwen3.8-27b llama-cpp-q4km-2gpu-tensor-ctx262k
```

The 262K slot was loaded and tested with short, 5,449-token, and cold
32,768-token prompts. The 32K request generated 1,024 tokens at 1,414.03
tok/s server-reported prefill and 51.20 tok/s server-reported decode.
A 262K-token *prefill* has not been benchmarked.

## Reproducible benchmark baseline

[`benchmark.env`](benchmark.env) records the GPU 3 cold-prefill workload,
and [`benchmark-prompt.txt`](benchmark-prompt.txt) is the exact prompt used
in the measured **32,768-token** requests (SHA-256
`4afb1718fe125320e62ff99962e7fe9056bc3e14c4273840f4acf83182b2cefe`).
HyperQwen AutoRound, llama.cpp Q4_K_M, and vLLM AWQ+MTP-2 each served with
a **262,144-token context** and prefix caching disabled; two one-token
prefill-oriented requests per runtime were measured on GPU 3. The long
prompt's token count was confirmed separately against each server's chat
template and response usage. The 1,024-token output setting in
`benchmark.env` was also measured twice per runtime with the same uncached
32K prompt, GPU 3 and 262K served context; see the
[long-request results](PERFORMANCE.md#gpu3-long-decode) for TTFT and
post-first-output token rates.

Generate and retain the resolved settings before each run:

```bash
./scripts/benchmark-config.sh qwen3.8-27b \
  llama-cpp-q4km-1gpu-ctx262k-prefill
./scripts/benchmark-config.sh qwen3.8-27b \
  vllm-awq-int4-1gpu-mtp2-gpu3-prefill
```

**Treat these as similar-INT4, independent baselines, not a same-weight runtime
comparison.** The quantized weights differ (AutoRound versus AWQ versus
Q4_K_M), and internal batching/quantized kernels are tuned differently.
HyperQwen does not load GGUF; vanilla vLLM's loader cannot map this Qwen3.8
GGUF, which also lacks an MTP head. vLLM has no direct `UBATCH_SIZE`
equivalent. Measured rates, raw artifacts and limitations are
in [PERFORMANCE.md](PERFORMANCE.md); the historical 4K/65K context results
are not used as the current GPU 3 baseline.

## Official unquantized checkpoint

The official checkpoint contains 55,562,855,904 bytes of BF16 language and
vision weights. RTX 8000 has no native BF16 execution, so the least-altered
practical test loads the official safetensors directly with vLLM and casts them
to FP16 at runtime. This is not weight quantization: there are no scales,
groups, calibration data, or replacement quantized kernels. The profile uses
`--language-model-only`, which omits the vision tower but leaves every language
model weight intact.

```bash
./scripts/download-model.sh qwen3.8-27b vllm-official-fp16-tp2
./scripts/serve-model.sh qwen3.8-27b vllm-official-fp16-tp2-mtp1
./scripts/benchmark-model.sh qwen3.8-27b vllm-official-fp16-tp2-mtp1 -- --max-tokens 512
```

MTP-1 raised steady single-request throughput from 16.99 to 31.44 tok/s with
84.5% draft acceptance. The first inference compiles a FlashInfer mask kernel
and is not representative; benchmark a second identical request for steady
state. The profile pins GCC 13 and the Conda CUDA runtime link path required by
that JIT on this host.

### Dedicated qwen38 Docker image with Q4 weights

The upstream `vllm/vllm-openai:qwen38` image can serve the same local official
model using the vLLM-compatible W4A16 AWQ checkpoint. The Q4 profile mounts
the checkpoint read-only, uses GPU 1, host networking and IPC, FP16 activations
and KV cache, Triton attention, and built-in MTP-2. Because this host does not
have NVIDIA Container Toolkit
configured for Docker, the profile passes the selected GPU devices and three
required host driver libraries directly into the otherwise unprivileged
container. The vLLM compile cache persists under
`/home/algore/.cache/vllm-qwen38-awq-int4`, avoiding the full first-start
compilation on every `--rm` container:

```bash
docker pull vllm/vllm-openai:qwen38
./scripts/serve-model.sh qwen3.8-27b vllm-docker-qwen38-awq-int4-1gpu-mtp2
./scripts/benchmark-model.sh qwen3.8-27b vllm-docker-qwen38-awq-int4-1gpu-mtp2 -- --max-tokens 128
```

The foreground server container is named `tokenfurnace-qwen38` and is removed
automatically when stopped. The FP16 Docker profile remains available for
precision comparisons.

## Single-GPU concurrent coding

For 2-5 simultaneous coding sessions on GPU 1, use the measured W4A16 AWQ
winner:

```bash
./scripts/download-model.sh qwen3.8-27b vllm-awq-int4-1gpu-mtp2
./scripts/serve-model.sh qwen3.8-27b vllm-awq-int4-1gpu-mtp2
```

For a **standalone GPU 3 AWQ serving baseline**, select the otherwise
identical GPU 3 profile instead:

```bash
./scripts/serve-model.sh qwen3.8-27b vllm-awq-int4-1gpu-mtp2-gpu3
```

The GPU 3 run is documented in [PERFORMANCE.md](PERFORMANCE.md). HyperQwen
AutoRound and llama.cpp Q4_K_M use different quantized weights, so do not
rank these independent baselines or calculate speedups between them.

This profile serves on port 8098 so it can coexist with the llama.cpp service
on port 8092. It uses FP16 activations and KV cache, 262,144-token context,
Triton attention, aligned prefix caching, medium reasoning, MTP-2, and
`max-num-seqs=8`. Server generation defaults use temperature zero and enforce
a 4,096-token maximum output. On fixed 512-token generations it measured
86.19, 171.85, and 182.71 aggregate output tok/s at concurrency 2, 4, and 5
respectively. MTP-2 beat both MTP-1 and no speculation at every tested
concurrency. See [PERFORMANCE.md](PERFORMANCE.md) for workload, per-session
rates, artifact hashes, and thermal caveats.

The paged KV cache holds roughly 272K tokens with MTP-2. It can therefore serve
one request near the full 262K limit, or multiple shorter coding sessions whose
combined active contexts fit that cache; it cannot hold five simultaneous
262K-token requests on one 48 GiB GPU.

```bash
./scripts/download-model.sh qwen3.8-27b
./scripts/benchmark-native.sh qwen3.8-27b
./scripts/serve-model.sh qwen3.8-27b
./scripts/benchmark-model.sh qwen3.8-27b -- --max-tokens 512
./scripts/status.sh qwen3.8-27b
```

Both coding-agent wrappers accept this target explicitly once its server is
running:

```bash
./scripts/copilot-local.sh qwen3.8-27b
./scripts/claude-local.sh qwen3.8-27b
```

## NInfer Turing setup

The NInfer profiles use
[`mr-september/ninfer-2080ti-22g`](https://github.com/mr-september/ninfer-2080ti-22g)
at commit `6460bf03a86f1288dc7b240d70c30887199099af`. They pin Hugging Face
revision `3526913004b1cf552cb57b88d6a5c6f5e4a89a70`, which contains the
container-v2 `qwen3_8_27b.ninfer` artifact expected by this fork. Its expected
size is 18,210,531,328 bytes and its SHA-256 is
`eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`.
The artifact's minimum runtime revision,
`52320554b5e71a9da96bff809ddf67ac5773ed63`, is ancestral to the pinned fork.

The fork's stock SM75 build has an open compile-time blocker in its
software-emulated BF16 decode-attention translation unit. The repository build
applies `patches/ninfer-sm75-int8-kv-only.patch`, which omits that unused BF16
KV decode instantiation and keeps the recommended INT8 KV path. TokenFurnace
rejects `KV_DTYPE=bf16` before launching the patched server or native benchmark.

The local build environment uses CUDA 12.9 because NInfer requires CUDA 12.8
or newer:

```bash
conda create -y -p /home/algore/.conda/envs/ninfer-build \
  -c nvidia -c conda-forge \
  cuda-toolkit=12.9 cuda-nvtx-dev=12.9 ffmpeg=6.1 \
  'libcurl>=7.85' pkg-config cmake=3.28 ninja

./scripts/setup-qwen3.8-27b.sh ninfer-groupwise-int-1gpu-mtp3
```

This installs the runtime under `/home/algore/ninfer-2080ti-22g`, builds
`ninfer`, `ninfer-serve`, and `ninfer_bench` for `sm_75`, and downloads the
pinned historical model to `/home/algore/models/qwen3.8-27b-ninfer-v2`.
Setup verifies its container header, manifest version, filename, size, and
SHA-256 before reporting success.

The repository's current default revision instead serves a 20,437,521,664-byte
container-v3 artifact (`NINFER\0\x03`, SHA-256
`81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da`).
Its manifest requires `Neroued/ninfer` revision
`98dada0e03cb073fd07f905400b5904bc6e82759` or newer and names `sm_120a`; the
pinned Turing fork correctly rejects it as non-v2. That failed attempt remains
documented in [PERFORMANCE.md](PERFORMANCE.md).

For the measured short 256-token API request, MTP-3 delivered 22.89 tok/s on
the first request and 22.27 tok/s on the second, versus 19.94 and 19.25 tok/s
for MTP-0. Use MTP-3 for similar interactive serving. Draft acceptance is
workload-dependent: the native corpus accepted only 26.5% of MTP drafts and
made MTP-3 decode 35.7% slower than MTP-0, so retain MTP-0 for low-acceptance
workloads. The historical v2 chat template also exposes a stray `</think>`
prefix on a correct non-thinking answer and failed strict exact-string
requests. See [PERFORMANCE.md](PERFORMANCE.md) for the controlled comparison
and correctness caveats.
