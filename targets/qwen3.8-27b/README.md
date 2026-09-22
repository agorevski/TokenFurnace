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
| `llama-cpp-q4km-2gpu-tensor-ctx262k` (default) | NVLink pair 2-3 | Full native 262,144-token context |
| `llama-cpp-q4km-2gpu-tensor` | NVLink pair 2-3 | Faster-starting 65,536-token context |
| `llama-cpp-q4km-1gpu` | GPU 3 | Lower resource use; leaves GPU 2 free |
| `llama-cpp-q4km-2gpu` | NVLink pair 2-3 | Layer-split baseline |
| `vllm-official-fp16-tp2` | NVLink pair 2-3 | Official unquantized safetensors; BF16 cast to native Turing FP16 |
| `vllm-official-fp16-tp2-mtp1` | NVLink pair 2-3 | Fastest official-weight latency; built-in MTP-1 |
| `vllm-docker-qwen38-fp16-tp2` | NVLink pair 2-3 | Dedicated `vllm/vllm-openai:qwen38` image with official weights cast to FP16 |
| `vllm-docker-qwen38-awq-int4-1gpu-mtp2` | GPU 1 | Dedicated `qwen38` image with W4A16 AWQ (Q4) and MTP-2 |
| `vllm-awq-int4-1gpu` | GPU 1 | W4A16 AWQ continuous-batching baseline |
| `vllm-awq-int4-1gpu-mtp1` | GPU 1 | One-token speculative control |
| `vllm-awq-int4-1gpu-mtp2` | GPU 1 | Highest measured throughput for 2-5 concurrent coding sessions |

The 17,106,773,984-byte Q4_K_M file is only 15.93 GiB and fits comfortably on
one 48 GiB RTX 8000. Despite that, tensor parallelism across NVLink pair 2-3 is
the measured winner: 40.37 tok/s native decode and 1113.88 tok/s at 4096-token
prefill, versus 28.10 and 659.75 tok/s on one GPU. The default extends this
topology to the model's full 262,144-token native context; a load-and-generate
test used about 17.85 GiB per GPU and delivered 44.38 output tok/s on a warm
59-token prompt. See [PERFORMANCE.md](PERFORMANCE.md) for measurement details.

The requested Unsloth repository did not expose a separate MTP draft GGUF when
this target was created. The GGUF profiles therefore do not silently combine
artifacts from another publisher. The official checkpoint does include its own
MTP head, exposed by the separate `vllm-official-fp16-tp2-mtp1` profile.

Row split was also tested and failed to load on this llama.cpp build, so no
nonfunctional row profile is shipped.

## KV and prefix caching

All profiles keep the generation KV cache enabled and reuse matching prompt
prefixes. The llama.cpp profiles reserve up to 128 GiB of host RAM for cached
KV prefixes, preserve idle slots, and reuse matching chunks of at least 256
tokens. The vLLM profiles enable automatic prefix caching with aligned hybrid
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

The default profile uses NVLink-connected GPUs 2 and 3, allocates the full
262,144-token context, enables reasoning at medium effort, and listens at
`http://127.0.0.1:8092/v1` with model alias `qwen3.8-27b`. From another
terminal, verify that loading completed:

```bash
./scripts/status.sh qwen3.8-27b
```

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
