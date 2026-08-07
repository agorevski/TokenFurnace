# Qwen3-Coder-Next 80B-A3B

Official model: [`Qwen/Qwen3-Coder-Next`](https://huggingface.co/Qwen/Qwen3-Coder-Next)
(`Qwen3NextForCausalLM`, `model_type: qwen3_next`).

## Architecture

Values below are taken directly from the downloaded `config.json` and model card.

| Property | Value |
|---|---|
| Total parameters | 79.67B (about 80B; 79B non-embedding) |
| Activated parameters per token | about 3B |
| Layers | 48 |
| Hybrid layout | 12 attention + 36 Gated DeltaNet (full attention every 4th layer) |
| Attention heads | 16 Q / 2 KV, head dim 256, partial rotary 0.25 |
| DeltaNet heads | 32 value / 16 key, head dim 128, conv kernel 4 |
| Experts | 512 experts, 10 routed per token, plus 1 shared expert |
| Hidden size | 2048 (MoE intermediate 512) |
| Native context | 262,144 tokens |
| Reference dtype | bfloat16 (see numerical note below) |
| All weights resident | yes; no layer offload to host |

The hybrid Gated DeltaNet + sparse MoE design is why only about 3B of the 80B
parameters are active per token, but **all** weights must be resident in VRAM.

### Weight footprint

Quant sizes are verified from the live Unsloth GGUF repo metadata.

| Format | Size | Fits 192 GiB VRAM |
|---|---:|---|
| Official BF16 safetensors (`Qwen/Qwen3-Coder-Next`) | about 148.4 GiB | Yes, with limited KV headroom |
| Unsloth GGUF `Q8_0` (3-shard split) | 78.99 GiB | Yes, comfortable |
| Unsloth GGUF `Q4_K_M` (single file) | 45.20 GiB | Yes, large KV headroom |

### Turing (sm_75) numerical note

`config.json` declares `bfloat16`, but Quadro RTX 8000 (Turing, sm_75) has no
BF16, TF32, FP8, or FP4 tensor-core execution and no FlashAttention-2. Run
unquantized weights as **FP16** (`--dtype half` in vLLM) and prefer weight-only
quantization (GGUF `Q4_K_M`/`Q8_0`) that ships an sm_75 kernel. See the
[hardware profile](../../hardware/4x-rtx8000/) for the full support matrix.

## Profiles

| Profile | Backend | Parallelism | Purpose |
|---|---|---|---|
| `llama-cpp-q4km` (default) | llama.cpp | layer split x4 | Fastest single-request decode / lowest latency |
| `vllm-fp16-tp2pp2` | vLLM FP16 | TP2 x PP2 (NVLink-pair aware) | Highest aggregate serving throughput / prefill |
| `vllm-fp16-tp4` | vLLM FP16 | TP4 | Original topology-agnostic FP16 baseline |

### Fastest single request (default)

```bash
./scripts/download-model.sh qwen3-coder-next-80b-a3b llama-cpp-q4km
./scripts/build-llama-cpp.sh            # sm_75 build; needs llama.cpp >= b7186 for qwen3_next
./scripts/serve-model.sh qwen3-coder-next-80b-a3b llama-cpp-q4km
```

The default uses the [Unsloth GGUF distribution](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF)
`Qwen3-Coder-Next-Q4_K_M.gguf` (imatrix-quantized, **45.20 GiB, single file** —
no `gguf-split` merge needed). It is the strongest candidate for the lowest
single-request latency and fastest decode on sm_75. Qwen3-Next / DeltaNet
support landed upstream in llama.cpp around release `b7186`; the reused DeepSeek
build is `b10298`, which is newer, but must still pass a real
load-and-generate test before results are trusted.

#### GGUF distribution and quant selection

File names and sizes below are verified from the live Unsloth repo metadata.

| Quant | File(s) | Size | Notes |
|---|---|---:|---|
| `Q4_K_M` (default) | `Qwen3-Coder-Next-Q4_K_M.gguf` | 45.20 GiB | Single file; best speed/quality tradeoff on sm_75 |
| `Q4_K_S` | `Qwen3-Coder-Next-Q4_K_S.gguf` | 42.40 GiB | Single file; marginally smaller/faster, lower quality |
| `Q4_0` | `Qwen3-Coder-Next-Q4_0.gguf` | 42.22 GiB | Single file; legacy quant, rarely faster than K-quant MMQ |
| `IQ4_XS` | `Qwen3-Coder-Next-IQ4_XS.gguf` | 39.75 GiB | Single file; smallest 4-bit, but IQ dequant can be slower on GPU |
| `UD-Q4_K_XL` | `Qwen3-Coder-Next-UD-Q4_K_XL.gguf` | 46.20 GiB | Single file; Unsloth Dynamic, quality upgrade at ~same speed |
| `Q8_0` | `Q8_0/…-00001..00003-of-00003.gguf` | 78.99 GiB | **3-shard split**; download all shards (first shard auto-loads the set) |
| `BF16` | `BF16/…-00001..00004-of-00004.gguf` | 148.51 GiB | **4-shard split**; reference precision, not for sm_75 speed |

`Q4_K_M` is kept as the default because none of the smaller 4-bit options is
reliably faster on sm_75: `IQ4_XS`/`Q4_0` trade K-quant MMQ kernel efficiency or
quality for a small size reduction. Switch to `UD-Q4_K_XL` for quality, or to a
split quant (`Q8_0`) for accuracy, by copying the profile and changing `MODEL`,
`HF_REPO` stays the same. For split quants, point `MODEL` at the
`-00001-of-000NN.gguf` shard and set `DOWNLOAD_PATTERNS` to the subdirectory
glob (e.g. `Q8_0/*`); llama.cpp loads the remaining shards automatically.

### Highest throughput / prefill

```bash
./scripts/download-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp2pp2
./scripts/serve-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp2pp2
```

`TP2 x PP2` maps each tensor-parallel group onto one NVLink pair (0-1 and 2-3)
and routes only pipeline activations across the slow cross-pair PCIe path, which
is expected to beat topology-agnostic `TP4` for aggregate prefill/decode
throughput. The official deployment minimum is vLLM 0.15.0 or SGLang 0.5.8; the
default `vllm` environment (0.17.1) satisfies this, but Turing kernel
compatibility for `qwen3_next` must still be validated.

## Operating the target

All four operations reuse the generic repository scripts; there are no
target-specific wrappers. Each resolves the profile through
`scripts/lib/common.sh` (hardware profile -> `target.env` -> profile).

Serve (launches the correct backend with the profile's settings):

```bash
./scripts/serve-model.sh qwen3-coder-next-80b-a3b llama-cpp-q4km      # llama.cpp
./scripts/serve-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp2pp2    # vLLM TP2xPP2
```

Benchmark (OpenAI `/v1/chat/completions`, served by both backends):

```bash
# Single-request decode latency (latency-oriented profile)
./scripts/benchmark-model.sh qwen3-coder-next-80b-a3b llama-cpp-q4km
# Prefill-oriented comparison: long prompt, one generated token
./scripts/benchmark-model.sh qwen3-coder-next-80b-a3b llama-cpp-q4km -- --prompt-file PROMPT.txt --max-tokens 1
# Aggregate throughput under concurrency (throughput-oriented profile)
./scripts/benchmark-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp2pp2 -- --concurrency 16 --requests 64
```

`end_to_end_prompt_tokens_per_second` includes HTTP and the one-token decode.
Use a long prompt with `--max-tokens 1` to make it a stable prefill-oriented
comparison; use backend-native metrics when an exact kernel-only prefill rate is
required.

Point coding agents at the local endpoint:

```bash
# GitHub Copilot CLI -> local OpenAI-compatible endpoint (works with either backend)
./scripts/copilot-local.sh qwen3-coder-next-80b-a3b llama-cpp-q4km
# Claude Code -> local Anthropic-compatible endpoint (llama.cpp only)
./scripts/claude-local.sh qwen3-coder-next-80b-a3b llama-cpp-q4km
```

`claude-local.sh` requires a llama.cpp profile because vLLM does not serve the
Anthropic API; use it with `llama-cpp-q4km` (the target default). `copilot-local.sh`
uses the OpenAI-compatible `/v1` endpoint and therefore works with either
backend. Both wrappers health-check the server first and export the correct
model alias (`qwen3-coder-next`) and base URL.

## Status and blockers

- **Local BF16 download is incomplete.** Shards `00003` and `00039` of 40 are
  missing from `/home/algore/models/qwen3-coder-next-80b-a3b`, so the vLLM
  profiles cannot load until the download is completed
  (`./scripts/download-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp4`).
- The `llama-cpp-q4km` profile fetches a single-file GGUF into a **separate**
  directory (`…-gguf/Qwen3-Coder-Next-Q4_K_M.gguf`) and does not depend on those
  safetensors shards.
- All performance numbers in [PERFORMANCE.md](PERFORMANCE.md) are **projected
  planning targets**, not measured RTX 8000 results.
