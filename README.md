# DeepSeek V4 Flash 0731 on 4x RTX 8000

Reproducible local inference setup for DeepSeek-V4-Flash-0731 on:

- 4x NVIDIA Quadro RTX 8000, 48 GB each
- Turing compute capability 7.5
- Two NVLink pairs: GPU 0-1 and GPU 2-3
- 192 GB total VRAM
- 256 GB system RAM

The scripts download Unsloth GGUFs, build a CUDA/NCCL-enabled llama.cpp, serve
the model with DSpark speculative decoding, and launch Claude Code or GitHub
Copilot CLI against the local server.

## Measured results

Measurements vary with prompt length and DSpark draft acceptance.

| Configuration | Prefill | Decode |
|---|---:|---:|
| Historical Q4 defaults | about 352-411 tok/s | 23.5-24.0 tok/s |
| Optimized Q4, synthetic 8K prompt | 736.8 tok/s | 24.0 tok/s without DSpark |
| Optimized Q4, real 21.6K server prompt | 463.9 tok/s | 28.2 tok/s with DSpark |
| Optimized Q4, short Claude Code request | 460.6 tok/s | 36.8 tok/s with DSpark |
| Q8 with DSpark, controlled 512-token output | - | 21.5 tok/s |

The 30-37 tok/s results are not a stable plain-GGUF baseline. They occur when
DSpark gets high draft acceptance. Plain Q4 decode is about 24 tok/s.

## Quick start

```bash
cd ~/GIT/agorevski/DeepSeekRTX8000

# Download the fast Q4 model and DSpark drafter.
./scripts/download-models.sh q4

# Build current llama.cpp for Turing SM 7.5.
./scripts/build-llama.sh

# Serve optimized Q4 + DSpark on http://127.0.0.1:8090.
./scripts/serve-q4.sh
```

In another terminal:

```bash
./scripts/claude-local.sh
./scripts/copilot-local.sh
```

## Scripts

| Script | Purpose |
|---|---|
| `download-models.sh` | Download `q4`, `q8`, `iq3`, `draft`, or `all` GGUFs |
| `build-llama.sh` | Clone/update and build llama.cpp with CUDA 12, NCCL, and SM 75 |
| `serve.sh` | Configurable server launcher |
| `serve-q4.sh` | Fast default: UD-IQ4_XS + DSpark |
| `serve-q8.sh` | Lossless UD-Q8_K_XL + DSpark |
| `claude-local.sh` | Launch Claude Code through llama-server's Anthropic endpoint |
| `copilot-local.sh` | Launch Copilot CLI through its supported BYOK provider mode |
| `benchmark.sh` | Run repeatable llama-bench prefill/decode tests |
| `status.sh` | Show health, slots, GPU memory, and server process |

Run any main script with `--help` for usage.

See [PERFORMANCE_TESTS.md](PERFORMANCE_TESTS.md) for the controlled Q4 DSpark
draft-depth and confidence-threshold experiments.

## Model profiles

### Q4 speed profile

`UD-IQ4_XS` is about 127 GiB and leaves enough VRAM for large prompt
microbatches and the 10 GiB DSpark drafter. This is the default.

```bash
./scripts/serve-q4.sh
```

Defaults:

- context: 65,536
- parallel slots: 1
- logical batch: 8,192
- physical microbatch: 2,048
- DSpark draft maximum: 2

### Q8 lossless profile

`UD-Q8_K_XL` is about 151 GiB on disk and reproduces the original weights.
It leaves much less VRAM headroom, so it uses conservative batch settings.

```bash
./scripts/serve-q8.sh
```

Defaults:

- context: 65,536
- parallel slots: 1
- logical batch: 2,048
- physical microbatch: 512
- DSpark draft maximum: 2

## Important tuning findings

### Settings that helped

- `CUDA_SCALE_LAUNCH_QUEUES=4x` was the largest prefill improvement.
- `--batch-size 8192 --ubatch-size 2048` was best for Q4 8K prefill.
- `--split-mode layer` was stable and fast.
- llama.cpp CUDA Flash Attention works on these Turing GPUs and should stay on.
- Default MMQ kernels outperformed forced cuBLAS.
- `--parallel 1` maximizes single-session throughput.
- DSpark `--spec-draft-n-max 2` performed better than the documented default 3
  as a balanced setting across the tested workloads.
- Deeper drafts can be faster for predictable output. The best synthetic result
  was 43.8 tok/s with draft maximum 5 and confidence threshold 0.40, but that
  profile was slower than the default on natural prose.

### Settings that did not help

- Forced P2P did not materially change throughput.
- Raising `GGML_CUDA_PEER_MAX_BATCH_SIZE` from 128 to 2048 changed performance
  by less than 1%, though the build keeps 2048 enabled.
- Forced cuBLAS reduced 8K prefill from about 737 to 383 tok/s and slightly
  reduced decode.
- `--split-mode row` and `--split-mode tensor` failed to load this model.
- DSpark with a three-token draft had only 37.7% acceptance in one controlled
  test and was slower than plain Q8.

### DSpark behavior

DSpark speed depends on how many drafted tokens the target model accepts.

| Test | Draft acceptance | Decode |
|---|---:|---:|
| Q8, draft max 3 | 37.7% | 18.3 tok/s |
| Q8, no DSpark | - | 19.0 tok/s |
| Q8, draft max 2 | 53.6% | 21.5 tok/s |
| Q4, draft max 2 | 48.4% | 25.9 tok/s |
| Q4, predictable 21K prompt output | 76.0% | 28.2 tok/s |
| Q4, short Claude Code output | 83.3% | 36.8 tok/s |

The advertised 1.5-1.9x gain is an "up to" result, usually measured on much
newer GPUs and favorable outputs.

## Agent prompt caching

Prompt caching is critical for coding agents because their system prompts and
conversation histories are large.

- Keep the server running; restarting discards in-memory caches.
- Reuse the same working directory so the system-prompt prefix stays stable.
- Avoid clearing or compacting sessions unless necessary.
- Set `PARALLEL` to at least the number of simultaneously warm conversations.
  Increasing it reduces per-session context and simultaneous decode speed.

Inspect slots with:

```bash
./scripts/status.sh
```

## Environment overrides

Copy `config.env.example` to `config.env` and edit it, or export variables
before launching a script.

Examples:

```bash
CTX_SIZE=131072 PARALLEL=2 ./scripts/serve-q4.sh
SERVER_PORT=8091 ./scripts/serve-q8.sh
DSPARK=0 ./scripts/serve-q4.sh
SPEC_DRAFT_N_MAX=5 SPEC_DRAFT_P_MIN=0.40 ./scripts/serve-q4.sh
```

## Further speed options

1. Test `UD-IQ3_XXS` (about 103 GB). It should reduce weight bandwidth but
   trades away quality.
2. Disable model thinking for short operational tasks to reduce total generated
   tokens. This improves elapsed time, not tokens per second.
3. Keep prompt caches warm. This is often a larger agent-latency improvement
   than another small decode optimization.
4. Move to newer GPUs for a major decode increase. Turing lacks the newer
   low-precision execution paths used in published B200/Hopper results.

Do not use quantized KV cache for this model until the current llama.cpp
DeepSeek-V4 quantized-KV correctness issue is resolved.

## Sources

- [Unsloth DeepSeek V4 guide](https://unsloth.ai/docs/models/deepseek-v4)
- [Unsloth GGUF repository](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [DSpark llama.cpp integration](https://github.com/ggml-org/llama.cpp/pull/25784)
