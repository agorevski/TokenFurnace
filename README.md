# RTX 8000 LLM Lab

Reproducible model-serving and performance experiments for a workstation with:

- 4x NVIDIA Quadro RTX 8000, 48 GiB each
- NVLink pairs between GPUs 0-1 and 2-3
- 192 GiB aggregate VRAM
- Intel Xeon W-2295 and 256 GiB system RAM

The repository is organized by hardware, model target, runtime backend, and
tuning profile rather than around one model.

## Targets

| Target | Backend | Status |
|---|---|---|
| [DeepSeek V4 Flash 0731](targets/deepseek-v4-flash-0731/) | llama.cpp + DSpark | Optimized and measured |
| [Qwen3-Coder-Next 80B-A3B](targets/qwen3-coder-next-80b-a3b/) | llama.cpp Q4_K_M 2-GPU (default) / vLLM FP16 | llama.cpp Q4_K_M measured; vLLM FP16 projected |
| [Qwen3.6-35B-A3B](targets/qwen3.6-35b-a3b/) | llama.cpp Q4_K_M + MTP (default) / vLLM FP16 | llama.cpp measured; vLLM pending |
| [Qwen3.8-27B](targets/qwen3.8-27b/) | llama.cpp Q4_K_M, 2-GPU tensor split | Optimized and measured |

## Layout

```text
hardware/<profile>/       Workstation topology, firmware, and runtime support
targets/<model>/          Model metadata, profiles, and performance results
scripts/backends/         Runtime-specific launch implementations
scripts/serve-model.sh    Generic target/profile launcher
scripts/download-model.sh Generic Hugging Face downloader
scripts/benchmark-model.sh OpenAI-compatible API benchmark
scripts/benchmark-native.sh llama.cpp prefill/decode kernel benchmark
```

## Usage

DeepSeek balanced profile:

```bash
./scripts/serve-model.sh deepseek-v4-flash-0731 q4-balanced
```

Qwen fastest single-request decode (llama.cpp `Q4_K_M`, 2-GPU NVLink pair — the
default):

```bash
./scripts/download-model.sh qwen3-coder-next-80b-a3b llama-cpp-q4km-2gpu
./scripts/serve-model.sh qwen3-coder-next-80b-a3b            # default = llama-cpp-q4km-2gpu (GPUs 0,1)
```

Qwen explicit 4-GPU long-prefill throughput profile (llama.cpp `Q4_K_M`):

```bash
./scripts/serve-model.sh qwen3-coder-next-80b-a3b llama-cpp-q4km
```

Qwen highest-throughput profile (vLLM FP16, NVLink-pair-aware `TP2 x PP2`):

```bash
./scripts/download-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp2pp2
./scripts/serve-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp2pp2
```

Qwen3.6 fastest interactive profile (single GPU, Q4_K_M, MTP):

```bash
./scripts/download-model.sh qwen3.6-35b-a3b
./scripts/serve-model.sh qwen3.6-35b-a3b
```

Qwen3.8 fastest profile (2-GPU NVLink tensor split, Unsloth Q4_K_M):

```bash
./scripts/download-model.sh qwen3.8-27b
./scripts/benchmark-native.sh qwen3.8-27b
./scripts/serve-model.sh qwen3.8-27b
```

Benchmark a running target (single-request latency, or aggregate throughput):

```bash
./scripts/benchmark-model.sh qwen3-coder-next-80b-a3b
./scripts/benchmark-model.sh qwen3-coder-next-80b-a3b llama-cpp-q4km -- --prompt-file PROMPT.txt --max-tokens 1
./scripts/benchmark-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp2pp2 -- --concurrency 16 --requests 64
```

Run the standard native llama.cpp sweep without starting a server:

```bash
./scripts/benchmark-native.sh qwen3.6-35b-a3b llama-cpp-q4km-1gpu
```

The prompt-rate fields are end-to-end API rates. With `--max-tokens 1` and a
long prompt they are useful prefill-oriented comparisons, but they are not
server-internal kernel timings.

Point coding agents at the local server (both wrappers default to the Qwen
target and its 2-GPU default profile on port 8090):

```bash
./scripts/copilot-local.sh   # OpenAI /v1, Qwen default (either backend)
./scripts/claude-local.sh    # Anthropic API, Qwen default (llama.cpp)
```

Inspect status:

```bash
./scripts/status.sh deepseek-v4-flash-0731
./scripts/status.sh qwen3-coder-next-80b-a3b
```

Downloaded model weights remain outside Git under `/home/algore/models`.

See the [four-RTX-8000 hardware profile](hardware/4x-rtx8000/) for the
firmware snapshot, installed runtime versions, numerical support, and backend
compatibility constraints.
