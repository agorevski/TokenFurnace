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
| [Qwen3-Coder-Next 80B-A3B](targets/qwen3-coder-next-80b-a3b/) | llama.cpp Q4_K_M (default) / vLLM FP16 | Architecture defined, targets projected, not yet measured |

## Layout

```text
hardware/<profile>/       Workstation topology, firmware, and runtime support
targets/<model>/          Model metadata, profiles, and performance results
scripts/backends/         Runtime-specific launch implementations
scripts/serve-model.sh    Generic target/profile launcher
scripts/download-model.sh Generic Hugging Face downloader
scripts/benchmark-model.sh OpenAI-compatible API benchmark
```

## Usage

DeepSeek balanced profile:

```bash
./scripts/serve-model.sh deepseek-v4-flash-0731 q4-balanced
```

Qwen fastest single-request profile (llama.cpp `Q4_K_M`, the default):

```bash
./scripts/download-model.sh qwen3-coder-next-80b-a3b llama-cpp-q4km
./scripts/serve-model.sh qwen3-coder-next-80b-a3b llama-cpp-q4km
```

Qwen highest-throughput profile (vLLM FP16, NVLink-pair-aware `TP2 x PP2`):

```bash
./scripts/download-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp2pp2
./scripts/serve-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp2pp2
```

Benchmark a running target (single-request latency, or aggregate throughput):

```bash
./scripts/benchmark-model.sh qwen3-coder-next-80b-a3b llama-cpp-q4km
./scripts/benchmark-model.sh qwen3-coder-next-80b-a3b llama-cpp-q4km -- --prompt-file PROMPT.txt --max-tokens 1
./scripts/benchmark-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp2pp2 -- --concurrency 16 --requests 64
```

The prompt-rate fields are end-to-end API rates. With `--max-tokens 1` and a
long prompt they are useful prefill-oriented comparisons, but they are not
server-internal kernel timings.

Point coding agents at the local server:

```bash
./scripts/copilot-local.sh qwen3-coder-next-80b-a3b llama-cpp-q4km   # OpenAI /v1 (either backend)
./scripts/claude-local.sh  qwen3-coder-next-80b-a3b llama-cpp-q4km   # Anthropic API (llama.cpp)
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
