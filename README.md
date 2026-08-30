# RTX 8000 LLM Lab

This repository is a hands-on lab for performance tuning and benchmarking
Hugging Face models on a four-GPU NVIDIA Quadro RTX 8000 workstation. It
captures the model artifacts, runtime builds, launch profiles, topology
experiments, and measured prefill/decode performance needed to reproduce each
result.

The workstation has:

- 4x NVIDIA Quadro RTX 8000, 48 GiB each
- NVLink pairs between GPUs 0-1 and 2-3
- 192 GiB aggregate VRAM
- Intel Xeon W-2295 and 256 GiB system RAM

The repository is organized by hardware, model target, runtime backend, and
tuning profile rather than around one model. Model weights are downloaded
separately and are not committed to Git.

## What's in this repository

- **Hardware documentation** records the GPU topology, NVLink pairings,
  firmware, drivers, CUDA capabilities, and installed inference runtimes.
- **Model targets** define Hugging Face sources, quantizations, GPU placement,
  runtime-specific settings, and recommended serving profiles.
- **Benchmark reports** preserve measured native llama.cpp and
  OpenAI-compatible API results, including unsuccessful configurations where
  they help explain the selected defaults.
- **Automation scripts** download model artifacts, build runtime variants,
  launch servers, run repeatable benchmarks, inspect status, and connect local
  models to coding agents.

## Documentation

Start with the [four-RTX-8000 hardware profile](hardware/4x-rtx8000/README.md)
for the constraints shared by every experiment. Each model then has a target
guide describing its artifacts and profiles, plus a separate performance
report containing measurements and conclusions.

| Model | Target guide | Performance report | Backend and status |
|---|---|---|---|
| DeepSeek V4 Flash 0731 | [Setup and profiles](targets/deepseek-v4-flash-0731/README.md) | [Q4 and DSpark results](targets/deepseek-v4-flash-0731/PERFORMANCE.md) | llama.cpp + DSpark; optimized and measured |
| Qwen3-Coder-Next 80B-A3B | [Setup and profiles](targets/qwen3-coder-next-80b-a3b/README.md) | [Measured and projected results](targets/qwen3-coder-next-80b-a3b/PERFORMANCE.md) | llama.cpp Q4_K_M measured; vLLM FP16 projected |
| Qwen3.6-35B-A3B | [Setup and profiles](targets/qwen3.6-35b-a3b/README.md) | [Q4_K_M and MTP results](targets/qwen3.6-35b-a3b/PERFORMANCE.md) | llama.cpp measured; vLLM pending |
| Qwen3.8-27B | [Setup and profiles](targets/qwen3.8-27b/README.md) | [Topology and runtime results](targets/qwen3.8-27b/PERFORMANCE.md) | llama.cpp Q4_K_M and official FP16 vLLM; optimized and measured |
| Qwen3.8-27B Uncensored OrcaRouter | [Setup and profiles](targets/qwen3.8-27b-orcarouter/README.md) | [Tensor split and MTP results](targets/qwen3.8-27b-orcarouter/PERFORMANCE.md) | llama.cpp Q4_K_M + MTP-2; optimized and measured |
| Qwen3.8-Flash-Next | [Setup and profiles](targets/qwen3.8-flash-next/README.md) | [Two- and four-GPU results](targets/qwen3.8-flash-next/PERFORMANCE.md) | llama.cpp Unsloth Q3_K_XL; optimized and measured |
| GLM-5.3-Flash | [Setup and profiles](targets/glm-5.3-flash/README.md) | [Four-GPU results](targets/glm-5.3-flash/PERFORMANCE.md) | llama.cpp Unsloth UD-Q3_K_XL; measured |
| Laguna S 2.1 | [Setup and profiles](targets/laguna-s-2.1/README.md) | [Baseline and DFlash results](targets/laguna-s-2.1/PERFORMANCE.md) | llama.cpp Q4_K_M; optimized and measured |
| Muse Glimmer 30B | [Setup and profiles](targets/muse-glimmer-30b/README.md) | [Baseline and DFlash results](targets/muse-glimmer-30b/PERFORMANCE.md) | llama.cpp dynamic Q4_K_XL; optimized and measured |

## Repository layout

```text
hardware/<profile>/         Workstation topology, firmware, and runtime support
targets/<model>/README.md   Model source, artifacts, profiles, and usage
targets/<model>/PERFORMANCE.md
                            Benchmark conditions, results, and conclusions
targets/<model>/profiles/   Reproducible runtime configuration
scripts/backends/           Runtime-specific launch implementations
scripts/serve-model.sh      Generic target/profile launcher
scripts/download-model.sh   Generic Hugging Face downloader
scripts/benchmark-model.sh  OpenAI-compatible API benchmark
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

Qwen3.8 official unquantized checkpoint (official BF16 safetensors cast to
native Turing FP16 on load):

```bash
./scripts/download-model.sh qwen3.8-27b vllm-official-fp16-tp2
./scripts/serve-model.sh qwen3.8-27b vllm-official-fp16-tp2-mtp1
```

GLM-5.3-Flash UD-Q3_K_XL on all four GPUs:

```bash
./scripts/download-model.sh glm-5.3-flash
./scripts/serve-model.sh glm-5.3-flash
```

Laguna S 2.1 Q4_K_M on all four GPUs:

```bash
./scripts/download-model.sh laguna-s-2.1
./scripts/serve-model.sh laguna-s-2.1
```

Muse Glimmer 30B dynamic Q4_K_XL with DFlash on one GPU:

```bash
./scripts/download-model.sh muse-glimmer-30b
./scripts/serve-model.sh muse-glimmer-30b
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
