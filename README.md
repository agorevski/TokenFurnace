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
| [Qwen3-Coder-Next 80B-A3B](targets/qwen3-coder-next-80b-a3b/) | vLLM baseline | Initial bring-up |

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

Qwen initial vLLM profile:

```bash
./scripts/download-model.sh qwen3-coder-next-80b-a3b
./scripts/serve-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp4
```

Benchmark a running target:

```bash
./scripts/benchmark-model.sh qwen3-coder-next-80b-a3b
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
