# Qwen3.6-35B-A3B

Official checkpoint: [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).

Qwen3.6 is a multimodal MoE with 35B total / 3B active parameters, 40 layers,
256 experts (8 routed plus 1 shared), and 262,144-token native context. Its
language stack uses 30 Gated DeltaNet layers and 10 full-attention layers. The
RTX 8000 lacks BF16, FP8, and FP4 tensor-core execution, so the profiles use
FP16 for vLLM and a CUDA-friendly GGUF K-quant for llama.cpp.

## Profiles

| Profile | GPUs | Use |
|---|---|---|
| `llama-cpp-q4km-mtp-1gpu` (default) | GPU 2 | Lowest interactive latency; 21.10 GiB Q4_K_M plus two-token in-model MTP |
| `llama-cpp-q4km-1gpu` | GPU 2 | Identical no-MTP A/B baseline |
| `vllm-fp16-tp2` | GPUs 2-3 | Text-only, expert-parallel batched throughput on one NVLink pair |

The single-GPU profile is deliberately the default. The quantized model fits
entirely on one 48 GiB card, so adding GPUs would add synchronization without
reducing the bytes read by that card during serial decode. GPU 2 also allows it
to coexist with the repository's Qwen3-Coder default on GPUs 0-1.

```bash
./scripts/download-model.sh qwen3.6-35b-a3b
./scripts/benchmark-native.sh qwen3.6-35b-a3b llama-cpp-q4km-1gpu
./scripts/serve-model.sh qwen3.6-35b-a3b
./scripts/benchmark-model.sh qwen3.6-35b-a3b -- --max-tokens 512
```

For the MTP A/B comparison, run the same prompt and output length against each
llama.cpp profile after restarting the server. Use at least 512 output tokens:
short generations make startup and prompt evaluation dominate.

```bash
./scripts/serve-model.sh qwen3.6-35b-a3b llama-cpp-q4km-1gpu
./scripts/benchmark-model.sh qwen3.6-35b-a3b llama-cpp-q4km-1gpu -- --max-tokens 512
```

The vLLM profile requires a runtime with Qwen3.6 support. It strips the vision
encoder, enables MoE expert parallelism and automatic prefix caching, and keeps
the tensor-parallel group inside NVLink pair 2-3:

```bash
./scripts/download-model.sh qwen3.6-35b-a3b vllm-fp16-tp2
./scripts/serve-model.sh qwen3.6-35b-a3b vllm-fp16-tp2
./scripts/benchmark-model.sh qwen3.6-35b-a3b vllm-fp16-tp2 -- --concurrency 16 --requests 64
```
