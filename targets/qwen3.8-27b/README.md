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
| `llama-cpp-q4km-2gpu-tensor` (default) | NVLink pair 2-3 | Fastest measured prefill and decode |
| `llama-cpp-q4km-1gpu` | GPU 3 | Lower resource use; leaves GPU 2 free |
| `llama-cpp-q4km-2gpu` | NVLink pair 2-3 | Layer-split baseline |
| `vllm-official-fp16-tp2` | NVLink pair 2-3 | Official unquantized safetensors; BF16 cast to native Turing FP16 |
| `vllm-official-fp16-tp2-mtp1` | NVLink pair 2-3 | Fastest official-weight latency; built-in MTP-1 |

The 17,106,773,984-byte Q4_K_M file is only 15.93 GiB and fits comfortably on
one 48 GiB RTX 8000. Despite that, tensor parallelism across NVLink pair 2-3 is
the measured winner: 40.37 tok/s native decode and 1113.88 tok/s at 4096-token
prefill, versus 28.10 and 659.75 tok/s on one GPU. See
[PERFORMANCE.md](PERFORMANCE.md) for the complete sweep.

The requested Unsloth repository did not expose a separate MTP draft GGUF when
this target was created. The GGUF profiles therefore do not silently combine
artifacts from another publisher. The official checkpoint does include its own
MTP head, exposed by the separate `vllm-official-fp16-tp2-mtp1` profile.

Row split was also tested and failed to load on this llama.cpp build, so no
nonfunctional row profile is shipped.

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
