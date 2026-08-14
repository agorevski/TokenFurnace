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

The 17,106,773,984-byte Q4_K_M file is only 15.93 GiB and fits comfortably on
one 48 GiB RTX 8000. Despite that, tensor parallelism across NVLink pair 2-3 is
the measured winner: 40.37 tok/s native decode and 1113.88 tok/s at 4096-token
prefill, versus 28.10 and 659.75 tok/s on one GPU. See
[PERFORMANCE.md](PERFORMANCE.md) for the complete sweep.

The requested Unsloth repository did not expose a separate MTP draft GGUF when
this target was created. The profiles therefore do not silently combine
artifacts from another publisher. MTP remains disabled until a compatible head
appears in `unsloth/Qwen3.8-27B-GGUF` and passes a correctness test.

Row split was also tested and failed to load on this llama.cpp build, so no
nonfunctional row profile is shipped.

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
