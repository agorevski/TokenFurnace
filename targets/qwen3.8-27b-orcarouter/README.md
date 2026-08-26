# Qwen3.8-27B Uncensored OrcaRouter

GGUF distribution:
[`chimingw/Qwen3.8-27B-Uncensored-OrcaRouter-GGUF`](https://huggingface.co/chimingw/Qwen3.8-27B-Uncensored-OrcaRouter-GGUF).
The model is derived from
[`orcarouter/Qwen3.8-27B-Uncensored-FP8`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8).

## Profiles

| Profile | GPUs | Goal |
|---|---|---|
| `llama-cpp-q4km-2gpu-tensor-mtp2` (default) | NVLink pair 2-3 | Fastest measured interactive decode |
| `llama-cpp-q4km-2gpu-tensor` | NVLink pair 2-3 | Non-speculative tensor baseline |
| `llama-cpp-q4km-1gpu` | GPU 3 | Single-device baseline |
| `llama-cpp-q4km-2gpu` | NVLink pair 2-3 | Layer-split baseline |

All profiles use the publisher's Q4_K_M GGUF. This avoids FP8 execution on the
Turing GPUs while retaining the model derived from the FP8 checkpoint. The
main GGUF includes its matching MTP prediction head; MTP-2 improves measured
end-to-end output throughput from 42.62 to 67.73 tok/s.

```bash
./scripts/download-model.sh qwen3.8-27b-orcarouter
./scripts/benchmark-native.sh qwen3.8-27b-orcarouter
./scripts/serve-model.sh qwen3.8-27b-orcarouter
./scripts/benchmark-model.sh qwen3.8-27b-orcarouter -- --max-tokens 512
```
