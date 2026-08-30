# Laguna S 2.1

[`poolside/Laguna-S-2.1`](https://huggingface.co/poolside/Laguna-S-2.1)
is a 118B-parameter sparse model with 8B active parameters per token. This
target uses Poolside's official
[`Q4_K_M` GGUF](https://huggingface.co/poolside/Laguna-S-2.1-GGUF) across all
four GPUs. The downloaded file is 96.03 GB despite the model card listing
68 GB.

Laguna support is not yet in upstream llama.cpp. The target uses Poolside's
[`laguna` branch](https://github.com/poolsideai/llama.cpp/tree/laguna), which
also supports the official DFlash speculative-decoding model.

```bash
./scripts/download-model.sh laguna-s-2.1
./scripts/benchmark-native.sh laguna-s-2.1 llama-cpp-q4km-4gpu
./scripts/serve-model.sh laguna-s-2.1
./scripts/benchmark-model.sh laguna-s-2.1 -- --max-tokens 512
```

See [PERFORMANCE.md](PERFORMANCE.md) for baseline and DFlash measurements.
