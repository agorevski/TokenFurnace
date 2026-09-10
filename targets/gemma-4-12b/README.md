# Gemma 4 12B

[`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) is a
12B-parameter multimodal model. This target uses Unsloth's
[`Q4_K_M` GGUF and F16 vision projector](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF)
on one RTX 8000.

The baseline profile measures text generation at a 65,536-token context.
Vision is tested separately with the downloaded `mmproj-F16.gguf` so text and
multimodal behavior are both validated without conflating their throughput.

```bash
./scripts/download-model.sh gemma-4-12b
./scripts/benchmark-native.sh gemma-4-12b
./scripts/serve-model.sh gemma-4-12b
./scripts/benchmark-model.sh gemma-4-12b -- --max-tokens 512
```

See [PERFORMANCE.md](PERFORMANCE.md) for measured results.
