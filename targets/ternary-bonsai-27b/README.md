# Ternary Bonsai 27B

[`prism-ml/Ternary-Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)
is a ternary quantization of Qwen3.6-27B. This target uses the publisher's
7.2 GB `PQ2_0` group-128 pack and custom
[`PrismML-Eng/llama.cpp`](https://github.com/PrismML-Eng/llama.cpp) kernels.

The initial text-only profile uses one RTX 8000 and a practical 65,536-token
context. The optional vision projector and DSpark drafter are excluded from
the baseline so target-model throughput is measured independently.

```bash
./scripts/download-model.sh ternary-bonsai-27b
./scripts/benchmark-native.sh ternary-bonsai-27b
./scripts/serve-model.sh ternary-bonsai-27b
./scripts/benchmark-model.sh ternary-bonsai-27b -- --max-tokens 512
```

See [PERFORMANCE.md](PERFORMANCE.md) for measured results.
