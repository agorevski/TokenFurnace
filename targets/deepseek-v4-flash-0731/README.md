# DeepSeek V4 Flash 0731

This target uses llama.cpp with CUDA, NCCL, and optional DSpark speculative
decoding.

```bash
./scripts/download-model.sh deepseek-v4-flash-0731 q4-balanced
./scripts/build-llama-cpp.sh
./scripts/serve-model.sh deepseek-v4-flash-0731 q4-balanced
```

Available profiles:

- `q4-balanced`: best general-purpose profile found so far
- `q4-predictable`: deeper drafts for repetitive or rigid output
- `q4-long-context`: confidence-gated drafts for long prompts
- `q4-plain`: Q4 without DSpark
- `q8-lossless`: lossless quantization with DSpark
- `iq3`: smaller experimental quantization

See [PERFORMANCE.md](PERFORMANCE.md) for measurements.
