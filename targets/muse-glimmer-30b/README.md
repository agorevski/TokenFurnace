# Muse Glimmer 30B

[`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B)
is a 29.6B-parameter dense multimodal model. This text-inference target uses
the official 19.7 GB dynamic
[`Q4_K_XL` GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF),
which has 0.2% average reported benchmark degradation from full precision.

Muse Glimmer requires llama.cpp build `b10353` or newer. The profile uses one
RTX 8000 and optionally loads the official DFlash drafter for speculative
decoding. The perception encoder is not downloaded because this target
benchmarks text generation.

```bash
./scripts/download-model.sh muse-glimmer-30b
./scripts/benchmark-native.sh muse-glimmer-30b llama-cpp-q4kxl-1gpu
./scripts/serve-model.sh muse-glimmer-30b
./scripts/benchmark-model.sh muse-glimmer-30b -- --max-tokens 512
```

See [PERFORMANCE.md](PERFORMANCE.md) for baseline and DFlash measurements.
