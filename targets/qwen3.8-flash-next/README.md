# Qwen3.8-Flash-Next

Official model:
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
The llama.cpp profiles use its
[`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
distribution because RTX 8000 GPUs do not support the official checkpoint's
native BF16 execution.

The selected `UD-Q3_K_XL` quant is 90.0 GB and retains 90.4% same-top-token
agreement in Unsloth's published quantization analysis. It fits on one 96 GiB
NVLink pair at the default 64K context, avoiding four-GPU collectives across
the workstation's PCIe-separated NVLink pairs.

This model and its external MTP draft head currently require
[llama.cpp PR 28243](https://github.com/ggml-org/llama.cpp/pull/28243). The
profiles therefore use an isolated checkout and `build-sm75` build rather than
the repository's stable llama.cpp build.

The optional `llama-cpp-q3xl-2gpu-mtp-q8` profile pairs the Q3 main-model GGUF
with Unsloth's shared Q8_0 MTP GGUF. It improves measured single-request decode
throughput by 45.7%, with about 79.6% draft acceptance. Fixed-seed output was
valid but not byte-identical to the no-MTP result, so the no-MTP profile remains
the behavior-safe default. Tensor split also aborts in the required PR build;
both supported profiles use the measured two-GPU layer split.

```bash
./scripts/download-model.sh qwen3.8-flash-next
./scripts/benchmark-native.sh qwen3.8-flash-next
./scripts/serve-model.sh qwen3.8-flash-next
./scripts/benchmark-model.sh qwen3.8-flash-next -- --max-tokens 512

# Optional shared-Q8 MTP path
./scripts/download-model.sh qwen3.8-flash-next llama-cpp-q3xl-2gpu-mtp-q8
./scripts/serve-model.sh qwen3.8-flash-next llama-cpp-q3xl-2gpu-mtp-q8
./scripts/benchmark-model.sh qwen3.8-flash-next \
  llama-cpp-q3xl-2gpu-mtp-q8 -- --max-tokens 256
```

See [PERFORMANCE.md](PERFORMANCE.md) for the measured topology, ubatch, and MTP
comparisons.
