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

This model currently requires
[llama.cpp PR 27742](https://github.com/ggml-org/llama.cpp/pull/27742). The
profiles therefore use the isolated `build-qwen38-next` build rather than the
repository's stable llama.cpp build.

The current Unsloth Q3 GGUF does not include the official checkpoint's MTP
layers, so speculative MTP is unavailable. Tensor split also aborts in the
required PR build; the default uses the measured two-GPU layer split.

```bash
./scripts/download-model.sh qwen3.8-flash-next
./scripts/benchmark-native.sh qwen3.8-flash-next
./scripts/serve-model.sh qwen3.8-flash-next
./scripts/benchmark-model.sh qwen3.8-flash-next -- --max-tokens 512
```

See [PERFORMANCE.md](PERFORMANCE.md) for the measured topology and ubatch
sweeps.
