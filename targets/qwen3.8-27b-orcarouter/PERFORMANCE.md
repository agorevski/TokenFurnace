# Qwen3.8-27B Uncensored OrcaRouter performance

## Measured results

Recorded 2026-08-18 with
`chimingw/Qwen3.8-27B-Uncensored-OrcaRouter-GGUF`
`Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf` (16,810,714,496 bytes),
llama.cpp build `15586e2` (`b10298`), CUDA sm_75, flash attention, batch 8192,
ubatch 2048, and five repetitions per native test.

### Native topology and split sweep

Values are mean +/- standard deviation in tokens/second.

| Configuration | 512 prefill | 4096 prefill | 128 decode |
|---|---:|---:|---:|
| 1 GPU, layer | 717.73 +/- 5.32 | 666.54 +/- 25.53 | 28.35 +/- 0.12 |
| 2 GPU, layer | 715.93 +/- 2.35 | 889.24 +/- 13.83 | 29.87 +/- 0.04 |
| **2 GPU, tensor** | **1150.53 +/- 43.10** | **1064.76 +/- 26.30** | **40.65 +/- 0.96** |

Tensor parallelism is **60.3% faster** than one GPU at 512-token prefill,
**59.7% faster** at 4096-token prefill, and **43.4% faster** at decode. It
beats the two-GPU layer split by 60.7%, 19.7%, and 36.1%, respectively.

Native `llama-bench` rates exclude HTTP, chat-template, and speculative
decoding overhead.

### OpenAI-compatible server

The tensor profile loaded successfully and served a cold 59-token prompt plus
512 generated tokens. The same request was repeated with the GGUF's in-model
MTP head and two draft tokens.

| Configuration | Latency | E2E output | Server prompt eval | Server decode | Acceptance |
|---|---:|---:|---:|---:|---:|
| Tensor, MTP off | 12.013 s | 42.62 tok/s | 156.63 tok/s | 44.06 tok/s | - |
| **Tensor, MTP-2 (default)** | **7.560 s** | **67.73 tok/s** | 132.19 tok/s | **72.09 tok/s** | 77.95% (311/399) |

MTP-2 improves practical end-to-end output throughput by **58.9%** and
server-reported decode by **63.6%**. The prompt-evaluation rate is lower
because llama.cpp creates the draft context and falls back to CPU sampling for
MTP with tensor split, but the high draft acceptance more than offsets that
overhead during generation.

### Interpretation

The model is dense, so its large matrix multiplications benefit from tensor
parallelism over the two-link NVLink pair. Layer splitting offers little
short-prompt or decode benefit, while tensor splitting materially accelerates
all three native tests. The matching MTP head then provides the largest
interactive gain and is therefore enabled in the default profile.

The published checkpoint is derived from FP8, but these measurements use its
Q4_K_M GGUF. RTX 8000 does not execute FP8 natively; llama.cpp runs the
quantized GGUF through supported CUDA kernels instead.
