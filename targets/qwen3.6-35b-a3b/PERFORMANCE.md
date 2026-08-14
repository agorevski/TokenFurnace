# Qwen3.6-35B-A3B performance

## Measured results

Recorded 2026-08-12 on GPU 2, an NVIDIA Quadro RTX 8000. The model loaded and
generated valid text with llama.cpp build `15586e2` (`b10298`), CUDA sm_75,
flash attention enabled, all layers on GPU, batch 8192, and ubatch 2048.

### Native llama.cpp throughput

`llama-bench -r 5`, Q4_K_M on one GPU:

| Test | Throughput |
|---|---:|
| 512-token prefill | 2138.53 +/- 21.18 tok/s |
| 4096-token prefill | **2820.04 +/- 41.37 tok/s** |
| 128-token decode | **111.87 +/- 0.24 tok/s** |

The 4096-token prefill row followed the shorter tests and GPU 2 reached 70 C.
The decode result is extremely stable across five repetitions.

### MTP speculative decoding sweep

OpenAI-compatible `/v1/chat/completions`, identical 29-token prompt, 512 output
tokens, temperature 0, cache cold after each server restart:

| Draft tokens | E2E latency | E2E output | Native server decode | Acceptance |
|---:|---:|---:|---:|---:|
| Off | 4.912 s | 104.23 tok/s | 107.22 tok/s | - |
| 1 | 4.029 s | 127.07 tok/s | 133.22 tok/s | 88.19% (239/271) |
| **2 (default)** | **3.735 s** | **137.09 tok/s** | **144.59 tok/s** | 74.21% (305/411) |
| 3 | 3.715 s | 137.80 tok/s | 144.48 tok/s | 64.00% (336/525) |

MTP-2 improves end-to-end output throughput by **31.5%** over no speculation.
MTP-3 is only 0.5% faster end-to-end and is fractionally slower in native
server decode while generating 27.7% more draft candidates than MTP-2. The
publisher-recommended MTP-2 setting therefore remains the default: it captures
essentially all available speedup with less rejected work.

API `completion_tokens / total_request_time` includes prompt evaluation and HTTP
overhead, so it is intentionally reported separately from native decode.

### Remaining throughput measurement

The `vllm-fp16-tp2` profile is configured but not yet measured. Its purpose is
aggregate throughput under concurrency, not single-stream latency; no vLLM
number is claimed here until the official FP16 checkpoint completes a real
load-and-generate run on this host.

## Configuration rationale

- **Q4_K_M, one GPU:** 22,663,387,424 bytes (21.10 GiB), leaving ample room for
  compute buffers and a 65,536-token context while avoiding inter-GPU decode
  synchronization.
- **MTP draft length 2:** both the publisher recommendation and the measured
  efficiency winner. Draft length 3 was tied in throughput but did more rejected
  work; length 1 was 7.3% slower than length 2.
- **FP16 vLLM TP2:** Turing has native FP16 tensor cores but no BF16/FP8/FP4
  tensor-core path. GPUs 2-3 are an NVLink pair; TP4 would cross the slower PCIe
  bridge on every collective.
- **Text-only vLLM:** omits the vision encoder and spends the recovered VRAM on
  KV cache and continuous batching.
