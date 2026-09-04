# Qwen3.8-Flash-Next performance

## Measured results

Recorded 2026-08-26 with
`unsloth/Qwen3.8-Flash-Next-GGUF`
`UD-Q3_K_XL` (89,986,353,824 bytes across three shards), llama.cpp PR 27742
commit `035e227`, CUDA `sm_75`, flash attention, batch 8192, ubatch 2048, and
five repetitions per native test.

### Native topology sweep

Values are mean +/- standard deviation in tokens/second.

| Configuration | 512 prefill | 4096 prefill | 128 decode |
|---|---:|---:|---:|
| **2 GPU, layer** | **810.17 +/- 7.46** | 983.31 +/- 4.75 | **38.46 +/- 0.04** |
| 4 GPU, layer | 809.89 +/- 6.60 | **987.64 +/- 2.23** | 34.60 +/- 0.51 |
| 2 GPU, tensor | load failure | load failure | load failure |

Four GPUs are effectively tied at both prefill lengths and decode 10.0% more
slowly. The default therefore keeps the model on the NVLink-connected GPU 2-3
pair. Tensor split is not usable in this PR: model loading aborts at
`tensor->ne[axis] == n_embd + 2*n_embd_gqa`.

### Ubatch sweep on two GPUs

| Ubatch | 512 prefill | 4096 prefill | 128 decode |
|---:|---:|---:|---:|
| 1024 | 748.75 +/- 7.21 | 893.42 +/- 3.35 | **38.52 +/- 0.05** |
| **2048** | **810.17 +/- 7.46** | **983.31 +/- 4.75** | 38.46 +/- 0.04 |
| 4096 | 751.27 +/- 11.23 | 943.76 +/- 2.31 | 38.49 +/- 0.06 |

Decode differences are within noise. Ubatch 2048 improves short prefill by
7.9-8.2% and long prefill by 4.2-10.1%, so it remains the default.

### OpenAI-compatible server

The two-GPU profile served a deterministic 36-token coding prompt and produced
203 tokens of valid Python in 5.646 seconds:

| Metric | Result |
|---|---:|
| End-to-end output | 35.96 tok/s |
| Server decode | 37.77 tok/s |
| Prompt evaluation | 135.09 tok/s |
| GPU 2 memory | 31,359 MiB |
| GPU 3 memory | 30,253 MiB |

A 65,536-token context also loads and reaches a healthy server state, so it is
the default. This leaves practical VRAM headroom because the architecture's
large n-gram embedding is memory-mapped rather than entirely resident in VRAM.

### External MTP comparison

Measured 2026-09-03 with the same Q3 main-model GGUF and two-GPU layer split,
llama.cpp PR 28243 commit `2857e51143bd88ec6fc0246246f42a5d0394d98a`,
and Unsloth's external
`MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`
(2,786,568,256 bytes,
SHA-256 `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6`).
The MTP profile used `draft-mtp`, two draft tokens, 65,536 context, temperature
zero, a fixed 17-token coding prompt, 256 output tokens, concurrency one, warm
prompt caching, and three steady-state repetitions.

| Profile | API output | End-to-end latency | Server decode |
|---|---:|---:|---:|
| No MTP | 44.08 +/- 0.36 tok/s | 5.807 +/- 0.048 s | 45.24 +/- 0.12 tok/s |
| Shared Q8_0 MTP | **64.24 +/- 1.86 tok/s** | **3.987 +/- 0.118 s** | **67.52 +/- 0.32 tok/s** |

MTP improves end-to-end output throughput by 45.7%, improves server-reported
decode by 49.2%, and reduces end-to-end latency by 31.3%. Steady-state draft
acceptance was 79.6% (156 accepted of 196 generated, mean draft length 2.59).
The server reported speculative decoding active, and both runs returned 256
tokens with `finish_reason=length` and valid Python output.

The fixed-seed no-MTP result reproduced byte-for-byte across server restarts.
The fixed-seed MTP result was valid but changed part of the hidden reasoning
trace and ended three characters later at the 256-token truncation boundary.
Because MTP was not byte-identical on this Turing build, it is an optional
performance profile rather than the default.

The shared draft head emits an expected warning while automatic fit measures it
before the main model is available for tensor borrowing. Loading subsequently
completes and the slot reports `speculative: true`. Post-load memory was 34,609
MiB on GPU 2 and 38,095 MiB on GPU 3, versus 34,455 MiB and 33,327 MiB without
MTP. Concurrent serving was not tested because this profile targets
single-request latency.
