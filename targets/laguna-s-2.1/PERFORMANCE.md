# Laguna S 2.1 performance

## Measured results

Recorded 2026-08-30 with
`poolside/Laguna-S-2.1-GGUF`
`laguna-s-2.1-Q4_K_M.gguf` (96,031,829,760 bytes), Poolside llama.cpp
`laguna` commit `06f8ceb`, CUDA `sm_75`, four-GPU layer split, flash
attention, batch 8192, ubatch 2048, and five repetitions per native test.

The downloaded file is 96.03 GB, not the 68 GB stated in the model card.
llama.cpp reports its hybrid contents as `laguna 118B.A8B Q8_0`.

### Native llama.cpp

Values are mean +/- standard deviation in tokens/second.

| Test | Throughput |
|---|---:|
| 512-token prefill | 236.67 +/- 2.79 |
| 4096-token prefill | 562.08 +/- 2.67 |
| 128-token decode | 45.57 +/- 0.38 |

A two-GPU NVLink-pair profile was attempted first but failed to load: the
96.03 GB model leaves insufficient space in the pair's 96.80 GiB usable VRAM
for allocator overhead. All four GPUs are required.

### OpenAI-compatible server

Both profiles used a 65,536-token context and generated 256 tokens from the
same 48-token templated prompt.

| Profile | End-to-end output | Server decode | Prompt evaluation |
|---|---:|---:|---:|
| **Baseline** | **37.47 tok/s** | **45.12 tok/s** | **42.59 tok/s** |
| DFlash | 8.16 tok/s | 8.68 tok/s | 26.33 tok/s |

The DFlash drafter accepted only 177 of 1,140 proposed tokens (15.5%) with a
mean accepted length of 3.27. Its verification overhead reduced end-to-end
output by 78.2%, so the non-speculative profile is the default.

### Server memory

| GPU | Baseline | DFlash |
|---:|---:|---:|
| 0 | 19,461 MiB | 21,957 MiB |
| 1 | 19,045 MiB | 19,237 MiB |
| 2 | 19,045 MiB | 19,237 MiB |
| 3 | 44,265 MiB | 45,077 MiB |
