# Muse Glimmer 30B performance

## Measured results

Recorded 2026-08-30 with
`meta-models/Muse-Glimmer-30B-GGUF`
`Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf` (19,653,960,832 bytes),
llama.cpp commit `c589f0e`, CUDA `sm_75`, one RTX 8000, flash attention,
batch 8192, ubatch 2048, and five repetitions per native test.

### Native llama.cpp

Values are mean +/- standard deviation in tokens/second.

| Test | Throughput |
|---|---:|
| 512-token prefill | 819.28 +/- 4.00 |
| 4096-token prefill | 778.35 +/- 20.17 |
| 128-token decode | 25.68 +/- 0.10 |

### OpenAI-compatible server

Both profiles used a 131,072-token context and generated 256 tokens from the
same 63-token templated prompt.

| Profile | End-to-end output | Server decode | Prompt evaluation |
|---|---:|---:|---:|
| Baseline | 25.03 tok/s | 25.71 tok/s | **219.85 tok/s** |
| **DFlash** | **35.40 tok/s** | **40.53 tok/s** | 69.83 tok/s |

The DFlash drafter accepted 173 of 244 proposed tokens (70.9%), with a mean
accepted length of 3.11. It improved end-to-end output by 41.4% and is the
default despite its one-time prompt-evaluation overhead.

### Server memory

| Profile | GPU 0 |
|---|---:|
| Baseline | 21,135 MiB |
| DFlash | 25,973 MiB |

Both configurations fit comfortably on one 48 GiB RTX 8000. The target
benchmarks text generation only; the separate perception encoder was not
downloaded or measured.
