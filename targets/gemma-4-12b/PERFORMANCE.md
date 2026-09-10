# Gemma 4 12B performance

## Measured results

Recorded 2026-09-09 with
`unsloth/gemma-4-12b-it-GGUF`
`gemma-4-12b-it-Q4_K_M.gguf` (7,121,861,440 bytes, SHA-256
`0a270ec9fe6b34f4a0d33992b6135117b484ebc4766ab76b51d4ae8c457e4c42`),
llama.cpp commit `c589f0e`, CUDA `sm_75`, one RTX 8000, flash attention,
batch 8192, ubatch 2048, and five repetitions per native test.

### Native llama.cpp

Values are mean +/- standard deviation in tokens/second.

| Test | Throughput |
|---|---:|
| 512-token prefill | 1,706.04 +/- 18.07 |
| 4096-token prefill | 1,604.71 +/- 31.45 |
| 128-token decode | 58.57 +/- 0.29 |

### OpenAI-compatible text server

The text profile loaded a 65,536-token context in 9,719 MiB on GPU 0. An
exact-answer request returned `RTX8000_OK`. Three sequential coding requests
generated 354 tokens before normal EOS:

| Metric | Result |
|---|---:|
| Wall time | 6.595 s |
| Aggregate end-to-end output | **53.68 tok/s** |
| Mean per-request output | **53.83 tok/s** |
| Server decode range | 57.19-58.08 tok/s |

### Vision validation

The F16 projector was 175,115,840 bytes with SHA-256
`91f086971e56d7a7d8d39e271873fccdb49541bd259d6e02c401a4f1cb7a219e`.
Loading it increased GPU 0 use to 10,063 MiB.

Two generated 64x64 PNGs contained only one RGB color. The model identified
the red image as `Red` and the blue image as `Navy blue`, proving that the
projector and OpenAI image-content path work rather than merely testing a
text-only server.
