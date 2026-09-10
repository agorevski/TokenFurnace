# Nex-N2.5-mini performance

## Measured results

Recorded 2026-09-09 with
`abenzerps/Nex-N2.5-mini-GGUF`
`Nex-N2.5-mini-Q4_K_M.gguf` (21,166,757,664 bytes, SHA-256
`dd296f683c798a3e4058fb1ef8c462e6a4cc8b89cd196742a8e5a3da4057fbb3`),
llama.cpp commit `434ddbb`, CUDA `sm_75`, one RTX 8000, flash attention,
batch 8192, ubatch 2048, and five repetitions per native test.

### Native llama.cpp

Values are mean +/- standard deviation in tokens/second.

| Test | Throughput |
|---|---:|
| 512-token prefill | 2,074.40 +/- 52.66 |
| 4096-token prefill | 2,969.05 +/- 26.66 |
| 128-token decode | 122.17 +/- 0.14 |

### OpenAI-compatible text server

The text profile loaded a 65,536-token context in 22,113 MiB on GPU 0. The
server identified 34,660,610,688 total parameters in the Q4_K_M GGUF. A
direct-response request with thinking disabled returned exactly `RTX8000_OK`.

Three sequential coding requests generated 318 tokens before normal EOS:

| Metric | Result |
|---|---:|
| Wall time | 3.197 s |
| Aggregate end-to-end output | **99.48 tok/s** |
| Mean per-request output | **100.16 tok/s** |

### Vision validation

The F16 projector was 899,282,976 bytes with SHA-256
`4734f7323dfc0e8dcd5c7c408991aad438021aeab761d223aec0b38a4457ca83`.
Loading it increased GPU 0 use to 23,221 MiB.

Two generated 64x64 PNGs contained only one RGB color. The model identified
them as `Red` and `Blue`, validating the projector and OpenAI image-content
path on Turing.
