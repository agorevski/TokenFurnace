# Ternary Bonsai 27B performance

## Measured results

Recorded 2026-09-09 with
`prism-ml/Ternary-Bonsai-27B-gguf`
`Ternary-Bonsai-27B-PQ2_0.gguf` (7,165,121,600 bytes, SHA-256
`e4781999f1997ef97ce0c58d05750835acc999d18d83ee6489ba7ac7b14cb5f6`),
PrismML llama.cpp commit `d8f26ee`, CUDA `sm_75`, one RTX 8000, flash
attention, batch 8192, ubatch 2048, and five repetitions per native test.

### Native llama.cpp

Values are mean +/- standard deviation in tokens/second.

| Test | Throughput |
|---|---:|
| 512-token prefill | 862.83 +/- 7.25 |
| 4096-token prefill | 824.25 +/- 17.44 |
| 128-token decode | 42.27 +/- 0.29 |

### OpenAI-compatible server

The publisher's group-128 ternary pack loaded a 65,536-token context in
11,701 MiB on GPU 0. The server identified 26,895,998,464 parameters and a
`PQ2_0` 2.13-bpw representation.

Three sequential thinking-mode coding requests each reached the 256-token
limit:

| Metric | Result |
|---|---:|
| Total generated tokens | 768 |
| Wall time | 21.969 s |
| Aggregate end-to-end output | **34.96 tok/s** |
| Mean per-request output | **35.93 tok/s** |

An exact-answer request with thinking disabled returned `RTX8000_OK` in eight
tokens. Its short-request server timings were 111.78 tok/s prompt evaluation
and 42.63 tok/s decode. With thinking enabled, the same 128-token budget was
spent entirely on hidden reasoning and exposed an empty final content field;
applications must budget for reasoning tokens or disable thinking for direct
answers.
