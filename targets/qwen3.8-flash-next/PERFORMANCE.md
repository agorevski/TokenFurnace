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

### MTP limitation

The official architecture has one MTP layer, but this Unsloth Q3 GGUF does not
contain MTP tensors. Starting `draft-mtp` fails explicitly with:

```text
context type MTP requested but model doesn't contain MTP layers
```

MTP is therefore not enabled or advertised by the target. It should be
retested only if the publisher adds MTP tensors to a later GGUF revision.
