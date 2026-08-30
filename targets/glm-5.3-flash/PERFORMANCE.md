# GLM-5.3-Flash performance

## Measured results

Recorded 2026-08-29 with
`unsloth/GLM-5.3-Flash-GGUF`
`UD-Q3_K_XL` (147,535,921,955 bytes across four shards), Unsloth llama.cpp
`glm5next/upstream` commit `f30bed8`, CUDA `sm_75`, four-GPU layer split,
flash attention, batch 8192, ubatch 2048, and five repetitions per native
test.

### Native llama.cpp

Values are mean +/- standard deviation in tokens/second.

| Test | Throughput |
|---|---:|
| 512-token prefill | 306.35 +/- 1.93 |
| 4096-token prefill | 554.96 +/- 1.77 |
| 128-token decode | 22.42 +/- 0.11 |

The 147.54 GB quant cannot fit on either 96 GiB NVLink pair, so the profile
must span all four GPUs and cross PCIe between the two pairs.

### OpenAI-compatible server

The default profile loaded a 65,536-token context and completed a 256-token
response:

| Metric | Result |
|---|---:|
| End-to-end output | 19.02 tok/s |
| Server decode | 19.87 tok/s |
| Server prompt evaluation (19 tokens) | 33.84 tok/s |
| GPU 0 memory | 36,135 MiB |
| GPU 1 memory | 44,621 MiB |
| GPU 2 memory | 43,955 MiB |
| GPU 3 memory | 35,541 MiB |

The response reached the requested length limit, confirming generation and
the OpenAI-compatible API path. The difference from native decode includes
chat templating, sampling, HTTP handling, and server scheduling.
