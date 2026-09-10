# MiniMax-M2.5 performance

## Measured results

Recorded 2026-09-09 with
`DevQuasar/MiniMaxAI.MiniMax-M2.5-GGUF`
`MiniMaxAI.MiniMax-M2.5.Q2_K` (83,294,393,344 bytes across six shards),
llama.cpp commit `434ddbb`, CUDA `sm_75`, the GPU 0-1 NVLink pair, flash
attention, batch 8192, ubatch 2048, and five repetitions per native test.

### Why this model

MiniMax-M2.5 was selected as a practical substitute for the inaccessible
GLM-5.3-Flash EXL3 2BPW repository. Both target frontier coding and agentic
workloads. The 83.3 GB Q2_K GGUF is larger than the 60.1 GB IQ2_XXS option,
preserving more model precision while still fitting on one 96 GiB NVLink pair
with a useful context.

### Native llama.cpp

Values are mean +/- standard deviation in tokens/second.

| Test | Throughput |
|---|---:|
| 512-token prefill | 417.03 +/- 2.33 |
| 4096-token prefill | 785.79 +/- 6.77 |
| 128-token decode | 63.46 +/- 0.42 |

Native decode is 2.83x the 22.42 tok/s measured for GLM-5.3-Flash Q3_K_XL on
all four GPUs. The different models and quantizations mean this is a practical
throughput comparison, not a quality-equivalence claim.

### OpenAI-compatible server

The model loaded a 32,768-token context and reported 228,689,764,864
parameters. Memory use was 46,039 MiB on GPU 0 and 43,723 MiB on GPU 1. This
leaves little room to increase context on the 96 GiB pair; the model's
196,608-token training context is not practical with this Q2_K artifact.

Three sequential coding requests each reached the 256-token limit:

| Metric | Result |
|---|---:|
| Total generated tokens | 768 |
| Wall time | 14.199 s |
| Aggregate end-to-end output | **54.09 tok/s** |
| Mean per-request output | **55.39 tok/s** |
| Server decode range | 62.33-63.09 tok/s |

A correctness smoke request returned `RTX8000_OK` as requested. The server
also exposed 57 reasoning tokens, for 68 completion tokens total, despite
requesting no reasoning; clients must account for the model's reasoning trace.

### Artifact provenance

| Shard | Bytes | SHA-256 |
|---|---:|---|
| 1 | 15,903,084,992 | `6610d42dc4098ea42c669bafeed48a42d14d3856e3cbc0afccbb534d3b3ffd3e` |
| 2 | 15,984,891,520 | `574a79fdc4c0a1746df263f2d56bd158da4ba2dd4b08069af29f45c472cf26e5` |
| 3 | 15,984,891,520 | `51da5d6a4e2fedf3e0aac129009542a1e09238db7ce514171de89de1de114f48` |
| 4 | 15,984,891,520 | `1b316cd6e296166b67216373503b8b0a04af43e1b701d317f974c19233eaaef1` |
| 5 | 15,984,891,520 | `3b4d35dc1058fbe5c52336b570ef090d326b5ec8cb11fc4ce361b4a30d62d5a1` |
| 6 | 3,460,030,432 | `154a4ba4dd9021ed6ff4fd166ce121bceda31dc27a149b76833d8afce1b868ba` |
