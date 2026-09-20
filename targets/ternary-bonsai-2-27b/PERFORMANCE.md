# Ternary Bonsai 2 27B performance

## Measured results

Recorded 2026-09-19 on one Quadro RTX 8000 (`sm_75`) with the PrismML
llama.cpp CUDA 12.4 release `prism-b10709-9a9394a` (build 10709, commit
`9a9394a89`). Both profiles used GPU 3, layer split, full GPU offload, flash
attention, batch 8192, ubatch 2048, FP16 KV, and five repetitions per native
test.

The exact artifacts were:

| Pack | Bytes on disk | SHA-256 |
|---|---:|---|
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5,946,648,928 | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7,206,168,928 | `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` |

### Native llama.cpp

Values are mean +/- standard deviation in tokens/second.

| Pack | 512-token prefill | 4096-token prefill | 128-token decode |
|---|---:|---:|---:|
| PTQ1_0 | 457.42 +/- 2.83 | 429.44 +/- 9.11 | 31.33 +/- 1.45 |
| **PQ2_0 (default)** | **758.08 +/- 4.01** | **732.66 +/- 21.46** | **37.07 +/- 0.16** |

PQ2_0 improved 512-token prefill by **65.7%**, 4096-token prefill by
**70.6%**, and decode by **18.3%**, calculated as
`(PQ2_0 / PTQ1_0 - 1) * 100`. Its artifact is 21.2% larger than PTQ1_0, but
the throughput advantage on Turing is decisive, so `llama-cpp-pq2-1gpu` is
the default.

### Correctness and server load

PQ2_0 loaded with a 65,536-token context and returned exactly `RTX8000_OK`
with thinking disabled. The eight-token response measured 39.13 tok/s server
decode; this is a correctness smoke test, not a long-generation throughput
result. No load, generation, or invalid-output errors appeared in the retained
server log.

### Measurement caveats

An existing idle Qwen server remained resident during both sweeps, including
1,192 MiB on the benchmark GPU. Telemetry observed only sparse 1% samples on
the other GPUs, but the host was not exclusively reserved, so these results
are explicitly co-tenant.

GPU 3 reached 87 C for PTQ1_0 and 88 C for PQ2_0. No hardware or software
thermal-throttle event was recorded; both runs periodically reached the
260 W board power limit. PTQ1_0 decode declined from 33.68 to 30.10 tok/s
across its five samples, while PQ2_0 remained between 36.79 and 37.18 tok/s.
Raw output and GPU telemetry are retained under
`.benchmark-runs/ternary-bonsai-2-27b/20260919T233239Z/`.
