# Spark-X2.5-4B performance

## Measured results

Recorded 2026-09-09 with
`XHToken/Spark-X2.5-4B-GGUF`
`Spark-X2.5-4B-Q8_0.gguf` (4,375,021,152 bytes, SHA-256
`5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2`),
XHToken llama.cpp commit `4a3635c`, CUDA `sm_75`, one RTX 8000, flash
attention, batch 8192, ubatch 2048, and five repetitions per native test.

### Native llama.cpp

Values are mean +/- standard deviation in tokens/second.

| Test | Throughput |
|---|---:|
| 512-token prefill | 4,410.82 +/- 102.58 |
| 4096-token prefill | 4,665.82 +/- 31.56 |
| 128-token decode | 103.59 +/- 0.15 |

### OpenAI-compatible server

The one-GPU profile loaded a 65,536-token context. Model and context occupied
7,491 MiB on GPU 0. A deterministic smoke request returned exactly
`RTX8000_OK`; server timings were 461.73 tok/s prompt evaluation and 74.14
tok/s decode for the short eight-token response.

Three sequential coding requests each reached the 256-token limit:

| Metric | Result |
|---|---:|
| Total generated tokens | 768 |
| Wall time | 9.015 s |
| Aggregate end-to-end output | **85.19 tok/s** |
| Mean per-request output | **85.21 tok/s** |
| Server decode range | 87.03-88.14 tok/s |

The API result validates chat templating and sustained generation rather than
only the native kernel path. The profile deliberately uses 65,536 context:
the model declares a 1,048,576-token training limit, but the maximum practical
KV cache must be measured separately rather than inferred from that metadata.
