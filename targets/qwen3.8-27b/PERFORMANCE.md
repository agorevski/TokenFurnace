# Qwen3.8-27B performance

## Measured results

Recorded 2026-08-14 with `unsloth/Qwen3.8-27B-GGUF`
`Qwen3.8-27B-Q4_K_M.gguf` (17,106,773,984 bytes), llama.cpp build `15586e2`
(`b10298`), CUDA sm_75, flash attention, batch 8192, ubatch 2048, and five
repetitions per native test.

### Native topology and split sweep

Values are mean +/- standard deviation in tokens/second.

| Configuration | 512 prefill | 4096 prefill | 128 decode |
|---|---:|---:|---:|
| 1 GPU, layer | 715.85 +/- 5.92 | 659.75 +/- 24.91 | 28.10 +/- 0.14 |
| 2 GPU, layer | 721.88 +/- 4.29 | 875.98 +/- 19.89 | 29.58 +/- 0.10 |
| **2 GPU, tensor (default)** | **1236.43 +/- 5.44** | **1113.88 +/- 35.67** | **40.37 +/- 1.16** |
| 2 GPU, row | failed to load | failed to load | failed to load |

Tensor parallelism is **72.7% faster** than one GPU at 512-token prefill,
**68.8% faster** at 4096-token prefill, and **43.7% faster** at decode. It also
beats the two-GPU layer split by 71.3%, 27.2%, and 36.5%, respectively.

The corrected native harness converts server-style `TENSOR_SPLIT=1,1` to
llama-bench's `1/1` syntax. Passing commas to llama-bench selects multiple
benchmark variants rather than one two-device fraction list.

### Interpretation

Qwen3.8-27B is dense, unlike the sparse Qwen3.6-35B-A3B. Its large dense FFN
matrix multiplies benefit substantially from tensor parallelism across the
two-link NVLink pair. The collective overhead is more than repaid in both
prefill and serial decode, making `llama-cpp-q4km-2gpu-tensor` the default.

Native `llama-bench` rates exclude HTTP and chat-template overhead. API output
rates include prompt evaluation and request handling and are reported
separately below.

### OpenAI-compatible server

The default tensor profile loaded successfully, returned valid text, and served
a cold 71-token prompt plus 512 generated tokens:

| Metric | Result |
|---|---:|
| End-to-end latency | 11.915 s |
| End-to-end output throughput | **42.97 tok/s** |
| Server-reported prompt evaluation | 166.16 tok/s |
| Server-reported decode | **44.61 tok/s** |

The server result is the practical interactive number. Its decode rate is
10.5% above the native `llama-bench` mean, while end-to-end throughput remains
6.3% below server decode because it includes prompt evaluation and HTTP.

### MTP status

The requested `unsloth/Qwen3.8-27B-GGUF` repository did not publish a separate
MTP draft GGUF at measurement time. No draft artifact from another publisher
was mixed into this result.
