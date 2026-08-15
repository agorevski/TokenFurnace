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

## Official unquantized checkpoint

The `vllm-official-fp16-tp2` profile downloads `Qwen/Qwen3.8-27B` directly and
loads its unquantized safetensors. The source weights are BF16; because Turing
cannot execute BF16 natively, vLLM casts them to FP16. Results are recorded
separately from GGUF so precision and runtime effects are not conflated.

The 18 official shards contain 55,563,006,776 bytes. vLLM 0.21.0 loaded them
across GPUs 2-3 with 25.24 GiB of model memory per GPU. The non-MTP profile
created a 511,180-token KV cache, enough for 15.6 concurrent 32,768-token
requests.

| Official FP16 runtime configuration | Result |
|---|---:|
| TP2, one request, 512 output tokens | **16.99 tok/s** |
| TP2, concurrency 4, eight 256-token requests | **70.68 aggregate tok/s** |
| Mean per-request rate at concurrency 4 | 17.71 tok/s |

### Built-in MTP-1

The official checkpoint's built-in MTP head works through vLLM with one draft
token. A stale FlashInfer build cache initially retained Conda GCC 15; after a
clean rebuild, the required host settings were GCC 13 and
`FLASHINFER_EXTRA_LDFLAGS=-L/home/algore/miniconda3/lib`.

| MTP-1 metric | Result |
|---|---:|
| Steady one-request throughput, 512 output tokens | **31.44 tok/s** |
| Improvement over non-MTP | **85.0%** |
| Accepted draft tokens | 468 / 554 (**84.5%**) |
| KV cache capacity | 453,174 tokens |
| Maximum 32,768-token concurrency | 13.83x |

The first request measured 11.91 tok/s because it included one-time FlashInfer
kernel compilation. The second identical request is the steady result above.
Generated output passed a direct correctness smoke test. MTP reduces KV
capacity by 11.3%, but nearly doubles interactive decode throughput, so
`vllm-official-fp16-tp2-mtp1` is the recommended official-weight profile.
