# Qwen3.8-27B performance

## Measured results

### Full-context serving smoke benchmark

Recorded 2026-09-20 UTC with `unsloth/Qwen3.8-27B-GGUF`
`Qwen3.8-27B-Q4_K_M.gguf` (17,106,773,984 bytes), llama.cpp commit
`035e22731a7fd70b9854b3a2d64ec68e9b1a45d3` (build 359), CUDA `sm_75`,
flash attention, tensor split `1,1` across NVLink-connected GPUs 2-3, context
262,144, batch 8192, ubatch 2048, and one server slot.

The server loaded successfully with a reported 262,144-token slot and used
17,851 MiB on GPU 2 and 17,861 MiB on GPU 3 after the measured request. A
59-token prompt was warmed once, then reused with prompt caching enabled for a
256-token generation:

| Metric | Result |
|---|---:|
| End-to-end latency | 5.769 s |
| End-to-end output throughput | **44.38 tok/s** |
| Server-reported decode | **45.77 tok/s** |
| Finish reason | `length` (256 / 256 tokens) |

A separate correctness request returned the requested `context-ok` text. This
quick test validates full-context allocation and short warm-cache generation;
it does **not** measure prefill throughput for a 262,144-token prompt. GPUs 2
and 3 reached 81 C and 79 C respectively at the post-request sample, with no
server errors observed. Raw evidence is retained under
`.benchmark-runs/qwen3.8-27b/20260920T045749Z/`.

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

## Single-GPU W4A16 AWQ serving

**Measured 2026-09-22 UTC.** These tests used
`philbert440/Qwen3.8-27B-W4A16-AWQ` commit
`7908d42a71077a5e4dc458f273682b12dfe384a0`, a compressed-tensors W4A16
checkpoint with asymmetric INT4 weights, group size 128, and FP16 activations.
The two weight files total 19,547,867,688 bytes: `model.safetensors`
(SHA-256 `15c5b07049149c73236254d53eca1d2f3274f9fb6803540ca47b1ce657dcf583`)
and the BF16 MTP head (SHA-256
`90fa0e3eed5a647c035c6df9ecabc416c0f8d573ff84ac12485b085f00a7cdf2`).
That is 14.3% larger than the 17,106,773,984-byte Q4_K_M GGUF, partly because
the AWQ repository retains the vision tower and MTP head. Serving used
`--language-model-only`.

The measured runtime was vLLM 0.21.0 under Python 3.13.9 on physical GPU 1,
with FP16 activations and KV cache, Triton attention, 32,768-token context, 94% memory
utilization, 8,192 batched tokens, aligned automatic prefix caching,
`max-num-seqs=8`, and medium reasoning. The coding workload used a 17-token
prompt, temperature zero, fixed 512-token output, two waves of requests, and
no shared prompt prefix. Aggregate rates are total output tokens divided by
wall time; mean request rates include each request's queue and service time.

| Profile | 2 sessions aggregate / mean | 4 sessions aggregate / mean | 5 sessions aggregate / mean |
|---|---:|---:|---:|
| No MTP | 56.68 / 28.41 tok/s | 111.00 / 27.76 tok/s | 133.82 / 26.77 tok/s |
| MTP-1 | 70.94 / 35.80 tok/s | 145.26 / 36.45 tok/s | 167.09 / 33.43 tok/s |
| **MTP-2** | **86.19 / 44.22 tok/s** | **171.85 / 43.06 tok/s** | **182.71 / 37.24 tok/s** |

MTP-2 improved aggregate throughput over no speculation by 52.1% at
concurrency 2, 54.8% at concurrency 4, and 36.5% at concurrency 5. It also
beat MTP-1 by 21.5%, 18.3%, and 9.3% respectively. The MTP-2 server reported
87-91% draft acceptance under the five-session load. A four-sequence
scheduler cap reduced five-session throughput to 151.51 tok/s because one
request queued; raising `max-num-seqs` to 8 produced the reported 182.71 tok/s.

Warm single-request 256-token throughput was 31.10 tok/s without speculation,
43.23 tok/s with MTP-1, and 53.36 tok/s with MTP-2. First requests were much
slower because of JIT and CUDA graph work and are not steady-state results. A
cold 9,511-token prefill on the non-MTP profile completed at 325.53 end-to-end
prompt tok/s. All three profiles returned the expected `benchmark-ok` final
text, and every timed generation completed the requested token count with
finish reason `length`.

The non-MTP profile exposed a 342,016-token KV cache; MTP-1 exposed 288,358
tokens and MTP-2 272,036 tokens. GPU 1 used about 44.1 GiB with the final
profiles and reached 86 C during sustained tests. Other services ran on GPUs
0, 2, and 3, but GPU 1 had no competing process. Because the card was near its
thermal ceiling and neighboring GPUs were also hot, these measurements should
be treated as sustained shared-host results rather than cool-card peaks. Raw
evidence is retained under
`.benchmark-runs/qwen3.8-27b/20260922T161449Z/`.

The shipped AWQ profiles now expose the model's native 262,144-token maximum
to avoid rejecting requests whose input plus output budget exceeds 32,768.
This changes the request validation limit, not the finite paged-KV capacity:
the measured MTP-2 cache held about 272K total tokens, shared by all active
requests. The throughput table above was measured with the earlier 32,768-token
limit and has not been relabeled as a 262K-context measurement.

## Dedicated qwen38 Docker image

**Measured 2026-09-22 UTC.** The
`vllm/vllm-openai:qwen38` image at digest
`sha256:4a2f33a884222f7049b983263ad9976f89452bb81affecf5b67d89ad35c1bc31`
contains vLLM `0.1.dev19754+g3a0914114`. It loaded the same 18-shard official
checkpoint described above, cast BF16 to FP16, and used TP2 across
NVLink-connected GPUs 2-3. The text-only profile used Triton attention,
32,768-token context, aligned prefix caching, and 94% memory utilization.

The first startup loaded weights in 27.65 seconds, then spent 332.22 seconds
profiling, compiling, and warming the Turing kernels. The resulting cache is
persisted outside the disposable container. vLLM reported 25.24 GiB of model
memory per GPU and a 477,866-token KV cache, enough for 14.58 concurrent
32,768-token requests. Each GPU used about 44.6 GiB after startup.

A temperature-zero, no-thinking OpenAI chat request with the prompt
`Reply with exactly: qwen38-local-ok` returned exactly `qwen38-local-ok`,
finish reason `stop`, 22 prompt tokens, and 7 completion tokens. This is a
successful load-and-generate correctness test, not a throughput benchmark.
Raw startup logs, failed NVIDIA-runtime attempts, profile snapshots, GPU
samples, and the response JSON are retained under
`.benchmark-runs/qwen3.8-27b/20260922T200248Z/`.

### W4A16 AWQ Docker profile

**Measured 2026-09-22 UTC.** The
`vllm-docker-qwen38-awq-int4-1gpu-mtp2` profile loaded
`philbert440/Qwen3.8-27B-W4A16-AWQ` in the same dedicated image on GPU 1.
vLLM identified `quantization=compressed-tensors` and selected
`MarlinLinearKernel for CompressedTensorsWNA16`, confirming that this profile
uses the Q4 weights rather than runtime-cast FP16 weights.

The two-shard 18.21 GiB checkpoint loaded in 3.85 seconds. With its built-in
MTP-2 head, the model occupied 17.72 GiB before cache and warmup allocations.
The complete server used about 41.7 GiB of GPU memory and exposed a
326,137-token KV cache, enough for 1.24 requests at the configured
262,144-token maximum. First-start engine profiling, compilation, and warmup
took 342.36 seconds; the resulting 276 MiB vLLM cache is persisted outside the
container.

A temperature-zero, no-thinking request with the prompt
`Reply with exactly: qwen38-q4-ok` returned exactly `qwen38-q4-ok`, finish
reason `stop`, 23 prompt tokens, and 8 completion tokens. This is a successful
load-and-generate correctness test, not a throughput benchmark. Raw evidence
is retained under `.benchmark-runs/qwen3.8-27b/20260922T203527Z/`.
