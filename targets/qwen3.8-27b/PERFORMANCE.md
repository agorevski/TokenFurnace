# Qwen3.8-27B performance

## Measured results

### Cold 32K prefill followed by 1K decode (2026-09-23 UTC)

**Measured** in one OpenAI-compatible API request against the four-GPU
`llama-cpp-q4km-4gpu-tensor-ctx262k` default, using the same
`Qwen3.8-27B-Q4_K_M.gguf` artifact (SHA-256
`7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b`)
and llama.cpp build 359 described below. The server reserved a 262,144-token
slot, used tensor fractions `1,1,1,1` on GPUs 0-3, CUDA P2P, NCCL, flash
attention, FP16 KV, stock CUDA launch queues, batch 8192, and ubatch 2048.
The prompt was a synthetic, repetitive compiler tutorial request, sized with
the local chat template/tokenizer to **exactly 32,768 tokens**; temperature
was zero and the output limit was **1,024 tokens**.

| Phase | Tokens | Server-reported time | Server-reported rate |
|---|---:|---:|---:|
| Prefill | 32,768 | 23.174 s | **1,414.03 tok/s** |
| Decode | 1,024 | 19.979 s | **51.20 tok/s** |

The end-to-end request took **43.525 s**, or **23.53 output tok/s** when
prefill, generation, and HTTP are all included. Response usage reported
32,768 prompt tokens, 1,024 completion tokens, and **zero cached tokens**.
The finish reason was `length`; the generated text was a coherent but
truncated compiler tutorial, with no malformed-output or server errors.
These are one-request measurements, not five-repeat native benchmark means.
GPUs began at 37-48 C and reached up to 86 C during the request; memory clocks
were 6500 MHz at the post-request sample, so sustained throughput may depend
on thermal conditions.

The exact prompt, its SHA-256 checksum, profile snapshot, response with
server `timings`, server log, GPU samples, and measurement JSON are retained
in `.benchmark-runs/qwen3.8-27b/20260923T190007Z-32k1k/`.
This validates a 32K prefill *inside* the 262K slot, not the throughput of
a 262K-token prefill.

### Matched four-layout topology sweep (2026-09-23 UTC)

**Measured** with the same 17,106,773,984-byte `Qwen3.8-27B-Q4_K_M.gguf`
(SHA-256 `7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b`)
and llama.cpp commit `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`,
build 359. NCCL was enabled in the CUDA build and resolved from the local
Conda installation despite not appearing in `ldconfig -p`. All four GPUs
reported two active NVLink links at 25.781 GB/s; pairs 0-1 and 2-3 are
connected to each other through PCIe host bridges. With
`GGML_CUDA_P2P=1`, flash attention, FP16 KV, batch 8192, ubatch 2048, stock
CUDA launch queues, and five repetitions per native workload:

| Layout / split | 512 prefill (tok/s) | 4096 prefill (tok/s) | 128 decode (tok/s) |
|---|---:|---:|---:|
| GPU 0, layer | 719.07 +/- 4.02 | 681.79 +/- 21.11 | 28.58 +/- 0.07 |
| NVLink 0-1, tensor | 1182.40 +/- 16.67 | 1139.27 +/- 7.27 | 44.96 +/- 0.22 |
| NVLink 2-3, tensor | 1245.92 +/- 6.36 | 1122.99 +/- 39.89 | 40.60 +/- 1.94 |
| **All 4, tensor (default)** | **1566.37 +/- 12.70** | **1508.69 +/- 9.18** | **54.17 +/- 0.38** |
| All 4, layer | 664.87 +/- 1.06 | 1075.81 +/- 7.09 | 11.94 +/- 0.00 |

Four-GPU tensor improved decode by **20.5%** over the faster 0-1 NVLink
pair, and 4096-token prefill by **32.4%** (`(four / pair - 1) * 100`).
The faster pair's decode advantage over pair 2-3 was 4.36 tok/s in this
sweep; both pairs shared the same GPU model and NVLink layout, but cards
2-3 reached 85 C and GPU 0 reached 87 C in the earlier single-card run.
Temperatures and sequential test order limit cross-pair attribution. Every
benchmark started without a competing GPU compute process.

An 8192-context pair 0-1 server also finished a 256-token API request at
42.16 end-to-end output tok/s (6.072 s, 59 prompt tokens), and a fixed
5491-token prefill-oriented request at 1083.76 end-to-end prompt tok/s
(5.067 s, one generated token). It returned exactly `nvlink-pair-ok`
with finish reason `stop` on a separate correctness request. The
`llama-cpp-q4km-2gpu-tensor-01-ctx262k` fallback subsequently loaded a
262,144-token slot on GPUs 0-1, used approximately 17.85 GiB per GPU,
and returned exactly `pair-full-context-ok`. It was not benchmarked with
a 262K-token input prompt.

The best **default configuration remains**
`llama-cpp-q4km-4gpu-tensor-ctx262k`; the best measured two-GPU decode
fallback is `llama-cpp-q4km-2gpu-tensor-01-ctx262k`. Native JSON, profile
snapshots, command strings, temperatures, API responses, and errors are
retained under `.benchmark-runs/qwen3.8-27b/20260923T184542Z-topology/`.

### Four-GPU tensor optimization (2026-09-23 UTC)

**Measured** on four Quadro RTX 8000 GPUs, with NVLink pairs 0-1 and 2-3
and PCIe traffic between pairs. The exact artifact was
`unsloth/Qwen3.8-27B-GGUF` `Qwen3.8-27B-Q4_K_M.gguf` (17,106,773,984 bytes,
SHA-256 `7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b`).
The runtime was llama.cpp commit `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`
(build 359), compiled for CUDA `sm_75` with NCCL. All native profiles used
`GGML_CUDA_P2P=1`, flash attention, FP16 KV, 8192 batch, 2048 ubatch, 512-
and 4096-token prefill and 128-token decode at five repetitions each. CUDA
launch queues were stock unless labeled `4x`. These are native `llama-bench`
rates (mean +/- standard deviation, tokens/second), not HTTP rates:

| GPU layout / split | 512 prefill | 4096 prefill | 128 decode |
|---|---:|---:|---:|
| 1 GPU (3), layer | 692.19 +/- 11.66 | 627.73 +/- 4.96 | 27.85 +/- 0.16 |
| 2 GPUs (2-3), tensor | 1156.56 +/- 41.20 | 1104.21 +/- 9.07 | 42.78 +/- 0.31 |
| **4 GPUs (0-3), tensor** | **1598.90 +/- 9.85** | **1625.56 +/- 25.34** | **55.90 +/- 0.24** |
| 4 GPUs (0-3), tensor, queues `4x` | 1572.82 +/- 7.08 | 1500.24 +/- 23.95 | 53.31 +/- 0.71 |
| 4 GPUs (0-3), layer | 681.56 +/- 2.24 | 1110.67 +/- 7.89 | 11.94 +/- 0.01 |
| 4 GPUs (0-3), layer, queues `4x` | 724.54 +/- 1.29 | 1111.43 +/- 9.81 | 28.77 +/- 0.09 |

The four-GPU tensor mode is 38.2% faster at 512-token prefill, 47.2% faster
at 4096-token prefill, and 30.7% faster at decode than the two-GPU tensor
mode (`(four / two - 1) * 100`). A repeat of four-GPU layer with stock queues
measured 677.94 +/- 1.24, 1094.19 +/- 9.77, and 11.94 +/- 0.01
respectively; stock-queue layer decode is particularly poor on this host.
The two-GPU layer control measured 718.26, 831.31, and 28.69 tokens/second.
An unrelated llama-cli job took GPU 0 during part of the sweep; the affected
layer run was repeated after that job stopped. GPU 0 reached 83 C and GPU 3
84 C by the final tensor-queue run; the sequential sweep was not a
temperature-matched randomized experiment.

For a single-request API comparison, both tensor profiles used one 8192-token
slot, temperature zero, prompt caching enabled, the same model, and a
256-token output from the harness's short default prompt. A separate fixed
5,491-token prompt generated one token for the prefill-oriented request;
its rate includes HTTP and the generated token, **not** just kernel prefill:

| Profile | Decode request, output tok/s (latency) | Prefill-oriented request, end-to-end prompt tok/s (latency) |
|---|---:|---:|
| 2 GPUs, tensor, 8192 context | 41.82 (6.121 s) | 995.47 (5.516 s) |
| **4 GPUs, tensor, 8192 context** | **48.86 (5.239 s)** | **1324.18 (4.147 s)** |

Both decode requests finished at the 256-token length limit. The four-GPU
API speedup was 16.8% for output and 33.0% for the prefill-oriented request.
A separate generation returned exactly `four-gpu-ok` with finish reason `stop`.

**Full-context load and generation:** The promoted
`llama-cpp-q4km-4gpu-tensor-ctx262k` profile loaded a real 262,144-token
server slot, used about 9.8 GiB per GPU at idle, and returned exactly
`full-context-ok` on a correctness request. With medium reasoning enabled,
its 256-token short-prompt request measured **51.50 end-to-end output tok/s**
(4.971 s, 17 prompt tokens), and a 5,449-token prefill-oriented request
measured **1366.31 end-to-end prompt tok/s** (3.988 s, one output token).
Both ended at their requested length. The full-context profile and the 8192
profile have different chat-template reasoning settings, so these are separate
load-and-generate measurements, not a context-size speedup claim. A
262,144-token prompt was not tested. llama.cpp logged that `cache_reuse` was
disabled for this tensor context during the measured run; the promoted profile
does not request that unsupported setting. These requests are not
partial-prefix-reuse benchmarks.

Raw profiles, commands, build log, artifact hash, GPU samples, native JSON,
server logs, API responses and failures are retained in
`.benchmark-runs/qwen3.8-27b/20260923T183007Z/`. GPUs 0 and 1 initially
hosted unrelated services; they were stopped with permission before the
four-GPU measurements. The earlier four-GPU layer result was also retaken
after an unrelated job restarted on GPU 0.

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
| **2 GPU, tensor (then-default)** | **1236.43 +/- 5.44** | **1113.88 +/- 35.67** | **40.37 +/- 1.16** |
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
prefill and serial decode; it was the earlier default. The newer four-GPU
comparison above establishes the current default for an idle four-GPU host.

Native `llama-bench` rates exclude HTTP and chat-template overhead. API output
rates include prompt evaluation and request handling and are reported
separately below.

### OpenAI-compatible server

The then-default two-GPU tensor profile loaded successfully, returned valid
text, and served
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
