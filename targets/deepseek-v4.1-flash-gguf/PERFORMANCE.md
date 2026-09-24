# DeepSeek V4.1 Flash: DwarfStar benchmark

## Active checkpoint: Q2

The local Q4 model file was removed at the user's request. Its benchmark
logs are retained below; **Q4 numbers are not Q2 measurements**. The
replacement is `DeepSeek-V4.1-Flash-Q2.gguf` from the same Hugging Face
repository, expected size 365,713,686,528 bytes and upstream SHA-256
`1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42`.
Q2's routed gate/up experts use `IQ2_XXS` and down experts use `Q2_K`;
its 151.77 GiB main weights still exceed one RTX 8000's VRAM. Q2 load,
output, and performance are **unmeasured** pending the completed download.

## Historical Q4 measurements (model deleted)

## Artifact and host

- Checkpoint: `antirez/deepseek-v4.1-flash-gguf` at Hugging Face revision
  `dd8a266f7145edc19e2334b46e19b6821f221dc7`.
- Exact Q4 file: `DeepSeek-V4.1-Flash-Q4.gguf`, 518,596,067,328 bytes,
  locally verified SHA-256
  `a5e2e2c3ada4b2e98d9f9e4b50f6d9c2a12c2c96f5da165c07e13aff9264984e`
  (the pinned upstream downloader verified both parts and their assembled
  file; see `download.log`).
- 4x Quadro RTX 8000 (`sm_75`, 48 GiB each); 0-1 and 2-3 are NVLink
  pairs, with PCIe between pairs. Intel Xeon W-2295, 251 GiB RAM,
  local NVMe storage, CUDA 12.4.
- Date: September 23, 2026 (local; run artifacts use UTC timestamps).
- Runtime: `antirez/ds4` commit `0aaea5a238fb41a35106a551e73c8409dfb751ac`,
  built with `make cuda CUDA_ARCH=sm_75`. CUDA runtime libraries on
  this host require `/home/algore/miniconda3/lib` in `LD_LIBRARY_PATH`.
- Raw logs and machine snapshots (ignored by Git):
  `.benchmark-runs/deepseek-v4.1-flash-gguf/20260923T215323Z/`.
  Repository revision and pre-existing worktree changes are captured there.

## Status

| Configuration | Model load and generation | Prefill | Decode | API throughput |
| --- | --- | --- | --- | --- |
| Four-GPU in-process TP, Q4 | Unsupported by ds4 V4.1 CUDA; not launched | Unmeasured | Unmeasured | Unmeasured |
| Single-GPU CUDA, Q4 SSD streaming, 8K context, default score tile | **Failed**: first request returns HTTP 500 (invalid CUDA attention launch at layer 20) | Unmeasured | Unmeasured | Unmeasured |
| Single-GPU CUDA, Q4 SSD streaming, 8K context, `DS4_CUDA_NO_SCORE_TILE=1` | **Measured**, valid generated text | 4.93 (512) / 20.80 (4096) tokens/s native | 1.15 (512) / 0.96 (4096) tokens/s native | 35 output tokens in 53.72-55.31 s, three API runs |
| llama.cpp, Q4 | Not comparable: DwarfStar-only GGUF and unverified model support | Unmeasured | Unmeasured | Unmeasured |

Native rates use a single `ds4-bench` sweep per frontier, **32 greedy
generated tokens**, allocated context 8192, automatic expert-cache budget,
and separate 512- and 4096-token prefills from fixed prompt files
(`prompt.sha256`, `prompt-4x.sha256`). The native first-token times were
2002.522 and 2103.963 ms respectively; steady decode rates excluding that
token were 1.20 and 0.99 tokens/s. The measured numbers are not maximum
throughput claims or a controlled comparison against other runtimes. They
are different prompt lengths, so do not compare their decode rates as a
speedup due solely to prefill size.

The API runs used one fixed 27-token question requesting a Python `sum`
function, greedy sampling, `think:false`, maximum 64 output tokens,
**35 actual tokens**, finish reason `stop`, and concurrency one. All three
answers were byte-identical, valid Python with a worked example. Reported
`cached_tokens=0` in each response. Captured end-to-end durations were
55.313, 53.769, and 53.722 s (mean 54.268 s); corresponding end-to-end
output rates were 0.633, 0.651, and 0.651 tokens/s, **including prefill**.
Server-reported prefill/decode durations were 24.718/30.586,
23.959/29.808, and 23.920/29.801 s; dividing 35 output tokens by
the measured decode durations gives 1.144, 1.174, and 1.174 tokens/s.
The separate first successful two-token smoke test was 16.226 s end to
end and is not included in these steady-request results. A 128-token API
request with default thinking ended at its token limit with an incomplete
reasoning block; its 180.195 s end-to-end result is retained in
`api-decode-128.json` but excluded from valid answer throughput.
Aggregate concurrency and API long-prompt prefill were not measured.

The GPU 0 server used approximately 40,205 MiB VRAM after its first request;
the other three GPUs were idle. Its startup budget reported 154.54 GiB
planned including 127.64 GiB dynamic expert cache and 14.24 GiB prefill
headroom, with disk-only Engram. The GPU 0 temperature after the three API
runs was 58 C. Thermal throttling indicators were not captured, so thermal
equivalence across the runs is unverified.

**Failed attempts retained:** explicit
`--gpu-devices 0 --gpu-vram auto --ssd-streaming` was rejected as
"not compatible with multi-GPU placement" (`server.log`). Removing
explicit placement loaded and exposed `/v1/models`, but its first generation
failed at V4.1 layer 20: "CUDA attention score tile shared-memory opt-in
failed: invalid argument" followed by "CUDA attention decode launch failed:
invalid argument" (`server-retry.log`). The existing
`DS4_CUDA_NO_SCORE_TILE=1` runtime knob avoided that unsupported Turing
kernel path without editing upstream code (`server-fallback.log`). Server
health without valid generation was not counted as a benchmark.

**Infrastructure checks (not model performance measurements):** the
`sm_75` CUDA build completed, and upstream
`make test-deepseek41-cuda CUDA_ARCH=sm_75`,
`make test-engram CUDA_ARCH=sm_75`, and
`make test-deepseek41-gguf CUDA_ARCH=sm_75` passed with CUDA runtime
libraries on the loader path. These synthetic/format tests do not load
the 483 GiB model and alone cannot establish correctness or tokens per second.

The official Q4 GGUF ships in 480,000,000,000- and 38,596,067,328-byte
parts. Only the joined GGUF is executable. The upstream model documentation
describes Q4 as requiring SSD streaming on smaller hosts and targets a
512 GiB Mac for resident operation. In-process multi-GPU CUDA is not
available for V4.1, and CUDA network tensor parallelism currently accepts
only Q2 (`IQ2_XXS` gate/up and `Q2_K` down) weights. DSpark support files
for V4 Flash 0731 are not compatible with V4.1. Consequently no four-GPU
speedup, 1.5-2x speculative gain, or superiority over llama.cpp/vLLM has
been established for this checkpoint on this host.

Sources: [DwarfStar model guide](https://github.com/antirez/ds4/blob/main/docs/MODELS.md#deepseek-v41-flash),
[CUDA guide](https://github.com/antirez/ds4/blob/main/docs/CUDA_MULTI_GPU.md),
[Q4 model card](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf),
and the pinned source and logs above.
