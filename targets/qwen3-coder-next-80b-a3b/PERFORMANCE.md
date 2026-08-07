# Qwen3-Coder-Next performance targets

> **All numbers on this page are PROJECTED PLANNING TARGETS, not measured
> results.** No validated RTX 8000 baseline for Qwen3-Coder-Next has been
> recorded yet. Every value is an unverified projection derived from the
> hardware envelope and comparable public runs. Replace each cell with a
> measured result (and cite the run) before treating it as fact.

## Measured results

None yet. The BF16 download is incomplete (shards `00003` and `00039` of 40 are
missing), and the GGUF build has not been load-tested for `qwen3_next` on this
host. This section stays empty until a real load-and-generate run is recorded
the way [DeepSeek PERFORMANCE.md](../deepseek-v4-flash-0731/PERFORMANCE.md)
records its runs (server-reported token counts and timings, fixed seed).

## How to read these targets

The "fastest tokens/sec" question has no single honest number. Speed depends on
five axes that are kept separate below:

1. **Metric** — single-stream *decode* latency vs single-stream *prefill*
   throughput vs *aggregate* server throughput under concurrency.
2. **Runtime + quantization** — llama.cpp GGUF `Q4_K_M` vs vLLM FP16.
3. **Concurrency** — 1 request vs many in-flight requests (`--parallel` /
   `--max-num-seqs`).
4. **Prompt length** — short prompts hide prefill cost; long prompts expose it.
5. **Confidence** — *realistic* (expected on a first competent tuning pass) vs
   *stretch* (best case if kernels and topology cooperate).

These projections reconcile two internal research passes that appeared to
conflict. They do not: one measured the FP16 aggregate-serving envelope, the
other the Q4 single-stream latency envelope. They describe different runtimes
and different metrics, so both are reproduced, each in its own row.

## Single-stream latency (1 request, lowest latency)

Goal: fastest tokens/sec for one interactive coding-agent session. This is what
`llama-cpp-q4km` targets.

| Runtime / quant | Metric | Realistic | Stretch | Notes |
|---|---|---:|---:|---|
| llama.cpp `Q4_K_M` | decode tok/s | 30-70 | 80-120 | Fewer bytes/token read; sm_75 MMQ kernels |
| llama.cpp `Q4_K_M` | short-prompt prefill tok/s | 500-1500 | ~2000 | Prompt <= a few hundred tokens |
| vLLM FP16 (`half`) | decode tok/s | 8-15 | — | Bandwidth-bound on 148 GiB resident weights |

Q4 decode is projected several times faster than FP16 for a single stream
because decode is dominated by weight-read bandwidth and Q4 reads roughly a
quarter of the bytes. llama.cpp with the Unsloth `Q4_K_M` GGUF
(`Qwen3-Coder-Next-Q4_K_M.gguf`, 45.20 GiB single file) is therefore the
recommended path for the *fastest single request*.

## Aggregate serving throughput (many concurrent requests)

Goal: maximum total tokens/sec across concurrent sessions and the fastest bulk
prefill. This is what `vllm-fp16-tp2pp2` targets.

| Runtime / quant | Metric | Realistic | Stretch | Notes |
|---|---|---:|---:|---|
| vLLM FP16 | aggregate decode tok/s | 180-320 | — | Summed across concurrent sequences |
| vLLM FP16 | aggregate prefill tok/s | 1800-3000 | — | Large batched prompts, continuous batching |

Aggregate throughput is a *sum over concurrent requests*; it is not achievable
by any single request and must never be quoted as a per-user speed.

## Topology-driven suggestions for the fastest configuration

1. **Pick the runtime by goal.** Lowest single-request latency → llama.cpp
   `Q4_K_M`. Highest aggregate/prefill throughput → vLLM FP16.
2. **Map parallelism onto the NVLink pairs.** GPUs 0-1 and 2-3 each have a
   2-link NVLink bond (about 51.6 GB/s total per pair); the cross-pair path is
   PCIe host bridge only. `TP2 x PP2` keeps every tensor-parallel all-reduce on
   NVLink and sends only pipeline activations across PCIe, so it is projected to
   beat topology-agnostic `TP4` (`vllm-fp16-tp4`). Benchmark both.
3. **For llama.cpp, benchmark split modes.** Start with `--split-mode layer`
   (default, most likely to load), then compare `SPLIT_MODE=row`. Row/tensor
   split failed to load DeepSeek-V4; whether they load `qwen3_next` is unknown
   and must be tested, not assumed.
4. **Extend KV headroom cheaply.** `Q4_K_M` leaves large VRAM headroom; try
   `CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0` to grow usable context toward the
   262,144 native limit. Verify correctness first.
5. **Set concurrency deliberately.** Use `--parallel 1` (llama.cpp) for latency;
   raise vLLM `MAX_NUM_SEQS` for aggregate throughput. They optimize different
   metrics and trade off against each other.
6. **Do not transfer FP8/Hopper/Blackwell benchmarks to Turing.** They do not
   apply to sm_75.

## Validation checklist before trusting any number

- [ ] Complete the BF16 download (for vLLM profiles) or the Unsloth
      `Qwen3-Coder-Next-Q4_K_M.gguf` single-file GGUF (for llama.cpp).
- [ ] Load-and-generate test: confirm llama.cpp actually runs `qwen3_next` on
      this `b10298`, sm_75 build.
- [ ] Record decode and prefill separately, with server-reported token counts,
      a fixed seed, and prompt-cache disabled.
- [ ] Sweep `llama-cpp-q4km` vs `vllm-fp16-tp2pp2` vs `vllm-fp16-tp4`.
- [ ] Replace each projected cell above with the measured value and its run
      reference; move confirmed results into "Measured results".
