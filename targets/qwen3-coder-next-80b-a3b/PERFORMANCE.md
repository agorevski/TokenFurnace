# Qwen3-Coder-Next performance results

> **This page now mixes two clearly separated kinds of data.** The
> [Measured results](#measured-results) section records a real RTX 8000
> load-and-generate run of the llama.cpp `Q4_K_M` GGUF (2026-08-07). The
> later sections labeled **PROJECTED** remain unverified planning targets —
> chiefly the vLLM FP16 rows, which are still unmeasured because the BF16
> download is incomplete. Do not treat any PROJECTED cell as fact.

## Measured results

Recorded 2026-08-07 from a real load-and-generate run of the llama.cpp `Q4_K_M`
GGUF on this host. The vLLM FP16 path remains **unmeasured** (the BF16 download
is still incomplete: shards `00003` and `00039` of 40 are missing), so every
vLLM number on this page stays in the PROJECTED sections below.

Caveat: no raw artifact files (JSON logs, server dumps) were persisted; the
values below were transcribed from `llama-bench` and `llama-server` console
output during the run.

### Test system and conditions

| Component | Value |
|---|---|
| Run date | 2026-08-07 |
| GPUs | 4x NVIDIA Quadro RTX 8000, 48 GiB each |
| GPU topology | NVLink pairs 0-1 and 2-3; PCIe between pairs |
| llama.cpp build | `b10298`, commit `15586e2` (`15586e2d7165570fb3aa7c26e0d442e289ef69de`), CUDA + NCCL, sm_75 |
| Model | Unsloth `Qwen3-Coder-Next-Q4_K_M.gguf`, single file, **48,528,320,544 bytes** (45.20 GiB) |
| Flash attention | on |
| GPU offload | all layers on GPU (`-ngl 99`) |
| Batch / ubatch | 8192 / 2048 |
| `llama-bench` repetitions | 5 (`-r 5`) |

### Native `llama-bench` throughput

Common command pattern (MODE and SPLIT vary per row):

```bash
CUDA_VISIBLE_DEVICES=... llama-bench \
  -m /home/algore/models/qwen3-coder-next-80b-a3b-gguf/Qwen3-Coder-Next-Q4_K_M.gguf \
  -ngl 99 -sm MODE -ts SPLIT -fa on -b 8192 -ub 2048 -r 5 \
  -pg 512,0 -pg 4096,0 -pg 0,128 -o json
```

Values are tokens/sec, mean ± stddev over 5 repetitions.

| Configuration | 512 prefill | 4096 prefill | 128 decode |
|---|---:|---:|---:|
| 4-GPU, layer split, `-ts 1,1,1,1` | 1249.03 ± 35.71 | **2820.03 ± 13.69** | 88.45 ± 5.26 |
| 2-GPU (0,1), layer split, `-ts 1,1` | 1256.25 ± 12.97 | 2098.91 ± 11.36 | **100.43 ± 0.14** |
| 4-GPU, **row** split | failed to load | failed to load | failed to load |
| 2-GPU (0,1), **row** split | failed to load | failed to load | failed to load |

**Row split is a measured incompatibility, not an untested option.** Both the
4-GPU and 2-GPU row-split runs failed to load the model on this `b10298` /
sm_75 build. The only diagnostic emitted was the generic
`llama_bench: error: failed to load model ...` — no specific kernel or tensor
error was surfaced. Do **not** recommend `SPLIT_MODE=row` for this model on this
build.

#### Interpretation

- **Long-prefill throughput favors 4 GPUs.** At 4096-token prefill the 4-GPU
  layer split is **34.4% faster** than the 2-GPU layer split
  (2820.03 vs 2098.91 tok/s).
- **Decode favors the 2-GPU NVLink pair.** The 2-GPU layer split decodes
  **13.5% faster** than 4 GPUs (100.43 vs 88.45 tok/s) and is far more stable
  (± 0.14 vs ± 5.26), because it keeps all traffic on the NVLink-bonded pair
  0-1 instead of crossing the PCIe host bridge.
- **512-token prefill is essentially tied** — the 2-GPU split is only 0.6%
  faster (1256.25 vs 1249.03 tok/s), within noise.
- **Conclusion:** the 2-GPU NVLink pair (`llama-cpp-q4km-2gpu`) is the fastest
  single-request DECODE and is now the **target default**, because interactive
  coding turns are decode-bound once the stable system prompt is prefix-cached.
  The 4-GPU `llama-cpp-q4km` layer split remains the right **explicit** choice
  for long, prefill-dominated prompts (+34.4% at 4096-token prefill). The 2-GPU
  profile drives GPU visibility through `CUDA_VISIBLE_DEVICES=0,1` and a matching
  `TENSOR_SPLIT=1,1`; because profile env files load last, the override is
  scoped to that profile only. The [tuning sweep below](#2-gpu-01-tuning-sweep--selecting-the-default-2026-08-07-pm)
  confirms none of the experimental knobs beats this plain layer split.

### 2-GPU (0,1) tuning sweep — selecting the default (2026-08-07 PM)

After the 2-GPU layer split was chosen as the fastest-decode configuration, a
controlled `llama-bench` sweep (`-r 5`, same b10298/sm_75 build, GPUs 0,1,
`-sm layer -ts 1,1 -b 8192`) tested every plausible safe knob to confirm nothing
faster or safer exists before promoting it to the **target default**. The plain
layer split with `UBATCH_SIZE=2048` won; **every experimental knob was neutral or
harmful and is not adopted.**

> **Thermal caveat for this session.** Under sustained back-to-back runs GPU 0
> reached 87-88 °C and the driver applied SW thermal slowdown (throttle reason
> `0x20`), dropping its SM clock from 1905 to ~1395 MHz. Absolute figures below
> therefore run **~15-20% under** the cooler-silicon `pp4096 = 2098` recorded in
> the table above, and this custom build's `llama-bench` runs each test set twice
> (the second pass is hotter/slower). Values quoted are the **first (coolest)
> pass**; comparisons are only made **within the same thermal window**, which is
> what determines the knob decisions.

| Experiment (vs plain layer, ub2048) | 512 prefill | 4096 prefill | 8192 prefill | 128 decode | Decision |
|---|---:|---:|---:|---:|---|
| **Plain layer split, `ub2048` (selected)** | **1242.85 ± 17.42** | **1705.48 ± 42.91** | **1529.17 ± 13.37** | **92.72 ± 0.42** | **Adopted as default** |
| `CUDA_SCALE_LAUNCH_QUEUES=4x` | 1229.08 ± 17.52 | 1657.90 ± 50.80 | — | 91.23 ± 0.62 | Reject — within noise of unset |
| `-sm tensor` (2-GPU tensor split) | 1211.95 ± 16.10 | 1617.31 ± 44.23 | 1481.17 ± 13.83 | 80.58 ± 0.44 | Reject — decode −13%, prefill no better |
| `UBATCH_SIZE=512` | 1227.55 ± 18.24 | 1107.17 ± 36.97 | — | 89.90 ± 0.65 | Reject — long prefill much slower |
| `UBATCH_SIZE=1024` | 1072.31 ± 7.42 | 1328.60 ± 3.08 | — | 88.78 ± 0.79 | Reject — long prefill slower |
| `UBATCH_SIZE=4096` | — | — | — | — | **Fails** — cannot create compute context |
| `THREADS=8, POLL=50` | — | — | — | 97.08 ± 0.77 | Noise — confounded by test ordering/thermals |
| `THREADS=18, POLL=50` | — | — | — | 92.80 ± 0.21 | Noise — decode is GPU-bandwidth-bound |
| `THREADS=8/18, POLL=100` | — | — | — | 88.88–91.35 | Noise — no reproducible win |
| `GGML_CUDA_P2P=1` | — | 1703.57 ± 52.16 | — | 91.96 ± 0.43 | Reject — no gain; upstream warns of corruption/crashes |

Native `llama-bench` command pattern for the sweep (values vary per row):

```bash
CUDA_VISIBLE_DEVICES=0,1 llama-bench \
  -m /home/algore/models/qwen3-coder-next-80b-a3b-gguf/Qwen3-Coder-Next-Q4_K_M.gguf \
  -ngl 99 -sm layer -ts 1,1 -fa on -b 8192 -ub 2048 -r 5 \
  -p 512 -p 4096 -p 8192 -n 128
# knob overrides tested: CUDA_SCALE_LAUNCH_QUEUES=4x (env) | -sm tensor |
#   -ub 512 / 1024 / 4096 | -t 8/-t 18 --poll 50/100 | GGML_CUDA_P2P=1 (env)
```

**Decisions applied to the profiles/backend.**

- **`UBATCH_SIZE=2048` kept.** Long-prefill throughput rises with ubatch
  (`pp4096` 1107 → 1329 → 1540 across 512/1024/2048) but **4096 fails to create
  the compute context** on 2 GPUs, so 2048 is the optimal working value —
  matching the value the profile already used.
- **`-sm tensor` rejected.** It loads on 2 GPUs (F16 KV + FA + NCCL), but the
  per-layer all-reduce over the 2-link NVLink bond on Turing costs **−13% decode
  (80.6 vs 92.7 tok/s)** with no prefill benefit, so it is not adopted and text
  correctness was not pursued further.
- **`CUDA_SCALE_LAUNCH_QUEUES` explicitly unset for the 2-GPU profile.** `4x`
  was within noise but slower in both measured rows, so the profile restores
  CUDA's stock behavior while the backend preserves DeepSeek's measured `4x`
  default.
- **`THREADS`/`POLL` left at server defaults.** Decode is GPU-bandwidth-bound
  (~3B active params, all resident in VRAM); the apparent `t8 > t18` gap tracks
  test ordering and GPU temperature, not a real dispatch win, so no knob is set.
- **`GGML_CUDA_P2P=1` left unset.** No measured gain and upstream warns of
  corruption/crashes, so it is not enabled by default.
- **F16 KV kept.** Q4_K_M already leaves large VRAM headroom; no long-context Q8
  test justified lowering KV quality for unused VRAM.

The winning configuration is exactly the plain 2-GPU layer split the
`llama-cpp-q4km-2gpu` profile already encoded, which is now
`DEFAULT_PROFILE` for the Qwen target.

### Actual 4-GPU server run (cold vs warm, prefix caching)

Served with the `llama-cpp-q4km` profile (4-GPU layer split). The prompt was the
deterministic example from llama.cpp `docs/function-calling.md`:
**7590 prompt tokens, 256 generated tokens** per request, sent three times
back-to-back on one slot.

| Request | E2E latency | E2E output tok/s | Prompt tokens evaluated | Native prefill | Native decode tok/s |
|---|---:|---:|---:|---:|---:|
| Cold | 6.727 s | 38.05 | 7590 | 7590 / 3.750 s = **2024.00 tok/s** | 87.02 |
| Warm 1 | 3.268 s | 78.34 | 4 (in 86.35 ms) | — (prefix reused) | 81.28 |
| Warm 2 | 3.253 s | 78.70 | 4 (in 81.31 ms) | — (prefix reused) | 81.23 |

- **Prefix caching works and roughly halves warm end-to-end latency** for a
  repeated long prompt: 6.727 s cold → ~3.26 s warm (E2E output rate 38.05 →
  ~78.5 tok/s). On the warm requests only **4** of 7590 prompt tokens were
  evaluated (86.35 ms / 81.31 ms), the rest served from the cached prefix.
- Server logs reported **LCP similarity 1.000, `f_keep=0.967`, 7586 prompt
  tokens reused (99.95%)**.
- The native decode rate (~81–87 tok/s) is consistent with the `llama-bench`
  4-GPU decode figure (88.45 tok/s); the small warm dip is expected under the
  full server path.
- **Caching verification limitation:** `/metrics` exposed aggregate timing and
  token counters but **no explicit cache-hit counter**, so cache reuse is
  confirmed from the per-request prompt-eval collapse and the server log
  (`LCP similarity` / `n_past` reuse) rather than a dedicated Prometheus gauge.

### Load time and GPU utilization

| Configuration | Load time | Per-GPU VRAM | Utilization during generation |
|---|---:|---|---|
| 4-GPU layer | ~10.06 s | GPU0 14703, GPU1 13283, GPU2 13281, GPU3 13169 MiB | all four active, ~26/23/23/25% |
| 2-GPU (0,1) layer | — | ~24.6 / 24.0 GiB | both GPUs reached 100% |

The 4-GPU split spreads the 45 GiB of weights thinly (~13–15 GiB per card) and
leaves each GPU lightly loaded (~25%) during decode — decode is bandwidth-bound
per layer, and splitting across four cards plus the cross-pair PCIe hop limits
per-request decode. The 2-GPU split packs ~24 GiB per card and saturates both,
which is why it decodes faster.

## How to read the PROJECTED targets below

> **Everything from here down is PROJECTED**, except where a row explicitly
> cites a measured value. The llama.cpp `Q4_K_M` projections have now been
> confirmed by the [Measured results](#measured-results) above; the vLLM FP16
> projections remain unverified.

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
`llama-cpp-q4km-2gpu` (2-GPU, the **default**) and the explicit `llama-cpp-q4km`
(4-GPU) target. The llama.cpp rows are now **confirmed by measurement** (see
[Measured results](#measured-results)); the vLLM row stays PROJECTED.

| Runtime / quant | Metric | Projected realistic | Projected stretch | Measured 2026-08-07 | Notes |
|---|---|---:|---:|---:|---|
| llama.cpp `Q4_K_M` | decode tok/s | 30-70 | 80-120 | **88.5** (4-GPU) / **100.4** (2-GPU) | `llama-bench`, 128 decode; sm_75 MMQ kernels |
| llama.cpp `Q4_K_M` | short-prompt prefill tok/s | 500-1500 | ~2000 | **1249** (512-tok, 4-GPU) | measured 4096-tok prefill reaches 2820 tok/s |
| vLLM FP16 (`half`) | decode tok/s | 8-15 | — | not measured | Bandwidth-bound on 148 GiB resident weights (PROJECTED) |

Q4 decode is projected several times faster than FP16 for a single stream
because decode is dominated by weight-read bandwidth and Q4 reads roughly a
quarter of the bytes. llama.cpp with the Unsloth `Q4_K_M` GGUF
(`Qwen3-Coder-Next-Q4_K_M.gguf`, 45.20 GiB single file) is therefore the
recommended path for the *fastest single request*.

## Aggregate serving throughput (many concurrent requests) — PROJECTED

Goal: maximum total tokens/sec across concurrent sessions and the fastest bulk
prefill. This is what `vllm-fp16-tp2pp2` targets. **These vLLM FP16 rows are
unmeasured projections** (the BF16 download is incomplete).

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
3. **For llama.cpp, use layer split; row split is a measured failure.** Layer
   split is the default and loads reliably. Row split was **measured to fail to
   load** this model on both 4 and 2 GPUs (2026-08-07, `b10298`/sm_75); it also
   failed to load DeepSeek-V4. Do not use `SPLIT_MODE=row` here. For fastest
   single-request decode — the **default** — restrict to the NVLink pair 0-1 via
   the `llama-cpp-q4km-2gpu` profile (measured +13.5% decode over 4-GPU); select
   the explicit 4-GPU `llama-cpp-q4km` profile for long-prompt prefill (measured
   +34.4% at 4096-token prefill). A 2-GPU tuning sweep (2026-08-07) confirmed
   `-sm tensor`, `CUDA_SCALE_LAUNCH_QUEUES=4x`, `UBATCH_SIZE!=2048`,
   `THREADS`/`POLL`, and `GGML_CUDA_P2P=1` are all neutral or harmful and are not
   adopted.
4. **Extend KV headroom cheaply.** `Q4_K_M` leaves large VRAM headroom; try
   `CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0` to grow usable context toward the
   262,144 native limit. Verify correctness first.
5. **Set concurrency deliberately.** Use `--parallel 1` (llama.cpp) for latency;
   raise vLLM `MAX_NUM_SEQS` for aggregate throughput. They optimize different
   metrics and trade off against each other.
6. **Do not transfer FP8/Hopper/Blackwell benchmarks to Turing.** They do not
   apply to sm_75.

## Prompt / prefix caching for coding agents

Coding-agent CLIs resend a large, stable system prompt + tool schema every turn.
Caching that prefix removes almost all repeated prefill for short follow-ups, so
it is the single biggest interactive-latency win once a session is warm. It does
**not** speed up the first (cache-cold) request or the decode phase.

- **llama.cpp** (`llama-cpp-q4km`): explicit `--cache-prompt`, `--cache-ram
  32768`, `--cache-idle-slots`, `--cache-reuse 256`, `--slot-prompt-similarity
  0.01`, `--metrics`. 32 GiB of host-RAM cache is safe here — the 45 GiB Q4_K_M
  weights are resident in VRAM, not host RAM, so the cache only competes with OS
  page cache on the 251 GiB host. Do not raise toward `--cache-ram -1` without
  monitoring host RAM.
- **vLLM** (`vllm-fp16-*`): `--enable-prefix-caching --mamba-cache-mode align
  --prefix-caching-hash-algo sha256`. `align` is experimental and specific to the
  GatedDeltaNet/Mamba layers; KV dtype stays FP16 on Turing.

Effect on the numbers above: prefix caching improves **warm-session
time-to-first-token / effective prefill** for repeated prompts; it does not
change the steady-state decode tok/s or the cache-cold prefill figures. This is
now **measured** (see [Measured results](#measured-results)): a repeated
7590-token prompt went from 6.727 s cold to ~3.26 s warm, with only 4 prompt
tokens re-evaluated on the warm turns (99.95% prefix reuse), while decode held
at ~81 tok/s. Record cache-warm and cache-cold separately.

To measure a clean cache-cold prefill baseline, copy the profile and set
`CACHE_PROMPT=0` for llama.cpp, or set `ENABLE_PREFIX_CACHING=0`,
`MAMBA_CACHE_MODE=""`, and `PREFIX_CACHING_HASH_ALGO=""` for vLLM. Profile files
load after `config.env`, so a dedicated benchmark profile is required. Then
re-enable and re-send the identical prompt to observe the warm delta via
`/metrics` (see the target README verification commands). Exact vLLM
prefix-cache metric names must be confirmed on this host's running server before
being cited.

## Validation checklist before trusting any number

- [x] Load-and-generate test: llama.cpp runs `qwen3_next` on this `b10298`,
      sm_75 build (confirmed 2026-08-07).
- [x] Record decode and prefill separately with server-reported token counts,
      and the cache-warm vs cache-cold delta (confirmed 2026-08-07, llama.cpp
      `Q4_K_M`).
- [ ] Complete the BF16 download (for vLLM profiles); shards `00003` and
      `00039` of 40 are still missing.
- [ ] Sweep the still-**unmeasured** vLLM paths: `vllm-fp16-tp2pp2` vs
      `vllm-fp16-tp4`.
- [ ] Replace each remaining PROJECTED vLLM cell above with a measured value and
      its run reference; the llama.cpp cells are done.
