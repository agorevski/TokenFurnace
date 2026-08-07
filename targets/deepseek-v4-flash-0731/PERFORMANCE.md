# Q4 and DSpark performance tests

This file records controlled performance experiments for
DeepSeek-V4-Flash-0731 `UD-IQ4_XS` on four Quadro RTX 8000 GPUs.

## Conclusions

The configuration space is not exhausted, but the current defaults are the
best general-purpose settings found so far:

```text
CUDA_SCALE_LAUNCH_QUEUES=4x
--batch-size 8192
--ubatch-size 2048
--parallel 1
--split-mode layer
--flash-attn on
--spec-draft-n-max 2
--spec-draft-p-min 0
```

Deeper DSpark drafts can be faster when output is highly predictable, but are
not safe as a global default:

| Profile | Settings | Best use | Observed decode |
|---|---|---|---:|
| Balanced default | `n_max=2`, `p_min=0` | Mixed coding and prose | 27.19 tok/s on natural prose |
| Predictable output | `n_max=5`, `p_min=0.40` | Lists, repetitive formats, boilerplate | 44.81 tok/s |
| High-confidence long context | `n_max=4`, `p_min=0.50` | Long prompt with a short predictable answer | 34.24 tok/s |

The specialized profiles are workload-dependent. On the natural-prose test,
the predictable profile fell to 22.87 tok/s, below both the balanced profile
and plain decoding.

## Test system

Tests in this section were run on 2026-08-06.

| Component | Value |
|---|---|
| GPUs | 4x NVIDIA Quadro RTX 8000, 48 GiB each |
| GPU topology | NVLink pairs 0-1 and 2-3; PCIe between pairs |
| GPU power limit | 260 W per GPU, the card maximum |
| CPU | Intel Xeon W-2295, 18 cores / 36 threads |
| RAM | 251 GiB |
| CUDA toolkit | 12.4 |
| NVIDIA driver | 580.173.02 |
| llama.cpp commit | `15586e2d7165570fb3aa7c26e0d442e289ef69de` |
| llama.cpp build | CUDA, NCCL, Turing `SM 75` |
| Target model | `DeepSeek-V4-Flash-0731-UD-IQ4_XS` |
| Draft model | `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` |
| Context size | 65,536 |
| Parallel slots | 1 |
| Sampling | temperature 0, seed 1234 |

All tests used GPU Flash Attention, full GPU layer offload, layer split,
`--tensor-split 1,1,1,1`, fit target 2048 MiB, and
`CUDA_SCALE_LAUNCH_QUEUES=4x`.

## Workloads

The server `/completion` endpoint reported all token counts and timings.
Prompt caching was disabled for each request.

| Workload | Prompt tokens | Generated tokens | Purpose |
|---|---:|---:|---|
| `predictable-256` | 31 | 256 | Continue the integer sequence from 1 through 256 |
| `natural-256` | 29 | 232 | Explain blue skies and red sunsets in technical prose |
| `long-context-64` | 15,018 | 64 | Extract a simple fact from a repeated reference passage |

The natural response ended at EOS after 232 tokens. The predictable workload
is intentionally favorable to speculative decoding; it is an upper-bound test,
not a representative coding-agent workload.

Short-prompt prefill results are not used for conclusions. The first request
after loading DSpark includes lazy initialization overhead, which particularly
distorts the 31-token workload. The 15K prompt is long enough for meaningful
prefill comparison.

## Draft-depth sweep

`p_min` was zero for this sweep.

### Decode throughput

| Draft maximum | Predictable | Natural prose | Long context |
|---:|---:|---:|---:|
| Disabled | 23.76 | 23.84 | 22.81 |
| 1 | 30.70 | 26.03 | 27.37 |
| **2** | **37.35** | **27.32** | 30.22 |
| 3 | 41.21 | 24.21 | **31.65** |
| 4 | 41.20 | 21.12 | 17.77 |

Values are tokens per second. Draft maximum 2 is the only tested depth that
improved every workload substantially. Depth 3 adds speed only when acceptance
is high. Depth 4 is actively harmful when low-confidence tokens are not
filtered.

### Draft acceptance

| Draft maximum | Predictable | Natural prose | Long context |
|---:|---:|---:|---:|
| 1 | 100.0% | 63.5% | 82.4% |
| 2 | 100.0% | 56.2% | 74.0% |
| 3 | 100.0% | 41.0% | 67.7% |
| 4 | 100.0% | 36.2% | 29.8% |

Acceptance rate alone does not fully predict throughput because each target
verification step has a different draft width. It does explain why an
aggressive depth can become slower than plain decoding.

## Confidence-threshold sweep

`--spec-draft-p-min` stops a DSpark block at the first position below the
specified confidence. This makes deeper drafts less damaging, but it did not
produce one profile that beat `n_max=2` everywhere.

### Draft maximum 3

| `p_min` | Predictable | Natural prose | Long context |
|---:|---:|---:|---:|
| 0.00 | 41.21 | 24.21 | 31.65 |
| 0.10 | 41.44 | 24.06 | 31.69 |
| 0.20 | 41.30 | 24.48 | 31.02 |
| 0.30 | 41.35 | 25.30 | 32.27 |
| 0.40 | **41.49** | **25.69** | **32.70** |

Higher confidence gating recovered some natural-prose performance, but the
balanced `n_max=2` profile remained 6.3% faster on that workload.

### Draft maximum 4 and 5

| Draft maximum | `p_min` | Predictable | Natural prose | Long context |
|---:|---:|---:|---:|---:|
| 4 | 0.00 | 41.20 | 21.12 | 17.77 |
| 4 | 0.20 | 40.36 | 22.47 | 29.50 |
| 4 | 0.30 | 41.08 | 21.72 | 32.03 |
| 4 | 0.40 | 41.35 | 24.11 | 32.09 |
| 4 | 0.50 | 41.11 | 23.36 | **34.26** |
| 5 | 0.40 | **43.79** | 22.76 | 32.73 |

Values are tokens per second. `n_max=5` reaches the DSpark model's five-token
block limit and gave the highest single-run predictable-output result in this
sweep. The repeated confirmation above is the preferred comparison.

For the long-context workload, `n_max=4`, `p_min=0.50` was 50.2% faster than
plain decoding and 13.4% faster than the balanced profile. Its 97.8% accepted
draft rate shows why this setting worked for that prompt.

## Three-run confirmation

The balanced and two specialized profiles were each loaded once and every
workload was then repeated three times. Decode results were stable:

| Profile | Predictable mean (range) | Natural mean (range) | Long-context mean (range) |
|---|---:|---:|---:|
| Balanced | 37.70 (37.15-38.06) | **27.19 (27.12-27.23)** | 30.29 (30.18-30.38) |
| Predictable | **44.81 (44.09-45.39)** | 22.87 (22.81-22.93) | 32.70 (32.52-32.80) |
| Long-context | 41.97 (41.18-42.37) | 23.36 (23.34-23.38) | **34.24 (34.17-34.32)** |

Values are tokens per second. The predictable profile improved its target
workload by 18.9% over the balanced profile and 88.6% over plain Q4. The
long-context profile improved its target workload by 13.0% over balanced and
50.1% over plain Q4. Neither specialized profile is competitive on the
natural-prose workload.

## Prefill impact

On the 15,018-token prompt:

| Configuration | Prompt throughput |
|---|---:|
| Plain Q4 | 503.21 tok/s |
| DSpark depth 1 | 473.76 tok/s |
| DSpark depth 2 | 472.60 tok/s |
| DSpark depth 3 | 472.45 tok/s |
| DSpark depth 4 | 473.42 tok/s |

Loading DSpark reduced long-prompt prefill by about 6%. For workloads that
generate only a few tokens after very large prompts, plain decoding can
therefore have lower end-to-end latency even when DSpark decode tok/s is higher.

## Recommended profiles

Use the balanced profile unless the output distribution is known in advance:

```bash
./scripts/serve-model.sh deepseek-v4-flash-0731 q4-balanced
```

For deterministic lists, repetitive transforms, or rigid boilerplate, test:

```bash
./scripts/serve-model.sh deepseek-v4-flash-0731 q4-predictable
```

For a large stable prompt followed by a short, high-confidence answer, test:

```bash
./scripts/serve-model.sh deepseek-v4-flash-0731 q4-long-context
```

Do not select a specialized profile from synthetic results alone. Measure the
actual agent or application workload and inspect `draft acceptance` in the
server log.

## Previously tested settings

Earlier experiments, summarized in the README, established these defaults:

- `CUDA_SCALE_LAUNCH_QUEUES=4x` substantially improves prompt processing.
- Q4 prefill is best at `--batch-size 8192 --ubatch-size 2048`.
- Default CUDA MMQ kernels beat forced cuBLAS.
- Layer split works; row and tensor split fail to load this model.
- Forced P2P and a larger peer maximum batch changed performance by less than
  1%.
- `--parallel 1` is best for single-session throughput.
- Quantized KV cache is not currently recommended because of a DeepSeek-V4
  correctness issue in the tested llama.cpp version.

## Remaining experiments

The search is not mathematically exhaustive. Useful future work includes:

1. Replay real coding-agent traces and select draft policy by measured
   acceptance and end-to-end latency.
2. Test whether application clocks or persistence mode reduce run-to-run
   variance. The GPUs were already at their maximum 260 W power limit.
3. Re-test after llama.cpp or DSpark changes; both implementations are moving
   quickly, so optimal thresholds may change.
4. Compare `UD-IQ3_XXS` quality and throughput against Q4 on a representative
   coding benchmark.

The largest remaining practical gains are likely workload-aware draft policy,
prompt-cache reuse, fewer generated reasoning tokens, or newer GPU
architecture, not another universal Q4 launch flag.
