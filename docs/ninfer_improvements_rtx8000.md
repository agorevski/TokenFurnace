# NInfer tuning opportunities for 4x Quadro RTX 8000

## Scope and baseline

This review covers `/home/algore/ninfer-2080ti-22g` at
`6460bf03a86f1288dc7b240d70c30887199099af` plus the locally applied
`NINFER_SM75_INT8_KV_ONLY` patch. Recommendations are hypotheses until measured
on this host; no speedup is claimed here.

Host facts from [`hardware/4x-rtx8000`](../hardware/4x-rtx8000/README.md):

- 4x Quadro RTX 8000, 48 GiB GDDR6 and 72 SMs per GPU (`sm_75`);
- GPUs 0-1 and 2-3 are NVLink pairs; traffic across the pairs uses PCIe;
- 260 W board power limit, persistence mode currently disabled;
- system CUDA toolkit 12.4, while this fork requires CUDA 12.8 or newer
  (`CMakeLists.txt:32-36`).

NInfer presently creates one `DeviceContext` from one `options.device`, then
constructs one target and executor around it
(`src/runtime/engine/engine.cpp:133-166`). The server exposes only
`--device N` (`src/serve/serve_options.cpp:205-213`), and the model cards
explicitly state that multi-GPU execution and distributed serving are absent
(`model-cards/Qwen3.8-27B-NInfer/README.md:100-105`). Thus 192 GiB aggregate
VRAM is **not** one NInfer memory pool. Today, all four GPUs can be used only as
independent single-GPU replicas. NVLink topology has no effect until a
multi-GPU execution design is added.

## Priority recommendations

| Priority | Recommendation | Class | Expected benefit to validate |
|---|---|---|---|
| P0 | Run one server replica per GPU; route requests across four ports | Safe configuration | Aggregate throughput and availability |
| P0 | Build `sm_75` with the local INT8-KV-only option and serve with `--kv-dtype int8` | Safe configuration | Avoid the known pathological software-emulated BF16 decode path |
| P0 | Replace RTX 2080 Ti 68-SM cooperative limits with detected SM count | Low-risk code | Correct legal launch envelope for 72-SM RTX 8000 |
| P1 | Replace the 170-SM sparse-MoE persistent grid with a device-derived grid | Low-risk code | Avoid launching a 510-CTA persistent grid tuned for RTX 5090 on 72 SMs |
| P1 | Retune GQA split counts and launch geometries for 72 SMs | Measurement-driven kernel work | Decode/MTP latency and throughput |
| P1 | Tune context, KV capacity, concurrency, prefill chunk, and graph use rather than carrying over 22 GB settings | Safe configuration/measurement | Better use of 48 GiB and reduced unnecessary reservations |
| P2 | Make graph and KV headroom policies configurable or observation-driven | Low-risk code | More predictable capacity and startup behavior |
| P3 | Add topology-aware multi-GPU execution | Major architecture | Models/capacities beyond one GPU and possible throughput scaling |

## Safe configuration changes

### Use four independent replicas

Start four processes, each with one `--device` value and a distinct port.
Pinning may be expressed either with physical `--device 0..3` or with one
visible GPU per process and device ordinal 0. Keep artifacts and flags
identical for a controlled throughput test. This is the only current path to
use all four GPUs, and it avoids cross-device synchronization entirely.

### Keep the patched build on INT8 KV

The local patch adds `NINFER_SM75_INT8_KV_ONLY` and makes BF16 small-T decode
throw at runtime (`CMakeLists.txt:18-22`,
`src/ops/launcher/gqa_attention_decode.cu:14-20,260-270`). Build with:

```text
-DCMAKE_CUDA_ARCHITECTURES=75
-DNINFER_SM75_INT8_KV_ONLY=ON
```

and always pass `--kv-dtype int8`. This is a correctness/configuration
requirement for that patched binary, not a measured RTX 8000 optimization.
TokenFurnace's NInfer backend and native benchmark reject any other KV dtype
before launching the patched binary.
Resolve the CUDA-toolkit mismatch first by building with a compatible CUDA
12.8+ toolkit; do not silently weaken the fork's version check.

### Use the extra VRAM deliberately

The server defaults to explicit KV capacity equal to `--max-context`; automatic
sizing occurs only when `--kv-capacity auto` is explicitly supplied
(`src/serve/serve_options.cpp:275-281`). Auto sizing:

- subtracts a fixed 1 GiB headroom;
- allocates from current free memory after weights;
- is bounded by `max_context * max_concurrency`
  (`include/ninfer/types.h:20-50`,
  `src/runtime/engine/kv_capacity.cpp:71-132`,
  `src/targets/qwen3_6/impl/runtime/layouts_impl.h:546-565`).

Therefore, 48 GiB will not automatically increase the logical context or
request count. Set the intended `--max-context` and `--max-concurrency`, then
use `--kv-capacity auto` and record the startup memory summary. Start at
concurrency 1 for latency and sweep 2, 4, and 8 for serving throughput. Do not
raise the compile-time cap of 8 without broader work: it is embedded in fixed
arrays, scheduler state, and graph capture (`include/ninfer/types.h:20`;
`src/runtime/engine/concurrent_executor.h:299-333`).

### Control graph and prefill costs

CUDA graphs are enabled by default; `--no-cuda-graph` is the control.
Graph definitions are captured for every batch size through
`max_concurrency`, so setting concurrency higher than the real workload
increases startup work and reserved memory
(`src/targets/qwen3_6/impl/runtime/program_impl.h:1241-1339`).

Use `--prefill-chunk 1024` as the baseline and sweep 512, 2048, and 4096
(valid values are multiples of 128). Larger chunks may improve long-prefill
efficiency but also change workspace pressure; compare identical prompt
lengths.

For stable measurements, enable persistence mode if operational policy permits,
retain the stock 260 W power limit, warm the GPU, and log clocks, temperature,
power, throttling, and PCIe state before and after each run. Do not overclock,
and do not attribute a difference to code when clock or thermal state changed.

## Low-risk code changes

### Detect the actual SM count

The device properties are already queried and retained
(`src/core/device.cu:45-63`), but tuning logic uses only compute capability.
Pass `props.multiProcessorCount` into target/kernel policy selection or expose
it through `DeviceContext`.

The most concrete bug is the cooperative GDN legality calculation. SM75 routes
and validation are hard-coded for an RTX 2080 Ti's 68 SMs:

- 27B route boundaries: 256, 640, and 1408 tokens
  (`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:29-45`);
- resident cooperative grids: 68 CTAs for 27B and 136 CTAs for 35B
  (`.../bf16_gdn_gating_proj_plan.cpp:145-178`).

On 72 SMs, the corresponding *legality* ceilings are 72 CTAs at one CTA/SM and
144 CTAs at two CTAs/SM. For the documented 27B grid formulas, that permits
candidate boundaries up to 384, 768, and 1536 tokens respectively. These are
only legal ranges, not proven optimal routes. Prefer runtime occupancy/device
queries with a conservative fallback over introducing another GPU-name
constant.

### Derive sparse-MoE persistent grids from the device

The 35B sparse-MoE prefill path defines `kRtx5090SmCount = 170` and launches
three blocks per assumed SM, i.e. 510 persistent CTAs
(`src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:250-258,
1167-1178,1206-1233`). An RTX 8000-derived starting point is 72 x 3 = 216
blocks, subject to occupancy verification. Change the host launch count to
`multiprocessor_count * tested_blocks_per_sm`, capped by available work. This is
small code surface and removes an obvious foreign-device assumption.

### Make memory safety margins explicit

The graph allowance is architecture-wide rather than device- or
observation-derived: on SM75 it reserves 64 MiB per ordinary batch and
64/96 MiB per MTP batch class, multiplied by concurrency
(`src/targets/qwen3_6/impl/runtime/layouts_impl.h:643-674`). Graph preparation
later records actual consumption and fails if it exceeds the allowance
(`src/targets/qwen3_6/impl/runtime/program_impl.h:1385-1394`).

Low-risk improvements:

1. add `--kv-headroom-mib` for automatic KV sizing instead of fixing 1 GiB;
2. expose the planned and observed graph bytes prominently in machine-readable
   startup output;
3. optionally cache a conservative observed allowance by GPU UUID, driver,
   model, context, speculation mode, and concurrency;
4. instantiate only reachable batch-size graph families if the serving policy
   can declare them.

Keep a hard minimum margin and fail safely; reclaiming memory is a capacity
optimization until benchmarks show a throughput effect.

## Measurement-driven kernel experiments

1. **GDN route boundaries.** Compare current 68-SM routes against
   device-derived legal routes at token counts around 256/384, 640/768, and
   1408/1536. Record kernel time and correctness; legality alone does not select
   the fastest schedule.
2. **GQA split policy.** `DecodeSplits` is fixed at `85 * scale`
   (`src/ops/kernel/gqa_attention_geometry.cuh:9-22`). The INT8 T=6 5K-8K
   special case explicitly targets a one-wave 170-SM grid and clamps at
   `42 * scale` (`src/ops/launcher/gqa_attention_decode.cu:52-70`).
   Sweep split counts for 72 SMs at each graph boundary (128, 160, 512, 4096,
   8198, and 16390 visible keys), including MTP widths 4-6.
3. **CTA geometry and launch bounds.** Test the existing warp/key-block choices
   in `launch_tc_partial_i8` (`src/ops/launcher/gqa_attention_decode.cu:129-216`)
   with Nsight Compute occupancy, register, shared-memory, and DRAM metrics.
   Do not infer RTX 8000 behavior from either 2080 Ti or 5090 comments.
4. **Sparse-MoE persistent blocks.** Sweep 1, 2, and 3 resident blocks per SM
   and grid multipliers around 72 SMs for 35B prefill sizes 47, 128, 512, 768,
   1024, and 4096.
5. **Graph segmentation.** The graph profile boundaries are fixed around
   attention policy transitions
   (`src/targets/qwen3_6_27b/impl/variant.cpp:118-151`). Compare current graphs
   with eager execution and with coarser/finer profiles, measuring startup
   time, graph memory, and steady-state decode.

## Major architectural work: multi-GPU

Tensor or pipeline parallelism is not a flag-sized change. It requires weight
sharding, per-device arenas/KV state, multi-device scheduling, NCCL or peer
communication, graph capture across devices, failure handling, and artifact
metadata for shard placement.

If implemented, develop in this order:

1. two-GPU tensor parallelism confined to NVLink pair 0-1 (then 2-3);
2. two independent TP2 replicas for aggregate serving;
3. compare four-GPU TP4, which crosses PCIe, against topology-aware
   TP2 x PP2, where tensor collectives stay within each NVLink pair and only
   pipeline activations cross pairs.

More GPUs may reduce usable single-request performance because communication
can dominate decode. Promote no topology without measurement.

## Validation plan

For every proposed change, use the exact same artifact, commit, prompt/output
tokens, context, concurrency, cache state, CUDA-graph mode, and thermal window.
Preserve raw output and validate generated text.

| Change | Required comparisons | Primary evidence |
|---|---|---|
| Four replicas | 1 vs 2 vs 4 single-GPU servers at fixed per-server concurrency | Aggregate tok/s, p50/p95 latency, per-GPU utilization |
| KV/context tuning | Explicit vs auto; contexts 8K/32K/64K/128K/256K where valid; concurrency 1/2/4/8 | Load success, resolved KV tokens, free VRAM, decode/prefill throughput |
| CUDA graphs | graph on/off at each active concurrency and context class | Startup time, graph observed/planned bytes, decode latency |
| Prefill chunk | 512/1024/2048/4096 with 512/4K/32K prompts | Prefill tok/s, workspace peak, OOM/correctness |
| 72-SM GDN routes | Current vs device-derived boundaries near every crossover | Per-op and end-to-end prefill time |
| GQA split/geometry | Current vs candidate splits at short through 128K contexts, MTP0 and MTP3/5 | Per-token decode, accepted drafts, occupancy, DRAM throughput |
| Sparse-MoE grid | 170-SM constant vs 72-SM-derived grids | 35B prefill time, occupancy, tail duration |
| Multi-GPU prototype | 1 GPU, TP2 within a pair, two TP2 replicas, TP4, TP2 x PP2 | Latency, aggregate tok/s, NCCL/PCIe/NVLink traffic, VRAM |

Report cold startup separately from warmed steady state. A candidate passes only
if correctness is unchanged and its benefit repeats across enough runs to
exceed run-to-run variation.
