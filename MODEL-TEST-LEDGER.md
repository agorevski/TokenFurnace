# Model test ledger

This ledger tracks the model list supplied on 2026-09-09 against the
four-RTX-8000 workstation. A model is tested only when no prior measured target
exists and an artifact has a credible path within 192 GiB aggregate VRAM.

| Model | Exact artifact | Fit assessment | Status | Evidence / next action |
|---|---|---|---|---|
| Spark-X2.5-4B | `XHToken/Spark-X2.5-4B-GGUF` `Q8_0` | Fits one 48 GiB GPU | **Passed** | 65,536 context, 4,410.82/4,665.82 tok/s prefill, 103.59 tok/s native decode, 85.19 tok/s API output |
| Ternary Bonsai 27B | `prism-ml/Ternary-Bonsai-27B-gguf` `PQ2_0` | Fits one 48 GiB GPU | **Passed** | 65,536 context, 862.83/824.25 tok/s prefill, 42.27 tok/s native decode, 34.96 tok/s thinking-mode API output |
| Gemma 4 12B | `unsloth/gemma-4-12b-it-GGUF` `Q4_K_M` plus `mmproj` | Fits one 48 GiB GPU | **Passed** | 65,536 context, 1,706.04/1,604.71 tok/s prefill, 58.57 tok/s native decode, 53.68 tok/s API output; red/blue image tests passed |
| Qwen3.8-27B | `unsloth/Qwen3.8-27B-GGUF` and `Qwen/Qwen3.8-27B` | Fits one or two GPUs | **Previously tested** | See `targets/qwen3.8-27b/PERFORMANCE.md` |
| Nex-N2.5-mini | `abenzerps/Nex-N2.5-mini-GGUF` `Q4_K_M` plus `mmproj` | Fits one 48 GiB GPU | **Passed** | 65,536 context, 2,074.40/2,969.05 tok/s prefill, 122.17 tok/s native decode, 99.48 tok/s API output; red/blue image tests passed |
| Qwen3.8-Flash-Next | `unsloth/Qwen3.8-Flash-Next-GGUF` | Fits one NVLink pair in Q3 | **Previously tested** | See `targets/qwen3.8-flash-next/PERFORMANCE.md` |
| GLM-5.3-Flash EXL3 2BPW | `0xSero/GLM-5.3-Flash-EXL3-2BPW` | Claimed 128 GiB class; EXL3 is unreliable on Turing | **Excluded: inaccessible runtime artifact** | Authenticated Hugging Face dry-run could not access the repository; the exact EXL3 quant was not tested |
| GLM-5.3-Flash EXL3 4BPW | `0xSero/GLM-5.3-Flash-EXL3-4BPW` | Advertised for 196-256 GiB | **Excluded: VRAM** | Exceeds 192 GiB aggregate VRAM and EXL3 is not a supported Turing path |
| DeepSeek-V4-Flash-Vision | `deepseek-ai/DeepSeek-V4-Flash-Vision` | Advertised for 196-256 GiB | **Excluded: VRAM / prior family test** | Exceeds aggregate VRAM; DeepSeek-V4-Flash text inference is already measured |
| GLM-5.3 EXL3 3BPW REAP | `0xSero/GLM-5.3-EXL3-3bpw-REAP` | Advertised for 196-256 GiB | **Excluded: VRAM** | Exceeds 192 GiB aggregate VRAM and EXL3 is not a supported Turing path |
| Nex-N2.5-Pro | `nex-agi/Nex-N2.5-Pro` | Advertised for 196-256 GiB; repository currently has no weights | **Excluded: unavailable / VRAM** | Hugging Face repository currently contains documentation only |
| GLM-5.3 | `zai-org/GLM-5.3` | Advertised for 384-512 GiB | **Excluded: VRAM** | Exceeds 192 GiB aggregate VRAM |
| MiniMax-M2.5 | `DevQuasar/MiniMaxAI.MiniMax-M2.5-GGUF` `Q2_K` | 83.3 GB; fits one 96 GiB NVLink pair | **Passed** | 32,768 context, 417.03/785.79 tok/s prefill, 63.46 tok/s native decode, 54.09 tok/s API output |

## Execution order

Only one model was loaded or benchmarked at a time:

1. Spark-X2.5-4B
2. Ternary Bonsai 27B
3. Gemma 4 12B
4. Nex-N2.5-mini
5. MiniMax-M2.5

All four candidates from the original eligible set and the MiniMax-M2.5
substitute completed successfully. Models marked previously tested,
unavailable, or over the 192 GiB aggregate VRAM limit were not run.
