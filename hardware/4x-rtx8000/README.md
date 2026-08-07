# Four Quadro RTX 8000 workstation

This profile documents the hardware and software constraints that apply to
every model target in this repository.

## Installed hardware

| Component | Configuration |
|---|---|
| GPUs | 4x NVIDIA Quadro RTX 8000 |
| Architecture | Turing TU102 |
| CUDA compute capability | 7.5 (`sm_75`) |
| VRAM | 48 GiB GDDR6 per GPU, 192 GiB aggregate |
| Memory bandwidth | 672 GB/s per GPU |
| Tensor cores | 576 per GPU |
| CPU | Intel Xeon W-2295, 18 cores / 36 threads |
| System RAM | 256 GiB |
| PCIe | PCIe 3.0 x16 cards |
| Topology | GPUs 0-1 and 2-3 are separate NVLink pairs |

NVLink is not one four-GPU fabric. Traffic within each pair has two active
links reported at 25.781 GB/s each; traffic between pairs crosses PCIe host
bridges. Tensor-parallel profiles must therefore account for a slower
cross-pair collective path.

## Firmware and driver snapshot

Recorded on 2026-08-06:

| Item | Value |
|---|---|
| NVIDIA driver | 580.173.02 |
| Driver-reported CUDA compatibility | CUDA 13.0 |
| Installed CUDA toolkit used by llama.cpp | CUDA 12.4 |
| VBIOS, all four GPUs | 90.02.4A.00.01 |
| NVIDIA GSP firmware | `gsp_tu10x.bin` from driver 580.173.02 |
| Maximum board power | 260 W per GPU |
| Persistence mode | Disabled |
| ECC mode | Disabled |
| Maximum SM clock reported | 2100 MHz |
| Maximum memory clock reported | 7001 MHz |

The driver is new enough for the installed CUDA 12.4, CUDA 12.8, and CUDA 13.0
runtime builds. Package CUDA runtimes and the system `nvcc` toolkit do not need
to have identical minor versions.

## Native numerical support

Turing tensor cores accelerate FP16 and integer inference, but these GPUs do
not provide native BF16, TF32, FP8, or FP4 tensor-core execution. Consequences:

- use FP16 rather than BF16 for unquantized models;
- FP8 and NVFP4 profiles intended for Hopper or Blackwell are not appropriate;
- INT8/INT4 weight formats can fit larger models, but speed depends on whether
  the selected runtime ships an `sm_75` kernel for that quantization method;
- newer fused attention and MoE kernels may require Ampere (`sm_80`) or later.

## Runtime support matrix

| Runtime | Formal or observed status on `sm_75` | Repository use |
|---|---|---|
| NVIDIA CUDA | Fully supported by the installed driver | All GPU backends |
| NCCL | Working; used by the llama.cpp build | Multi-GPU collectives |
| llama.cpp | Working with CUDA, NCCL, Flash Attention, MMQ, and GGUF | DeepSeek baseline and compatibility fallback |
| vLLM | Formally supports NVIDIA compute capability 7.0+ | Preferred first Qwen server, subject to model-kernel validation |
| FlashAttention 2 | Requires Ampere or newer | Not available |
| xFormers | Traditional vLLM fallback for pre-Ampere GPUs | Not installed in the current vLLM environment |
| vLLM Triton attention | Available in current vLLM builds | Candidate Turing attention backend |
| SGLang | Framework supports Qwen3-Coder-Next; local `sm_75` path unvalidated | Candidate after vLLM |
| TensorRT-LLM | Current environment does not contain TensorRT-LLM | Not selected until architecture and Turing kernels are verified |
| Transformers | Supports the model architecture in the installed versions | Correctness fallback, not expected to be fastest |

Framework-level GPU support does not guarantee that every model-specific MoE,
Gated DeltaNet, attention, or quantization kernel supports Turing. Each target
must pass a real load and generation test.

## Installed inference environments

| Environment | Python | PyTorch | CUDA runtime | Framework |
|---|---:|---:|---:|---|
| Conda `base` | 3.13.9 | 2.11.0 | 13.0 | vLLM 0.21.0 |
| Conda `vllm` | 3.11.15 | 2.10.0 | 12.8 | vLLM 0.17.1 |
| Conda `trt-llm` | 3.12.13 | 2.5.1 | 12.4 | TensorRT-LLM not installed |

The official Qwen3-Coder-Next minimum is vLLM 0.15.0 or SGLang 0.5.8. Both
installed vLLM versions satisfy the version requirement. The dedicated
`vllm` environment is the conservative default because Python 3.11 has broader
CUDA extension compatibility than the base Python 3.13 environment.

## Backend selection guidance

1. Try current vLLM FP16 tensor parallelism first for Qwen3-Coder-Next.
2. If a model-specific fused kernel rejects `sm_75`, test the supported fallback
   implementation before changing frameworks.
3. Use a verified Turing-compatible weight-only quantization only if it improves
   measured throughput, not merely VRAM usage.
4. Keep llama.cpp/GGUF as the broadest compatibility path and single-user
   latency baseline.
5. Do not assume FP8 or Blackwell/Hopper benchmark results transfer to Turing.

## Operational checks

```bash
nvidia-smi
nvidia-smi topo -m
nvidia-smi nvlink --status
./scripts/status.sh TARGET PROFILE
```

## References

- [NVIDIA Quadro RTX 8000 product specifications](https://www.nvidia.com/en-us/design-visualization/quadro/rtx-8000/)
- [NVIDIA CUDA GPU compute capability list](https://developer.nvidia.com/cuda-gpus)
- [vLLM GPU installation requirements](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [vLLM quantization hardware support](https://docs.vllm.ai/en/stable/features/quantization/supported_hardware/)
- [Qwen3-Coder-Next model card](https://huggingface.co/Qwen/Qwen3-Coder-Next)
