# Qwen3-Coder-Next performance

No validated RTX 8000 baseline has been recorded yet.

The first experiments will compare:

1. vLLM FP16 tensor parallelism across four GPUs.
2. A supported weight-only quantization if FP16 leaves insufficient KV-cache
   headroom or Turing-compatible kernels are available.
3. llama.cpp GGUF as a compatibility and single-user latency baseline.
4. SGLang if its Turing attention and MoE kernels outperform vLLM.
