# Qwen3-Coder-Next 80B-A3B

Official model: `Qwen/Qwen3-Coder-Next`

- 80B total parameters
- 3B activated parameters
- 48 hybrid Gated DeltaNet / attention layers
- 512 experts, 10 routed experts per token
- 262,144-token native context
- non-thinking coding-agent model

The initial target is vLLM tensor parallelism across all four RTX 8000 GPUs.
The first profile limits context to 32K because FP16 weights consume most of
the 192 GiB aggregate VRAM. Backend and quantization selection will be revised
after direct SM75 compatibility and throughput measurements.

```bash
./scripts/download-model.sh qwen3-coder-next-80b-a3b
./scripts/serve-model.sh qwen3-coder-next-80b-a3b vllm-fp16-tp4
```

The official deployment requirements are vLLM 0.15.0 or newer or SGLang 0.5.8
or newer. This machine currently has a newer vLLM installation, but Turing
kernel compatibility must still be validated.
