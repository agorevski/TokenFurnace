# GLM-5.3-Flash

Official model:
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash).
This target uses the
[`unsloth/GLM-5.3-Flash-GGUF`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)
`UD-Q3_K_XL` quant for Turing-compatible llama.cpp inference.

The quant is 147.54 GB across four shards. It requires all four 48 GiB RTX
8000 GPUs and therefore spans the workstation's two separate NVLink pairs.

GLM-5.3-Flash support is not yet in upstream llama.cpp. The target uses
Unsloth's `glm5next/upstream` branch from
[`unslothai/llama.cpp`](https://github.com/unslothai/llama.cpp/tree/glm5next/upstream),
which tracks the implementation proposed in
[llama.cpp PR 27754](https://github.com/ggml-org/llama.cpp/pull/27754).

```bash
./scripts/download-model.sh glm-5.3-flash
./scripts/benchmark-native.sh glm-5.3-flash
./scripts/serve-model.sh glm-5.3-flash
./scripts/benchmark-model.sh glm-5.3-flash -- --max-tokens 512
```

See [PERFORMANCE.md](PERFORMANCE.md) for measured results and runtime details.
