# Ternary Bonsai 2 27B

[`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
is a ternary quantization of the dense Qwen3.8-27B architecture. It requires
the publisher's Hadamard-aware
[`PrismML-Eng/llama.cpp`](https://github.com/PrismML-Eng/llama.cpp) fork;
stock llama.cpp cannot run these rotated weights correctly.

The text-only profiles compare the 5.95 GB `PTQ1_0` dense-trit pack and 7.21 GB
`PQ2_0` two-bit-slot pack on one RTX 8000. The publisher reports that
`PTQ1_0` favors memory-bandwidth-bound decode on Ada and L4, but Turing is not
in the published matrix. Local measurements make `llama-cpp-pq2-1gpu` the
default: PQ2_0 was 18.3% faster for 128-token decode and 65.7-70.6% faster for
prompt processing despite its 21.2% larger artifact.
The optional vision projector is excluded so language-model throughput is
measured independently.

```bash
./scripts/setup-ternary-bonsai-2-27b.sh
./scripts/serve-model.sh ternary-bonsai-2-27b
```

The `llama-cpp-turboquant-1gpu` profile uses the optimized
[`agorevski/bonsai-squared`](https://github.com/agorevski/bonsai-squared)
runtime with `KTQ2_1` keys and `VTQ2_2` values. It runs on GPU 0 with a
65,536-token context and exposes the OpenAI model name
`ternary-bonsai-2-27b-turboquant` on port 8101. Reasoning is enabled with
`medium` effort by default:

```bash
./scripts/serve-model.sh ternary-bonsai-2-27b llama-cpp-turboquant-1gpu
```

Example OpenAI-compatible request:

```bash
curl http://127.0.0.1:8101/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ternary-bonsai-2-27b-turboquant",
    "messages": [{"role": "user", "content": "Reply with exactly: READY"}],
    "temperature": 0
  }'
```

The setup script installs the pinned PrismML CUDA 12.4 runtime required for
the rotated weights and downloads the measured PQ2_0 default. To install the
smaller PTQ1_0 pack instead:

```bash
./scripts/setup-ternary-bonsai-2-27b.sh llama-cpp-ptq1-1gpu
```

Benchmarking commands:

```bash
./scripts/download-model.sh ternary-bonsai-2-27b
./scripts/benchmark-native.sh ternary-bonsai-2-27b
./scripts/download-model.sh ternary-bonsai-2-27b llama-cpp-ptq1-1gpu
./scripts/benchmark-native.sh ternary-bonsai-2-27b llama-cpp-ptq1-1gpu
./scripts/serve-model.sh ternary-bonsai-2-27b
./scripts/benchmark-model.sh ternary-bonsai-2-27b -- --max-tokens 256
```

See [PERFORMANCE.md](PERFORMANCE.md) for measured results and runtime
provenance.
