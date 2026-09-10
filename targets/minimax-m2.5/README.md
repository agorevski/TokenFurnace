# MiniMax-M2.5

[`MiniMaxAI/MiniMax-M2.5`](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
is a frontier-oriented MoE model for coding, agentic tool use, and search. It
was selected as the closest practical substitute for the inaccessible
GLM-5.3-Flash EXL3 2BPW artifact: both target high-capability agent workloads,
while this 83.3 GB Q2_K GGUF has a Turing-compatible llama.cpp path.

The profile maps the model across the NVLink-connected GPU 0-1 pair and uses a
conservative 32,768-token context. GPUs 2-3 remain untouched for the existing
workload.

```bash
./scripts/download-model.sh minimax-m2.5
./scripts/benchmark-native.sh minimax-m2.5
./scripts/serve-model.sh minimax-m2.5
./scripts/benchmark-model.sh minimax-m2.5 -- --max-tokens 512
```

See [PERFORMANCE.md](PERFORMANCE.md) for measured results.
