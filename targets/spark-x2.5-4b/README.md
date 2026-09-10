# Spark-X2.5-4B

[`XHToken/Spark-X2.5-4B`](https://huggingface.co/XHToken/Spark-X2.5-4B)
is a 4.1B-parameter hybrid sliding-window/full-attention model with a native
1,048,576-token context limit. This target uses the official 4.4 GB
[`Q8_0` GGUF](https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF) for a
high-quality single-GPU test.

Spark-X2.5 support is supplied by the publisher's
[`XHToken/llama.cpp`](https://github.com/XHToken/llama.cpp) fork. The initial
profile uses a practical 65,536-token context; the architecture's advertised
maximum does not guarantee that a 1M-token KV cache fits in 48 GiB.

```bash
./scripts/download-model.sh spark-x2.5-4b
./scripts/benchmark-native.sh spark-x2.5-4b
./scripts/serve-model.sh spark-x2.5-4b
./scripts/benchmark-model.sh spark-x2.5-4b -- --max-tokens 512
```

See [PERFORMANCE.md](PERFORMANCE.md) for measured results.
