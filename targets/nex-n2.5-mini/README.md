# Nex-N2.5-mini

[`nex-agi/Nex-N2.5-mini`](https://huggingface.co/nex-agi/Nex-N2.5-mini) is a
35B-A3B Qwen3.5 MoE post-trained for agentic, coding, and multimodal work. This
target uses abenzerps' 21.2 GB
[`Q4_K_M` GGUF and F16 projector](https://huggingface.co/abenzerps/Nex-N2.5-mini-GGUF).

Both the text model and projector fit on one 48 GiB RTX 8000. The baseline
uses a practical 65,536-token context rather than assuming the full
262,144-token training context will fit with all runtime buffers.

```bash
./scripts/download-model.sh nex-n2.5-mini
./scripts/benchmark-native.sh nex-n2.5-mini
./scripts/serve-model.sh nex-n2.5-mini
./scripts/benchmark-model.sh nex-n2.5-mini -- --max-tokens 512
```

See [PERFORMANCE.md](PERFORMANCE.md) for measured results.
