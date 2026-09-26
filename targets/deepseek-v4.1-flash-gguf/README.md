# DeepSeek V4.1 Flash Q2 on DwarfStar

The active checkpoint is
[`antirez/deepseek-v4.1-flash-gguf`](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf),
`DeepSeek-V4.1-Flash-Q2.gguf` (`ds41f-q2`). Its 340.60 GiB file includes
151.77 GiB of main weights and 188.83 GiB of disk-read Engram tables.
The previous Q4 GGUF was removed; its measured results remain in
[PERFORMANCE.md](PERFORMANCE.md) as historical evidence, not Q2 performance.

Build and download with DwarfStar's own tools:

```bash
git clone https://github.com/antirez/ds4.git
cd ds4
make cuda CUDA_ARCH=sm_75
./download_model.sh ds41f-q2
DS4_CUDA_NO_SCORE_TILE=1 \
  LD_LIBRARY_PATH=/home/algore/miniconda3/lib:${LD_LIBRARY_PATH:-} \
  ./ds4-server --cuda --ssd-streaming \
    -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ctx 8192 \
    --host 127.0.0.1 --port 8091
```

The automatic SSD-streaming cache uses available memory for weights and
experts; Engram rows remain disk-backed. Do not assume the Q4 throughput
applies to Q2. The CUDA score-tile fallback was required on this host for
Q4; Q2 loaded and generated correct text with that same fallback.

**Measured on this host:** single-GPU Q2 native 512-/4096-token prefill
reached 8.73/39.93 tokens/s and 32-token decode reached 2.66/2.33
tokens/s respectively. Three no-thinking API requests with the same
27-token prompt each generated 40 tokens, taking 24.86-28.11 seconds
end to end. See [PERFORMANCE.md](PERFORMANCE.md) for the exact workload,
raw logs, and why Q2 and Q4 results cannot be treated as a same-precision
speedup.

Current DwarfStar V4.1 CUDA does not support in-process four-GPU tensor
parallelism. Its network tensor-parallel Q2 mode is designed for two
separate nodes, each with a full GGUF on local storage; it is not a
four-GPU single-host command. DSpark for V4 Flash 0731 does not apply to V4.1.
