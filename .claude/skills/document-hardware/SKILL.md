---
name: document-hardware
description: Refresh this repository's machine-specific hardware, firmware, GPU topology, CUDA/toolchain, runtime, and compatibility documentation from verified host observations.
---

# Document the RTX 8000 host

Use this skill when adding, auditing, or refreshing facts about the workstation
represented by `hardware/4x-rtx8000`.

## Respect the sources of truth

- `hardware/4x-rtx8000/profile.env` is the machine-readable host snapshot and is
  sourced by `scripts/lib/common.sh`. Keep every non-comment line a valid shell
  assignment. Preserve the operational keys in its first block unless all
  consumers are updated deliberately.
- `hardware/4x-rtx8000/README.md` is the human-readable interpretation of that
  snapshot: topology implications, runtime support status, compatibility
  constraints, and references. Values repeated there must agree with
  `profile.env`.
- `targets/*/target.env`, `targets/*/profiles/*.env`, and
  `targets/*/PERFORMANCE.md` own model-specific settings and measured results.
  Do not promote a target result into the hardware profile as a general host
  capability.
- Vendor/runtime documentation establishes formal support. Commands run on this
  host establish installed or observed state. Clearly distinguish those from
  recommendations and unvalidated hypotheses.

Do not treat marketing specifications as proof of the current machine state,
or a framework's broad CUDA support as proof that a model-specific kernel works
on Turing `sm_75`.

## Gather a coherent snapshot

Run read-only checks on the actual host. Record the date as `YYYY-MM-DD`, note
commands that were unavailable, and never infer a missing value. Leave an
unknown `profile.env` value empty or omit a new field rather than fabricating
it.

### GPU, firmware, PCIe, and topology

```bash
nvidia-smi -L
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,power.limit,persistence_mode,ecc.mode.current,vbios_version,clocks.max.sm,clocks.max.memory --format=csv
nvidia-smi -q
nvidia-smi topo -m
nvidia-smi nvlink --status
lspci -Dnn | grep -iE 'vga|3d|nvidia'
```

Use `nvidia-smi topo -m` and NVLink status together. For this repository,
explicitly verify whether GPUs 0-1 and 2-3 remain the active NVLink pairs and
whether cross-pair traffic still uses PCIe/host bridges. Do not describe the
four GPUs as one NVLink fabric. Check current and maximum PCIe generation/link
width in `nvidia-smi -q`; idle current link speed is not the maximum capability.

### CPU, memory, NUMA, OS, kernel, and storage

```bash
lscpu
lscpu -e=CPU,NODE,SOCKET,CORE
numactl --hardware
free -h
cat /etc/os-release
uname -r
uname -m
findmnt -T /home/algore/models
df -h /home/algore/models
```

If `numactl`, `lspci`, or another utility is absent, document that the check was
unavailable; do not install packages merely to refresh documentation.

### Driver, CUDA, compiler, and build toolchain

```bash
nvidia-smi
nvcc --version
gcc --version
cmake --version
ldconfig -p | grep -E 'libcuda|libcudart|libnccl'
```

Keep these concepts separate:

- the NVIDIA driver version;
- the maximum CUDA compatibility reported by the driver;
- the system CUDA toolkit and `nvcc` version;
- CUDA/NCCL libraries bundled in Python environments or native builds.

Different compatible minor versions are not automatically an inconsistency.
Record GSP firmware and VBIOS only when directly reported by the host.

### Installed inference runtimes

Inspect each named environment independently rather than relying on whichever
Python happens to be active:

```bash
/home/algore/miniconda3/bin/conda env list
/home/algore/miniconda3/bin/conda run -n base python -c 'import platform, torch; print(platform.python_version()); print(torch.__version__); print(torch.version.cuda)'
/home/algore/miniconda3/bin/conda run -n base python -c 'import vllm; print(vllm.__version__)'
/home/algore/miniconda3/bin/conda run -n vllm python -c 'import platform, torch, vllm; print(platform.python_version()); print(torch.__version__); print(torch.version.cuda); print(vllm.__version__)'
/home/algore/miniconda3/bin/conda run -n trt-llm python -c 'import platform, torch; print(platform.python_version()); print(torch.__version__); print(torch.version.cuda)'
/home/algore/miniconda3/bin/conda run -n trt-llm python -c 'import tensorrt_llm; print(tensorrt_llm.__version__)'
git -C /home/algore/llama.cpp-dspark-current rev-parse HEAD
git -C /home/algore/llama.cpp-dspark-current describe --tags --always --dirty
```

An import failure means “not installed” or “not importable in this
environment,” not “unsupported.” Keep runtime paths, versions, build commit,
and build tag in `profile.env`; put explanatory status and repository usage in
the hardware README.

## Document compatibility conservatively

For each runtime or feature, label the claim as one of:

- **formally supported**, with an authoritative vendor/runtime reference;
- **observed working**, with the concrete local load, generation, build, or
  benchmark that exercised it;
- **not installed**;
- **unvalidated**;
- **not supported**, with the exact architecture/version constraint.

Re-check claims affected by the Turing `sm_75` architecture, including BF16,
TF32, FP8/FP4, FlashAttention, fused attention or MoE kernels, quantization
kernels, TensorRT-LLM, vLLM, SGLang, and ExLlama. A successful package import
does not validate model loading or generation. Preserve known topology-aware
guidance unless new measurements supersede it.

## Update the documentation

1. Review `git diff` first so another contributor's edits are not overwritten.
2. Refresh `profile.env` as one coherent snapshot. Update
   `PROFILE_SNAPSHOT_DATE` and `PROFILE_SNAPSHOT_SOURCE` to match the checks
   actually used. Keep unknown values empty and quote values containing spaces
   or shell metacharacters.
3. Refresh the matching tables and prose in
   `hardware/4x-rtx8000/README.md`. Give time-varying observations an explicit
   snapshot date. Stable product specifications may cite vendor references.
4. Update runtime-support prose only after checking current authoritative
   documentation or a local validation. Retain old measured conclusions when
   no replacement measurement exists, with their original date/context.
5. Do not edit target profiles or performance files unless the task explicitly
   includes model-specific changes.

Avoid partial refreshes that combine current driver/runtime values with an
unmarked older topology or firmware snapshot.

## Validate before finishing

```bash
bash -n hardware/4x-rtx8000/profile.env
env -i bash --noprofile --norc -c 'source hardware/4x-rtx8000/profile.env; test -n "$GPU_COUNT"; test -n "$CUDA_ARCH"; test -n "$PROFILE_SNAPSHOT_DATE"'
awk -F= '/^[A-Z][A-Z0-9_]*=/{if (++seen[$1] > 1) {print "duplicate key: " $1; bad=1}} END{exit bad}' hardware/4x-rtx8000/profile.env
git diff --check
git --no-pager diff -- hardware/4x-rtx8000
```

On the documented host, also compare `GPU_COUNT` with `nvidia-smi -L`, bus IDs
with `nvidia-smi topo -m`, and runtime versions with the explicit-environment
checks above. Resolve discrepancies or describe them with dates; never silence
them by choosing the expected value.
