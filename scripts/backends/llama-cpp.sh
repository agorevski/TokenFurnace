#!/usr/bin/env bash

set -euo pipefail

server=$BUILD_DIR/bin/llama-server
require_file "$server"
require_file "$MODEL"
add_runtime_libraries

# Defensive consistency check: the number of --tensor-split fractions must match
# the number of visible CUDA devices, otherwise llama.cpp silently splits across
# the wrong device count (e.g. a 2-GPU profile that forgets to shrink a 4-entry
# split, or vice-versa). The effective split is the profile's TENSOR_SPLIT or the
# backend default (1,1,1,1). Counting is done entirely with bash builtins
# (`read -ra` on a comma IFS), so no external command is required and it works
# for the repo's numeric CUDA IDs (0,1,2,3) as well as any other comma-separated
# device list. The check is scoped to run only when CUDA_VISIBLE_DEVICES is set.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  _effective_ts=${TENSOR_SPLIT:-1,1,1,1}
  IFS=',' read -ra _ts_parts <<<"$_effective_ts"
  IFS=',' read -ra _dev_parts <<<"$CUDA_VISIBLE_DEVICES"
  if ((${#_ts_parts[@]} != ${#_dev_parts[@]})); then
    die "TENSOR_SPLIT has ${#_ts_parts[@]} entries but CUDA_VISIBLE_DEVICES lists ${#_dev_parts[@]} device(s); they must match (TENSOR_SPLIT='${_effective_ts}', CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}')"
  fi
fi

# DeepSeek's measured default is 4x. A profile may explicitly assign an empty
# value to restore CUDA's stock launch-queue behavior without changing other
# targets.
if [[ -n "${CUDA_SCALE_LAUNCH_QUEUES+x}" ]]; then
  if [[ -n "$CUDA_SCALE_LAUNCH_QUEUES" ]]; then
    export CUDA_SCALE_LAUNCH_QUEUES
  else
    unset CUDA_SCALE_LAUNCH_QUEUES
  fi
else
  export CUDA_SCALE_LAUNCH_QUEUES=4x
fi

args=(
  --model "$MODEL"
  --alias "$MODEL_ALIAS"
  --ctx-size "$CTX_SIZE"
  --parallel "$PARALLEL"
  --batch-size "$BATCH_SIZE"
  --ubatch-size "$UBATCH_SIZE"
  --n-gpu-layers 99
  --split-mode "${SPLIT_MODE:-layer}"
  --tensor-split "${TENSOR_SPLIT:-1,1,1,1}"
  --fit on
  --fit-target 2048
  --flash-attn "${FLASH_ATTN:-on}"
  --jinja
  --temp "${TEMPERATURE:-1.0}"
  --top-p "${TOP_P:-0.95}"
  --min-p "${MIN_P:-0.01}"
  --slot-save-path "$MODEL_DIR/slot-cache"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
)

if [[ -n "${TOP_K:-}" ]]; then
  args+=(--top-k "$TOP_K")
fi
if [[ -n "${CHAT_TEMPLATE_KWARGS:-}" ]]; then
  args+=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
fi

if [[ -n "${CACHE_TYPE_K:-}" ]]; then
  args+=(--cache-type-k "$CACHE_TYPE_K")
fi
if [[ -n "${CACHE_TYPE_V:-}" ]]; then
  args+=(--cache-type-v "$CACHE_TYPE_V")
fi

# Explicit prompt/prefix caching knobs (b10298). llama-server already defaults
# to --cache-prompt on, --cache-ram 8192 MiB, and --cache-idle-slots on, so
# these are only appended when a profile opts in, keeping existing targets
# (e.g. DeepSeek) on the stock defaults. Prefix matching is exact: any change to
# the system prompt, tool schemas, timestamps, or message ordering invalidates
# the cached prefix.
if [[ "${CACHE_PROMPT:-1}" == 0 ]]; then
  args+=(--no-cache-prompt)
fi
if [[ -n "${CACHE_RAM_MIB:-}" ]]; then
  args+=(--cache-ram "$CACHE_RAM_MIB")
fi
if [[ "${CACHE_IDLE_SLOTS:-1}" == 0 ]]; then
  args+=(--no-cache-idle-slots)
fi
if [[ -n "${CACHE_REUSE:-}" ]]; then
  args+=(--cache-reuse "$CACHE_REUSE")
fi
if [[ -n "${SLOT_PROMPT_SIMILARITY:-}" ]]; then
  args+=(--slot-prompt-similarity "$SLOT_PROMPT_SIMILARITY")
fi
if [[ "${ENABLE_METRICS:-0}" != 0 ]]; then
  args+=(--metrics)
fi

# Optional generic runtime knobs. All are unset by default so existing targets
# (DeepSeek, the Qwen 4-GPU profile) keep the server's stock behavior; a profile
# opts in only when a measured win justifies it. The 2-GPU decode tuning
# (2026-08-07) found THREADS/POLL differences to be within thermal noise for this
# GPU-bandwidth-bound decode, so no profile sets them today — they are exposed
# here for documented, reproducible experimentation.
#   THREADS    -> --threads     CPU threads for generation (default: -1 = auto)
#   POLL       -> --poll         busy-poll level 0..100 (default: 50)
#   PRIORITY   -> --prio         process/thread priority -1..2 (default: 0)
#   LOAD_MODE  -> --load-mode    model load mode: none|mmap|isolate (default: mmap)
#   OP_OFFLOAD -> --op-offload / --no-op-offload  host-op offload (default: on)
if [[ -n "${THREADS:-}" ]]; then
  args+=(--threads "$THREADS")
fi
if [[ -n "${POLL:-}" ]]; then
  args+=(--poll "$POLL")
fi
if [[ -n "${PRIORITY:-}" ]]; then
  args+=(--prio "$PRIORITY")
fi
if [[ -n "${LOAD_MODE:-}" ]]; then
  args+=(--load-mode "$LOAD_MODE")
fi
if [[ -n "${OP_OFFLOAD:-}" ]]; then
  if [[ "$OP_OFFLOAD" == 0 ]]; then
    args+=(--no-op-offload)
  else
    args+=(--op-offload)
  fi
fi

if [[ "${DSPARK:-0}" != 0 ]]; then
  require_file "$DRAFT_MODEL"
  args+=(
    --spec-draft-model "$DRAFT_MODEL"
    --spec-type draft-dspark
    --spec-draft-n-max "$SPEC_DRAFT_N_MAX"
    --spec-draft-p-min "$SPEC_DRAFT_P_MIN"
    --spec-draft-ngl 99
  )
elif [[ -n "${DRAFT_MODEL:-}" ]]; then
  require_file "$DRAFT_MODEL"
  args+=(--spec-draft-model "$DRAFT_MODEL")
  if [[ -n "${SPEC_DRAFT_NGL:-}" ]]; then
    args+=(--spec-draft-ngl "$SPEC_DRAFT_NGL")
  fi
  if [[ -n "${SPEC_DRAFT_OVERRIDE_TENSOR:-}" ]]; then
    args+=(--spec-draft-override-tensor "$SPEC_DRAFT_OVERRIDE_TENSOR")
  fi
fi

if [[ -n "${SPEC_TYPE:-}" ]]; then
  args+=(--spec-type "$SPEC_TYPE")
  if [[ -n "${SPEC_DRAFT_N_MAX:-}" ]]; then
    args+=(--spec-draft-n-max "$SPEC_DRAFT_N_MAX")
  fi
  if [[ -n "${SPEC_DRAFT_P_MIN:-}" ]]; then
    args+=(--spec-draft-p-min "$SPEC_DRAFT_P_MIN")
  fi
fi

mkdir -p "$MODEL_DIR/slot-cache"
exec "$server" "${args[@]}" "$@"
