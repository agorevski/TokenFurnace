#!/usr/bin/env bash

set -euo pipefail

require_command docker
[[ -d "$MODEL" ]] || die "model directory not found: $MODEL"
[[ -n "${DOCKER_IMAGE:-}" ]] || die "DOCKER_IMAGE is required"

container_model=${DOCKER_MODEL_PATH:-/model}
container_name=${DOCKER_CONTAINER_NAME:-tokenfurnace-$TARGET-$PROFILE}

if docker container inspect "$container_name" >/dev/null 2>&1; then
  container_status=$(docker container inspect "$container_name" --format '{{.State.Status}}')
  if [[ "$container_status" == running ]] && curl -sf --max-time 3 "$(health_url)" >/dev/null; then
    printf 'Container %s is already running and healthy at %s\n' \
      "$container_name" "$(health_url)"
    exit 0
  fi
  die "container $container_name already exists with status=$container_status; remove it with: docker rm -f $container_name"
fi

if [[ -n "${DOCKER_CACHE_DIR:-}" ]]; then
  mkdir -p "$DOCKER_CACHE_DIR"
fi

docker_args=(
  run --rm
  --name "$container_name"
  --network host
  --ipc host
  --mount "type=bind,src=$MODEL,dst=$container_model,readonly"
)

if [[ -n "${DOCKER_CACHE_DIR:-}" ]]; then
  docker_args+=(--mount "type=bind,src=$DOCKER_CACHE_DIR,dst=/root/.cache/vllm")
fi

if [[ "${DOCKER_GPU_MODE:-runtime}" == direct ]]; then
  IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
  docker_args+=(
    --device /dev/nvidiactl
    --device /dev/nvidia-uvm
    --device /dev/nvidia-uvm-tools
  )
  for device in "${devices[@]}"; do
    docker_args+=(--device "/dev/nvidia$device")
  done

  for library in libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1; do
    host_library=$(readlink -f "/usr/lib/x86_64-linux-gnu/$library")
    require_file "$host_library"
    docker_args+=(--mount "type=bind,src=$host_library,dst=/usr/lib/x86_64-linux-gnu/$library,readonly")
  done
  docker_args+=(--env "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64")
else
  docker_args+=(--runtime nvidia --gpus "device=$CUDA_VISIBLE_DEVICES")
fi

for name in \
  VLLM_USE_FLASHINFER_SAMPLER \
  ATTENTION_BACKEND \
  CC \
  CXX \
  FLASHINFER_EXTRA_LDFLAGS
do
  if [[ -n "${!name:-}" ]]; then
    docker_args+=(--env "$name=${!name}")
  fi
done

args=(
  "$container_model"
  --served-model-name "$MODEL_ALIAS"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
  --dtype "$DTYPE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)

if [[ "${PIPELINE_PARALLEL_SIZE:-1}" != 1 ]]; then
  args+=(--pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE")
fi
if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
  args+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi
if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
  args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi
if [[ "${ENABLE_PREFIX_CACHING:-}" == 0 ]]; then
  args+=(--no-enable-prefix-caching)
elif [[ -n "${ENABLE_PREFIX_CACHING:-}" ]]; then
  args+=(--enable-prefix-caching)
  if [[ -n "${MAMBA_CACHE_MODE:-}" ]]; then
    args+=(--mamba-cache-mode "$MAMBA_CACHE_MODE")
  fi
  if [[ -n "${PREFIX_CACHING_HASH_ALGO:-}" ]]; then
    args+=(--prefix-caching-hash-algo "$PREFIX_CACHING_HASH_ALGO")
  fi
fi
if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
  args+=(--attention-backend "$ATTENTION_BACKEND")
fi
if [[ "${ENABLE_AUTO_TOOL_CHOICE:-0}" != 0 ]]; then
  args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ -n "${REASONING_PARSER:-}" ]]; then
  args+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ "${LANGUAGE_MODEL_ONLY:-0}" != 0 ]]; then
  args+=(--language-model-only)
fi
if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
  args+=(--speculative-config "$SPECULATIVE_CONFIG")
fi
if [[ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]]; then
  args+=(--default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS")
fi
if [[ "${ENFORCE_EAGER:-0}" != 0 ]]; then
  args+=(--enforce-eager)
fi

exec docker "${docker_args[@]}" "$DOCKER_IMAGE" "${args[@]}" "$@"
