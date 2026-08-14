#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-examples/progress_probe/configs/progress_regression_v3_all_source.json}"
GPU_INDEX="${GPU_INDEX:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"

cd "$REPO_ROOT"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi is unavailable" >&2
  exit 1
fi
GPU_INFO="$(nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv,noheader,nounits -i "$GPU_INDEX")"
IFS=',' read -r GPU_NAME GPU_FREE GPU_UTIL <<< "$GPU_INFO"
GPU_FREE="${GPU_FREE//[[:space:]]/}"
echo "GPU $GPU_INDEX:${GPU_NAME}; free=${GPU_FREE} MiB; utilization=${GPU_UTIL}"
if (( GPU_FREE < MIN_FREE_MIB )); then
  echo "error: GPU $GPU_INDEX has less than ${MIN_FREE_MIB} MiB free" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
uv run python -m openpi.training.progress_probe.train_regression --config "$CONFIG" --device cuda
