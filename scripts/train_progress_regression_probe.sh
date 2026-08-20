#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-examples/progress_probe/configs/progress_regression_v3_all_source.json}"
GPU_INDEX="${GPU_INDEX:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
DEFAULT_LIBERO_DATA_ROOT="/mnt/data/ze/datasets/lerobot/physical-intelligence/libero"
export OPENPI_LIBERO_DATA_ROOT="${OPENPI_LIBERO_DATA_ROOT:-$DEFAULT_LIBERO_DATA_ROOT}"
export HF_DATASETS_CACHE="${OPENPI_HF_DATASETS_CACHE:-/mnt/data/ze/cache/huggingface/datasets}"

cd "$REPO_ROOT"
if [[ ! -f "$OPENPI_LIBERO_DATA_ROOT/meta/info.json" ]]; then
  echo "error: LIBERO LeRobot dataset not found at $OPENPI_LIBERO_DATA_ROOT" >&2
  echo "Set OPENPI_LIBERO_DATA_ROOT to the dataset root if it is stored elsewhere." >&2
  exit 1
fi
echo "LIBERO dataset: $OPENPI_LIBERO_DATA_ROOT"
mkdir -p "$HF_DATASETS_CACHE"
echo "HF datasets cache: $HF_DATASETS_CACHE"
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
