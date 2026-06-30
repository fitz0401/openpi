#!/bin/bash
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l walltime=12:00:00
#PBS -A starting_2026_047

# Compute the SHARED normalization stats for the continual LIBERO-Object benchmark.
# Computed once over the LIBERO data and reused across every budget/seed/task (avoids unstable
# few-shot statistics). Writes to assets/pi05_libero_object_continual/libero_object.

set -euo pipefail

cd /dodrio/scratch/projects/starting_2026_047/openpi

module load cluster/dodrio/gpu_rome_a100_80_rhel9
module load CUDA
source .venv/bin/activate

export HF_HOME=/dodrio/scratch/projects/starting_2026_047/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_DOWNLOAD_THREADS=2
# Redirect the openpi checkpoint download cache (gs://openpi-assets/...) to project scratch;
# it defaults to ~/.cache/openpi, which sits on the quota-limited user home.
export OPENPI_DATA_HOME=/dodrio/scratch/projects/starting_2026_047/cache/openpi

CONFIG=${CONFIG:-pi05_libero_object_continual}
MAX_FRAMES=${MAX_FRAMES:-200000}

nvidia-smi
echo "CONFIG=${CONFIG}  MAX_FRAMES=${MAX_FRAMES}"
uv run scripts/compute_norm_stats.py --config-name "${CONFIG}" --max-frames "${MAX_FRAMES}"
