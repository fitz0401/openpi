#!/bin/bash
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l walltime=12:00:00
#PBS -A starting_2026_047

set -euo pipefail

REPO_ROOT=/dodrio/scratch/projects/starting_2026_047/openpi
cd "${REPO_ROOT}"
module load cluster/dodrio/gpu_rome_a100_80_rhel9
module load CUDA
source .venv/bin/activate

export HF_HOME=/dodrio/scratch/projects/starting_2026_047/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export OPENPI_DATA_HOME=/dodrio/scratch/projects/starting_2026_047/cache/openpi

MAX_FRAMES=${MAX_FRAMES:-200000}
uv run scripts/compute_norm_stats.py \
  --config-name pi05_libero_low_data_full \
  --max-frames "${MAX_FRAMES}"
