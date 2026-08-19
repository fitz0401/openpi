#!/bin/bash
# Sofia profile for the generic cluster bootstrap (envs + Pi0.5 checkpoint + LIBERO dataset).

set -euo pipefail

export OPENPI_REPO_ROOT=${OPENPI_REPO_ROOT:-/sofia/projects/2026_start_025/openpi}
export OPENPI_CACHE_ROOT=${OPENPI_CACHE_ROOT:-/sofia/projects/2026_start_025/cache/openpi_low_data_cache}
export SLURM_ACCOUNT=${SLURM_ACCOUNT:-zen4-h200-2026_start_025-1}
export SLURM_PARTITION=${SLURM_PARTITION:-zen4_h200}
export SLURM_CPU_PARTITION=${SLURM_CPU_PARTITION:-zen4_h200}
export SLURM_GPU_GRES=${SLURM_GPU_GRES:-gpu:nvidia_h200:1}
export SLURM_CPUS_PER_GPU=${SLURM_CPUS_PER_GPU:-12}
export SLURM_GPU_MEMORY=${SLURM_GPU_MEMORY:-125G}
export OPENPI_GPU_MODULES=${OPENPI_GPU_MODULES:-CUDA/12.8.0}

exec "${OPENPI_REPO_ROOT}/scripts/bootstrap_low_data_cluster.sh"
