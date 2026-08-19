#!/bin/bash
# Sofia profile for the generic cluster bootstrap (envs + Pi0.5 checkpoint + LIBERO dataset).

set -euo pipefail

export OPENPI_REPO_ROOT=${OPENPI_REPO_ROOT:-/sofia/projects/2026_start_025/openpi}
export OPENPI_CACHE_ROOT=${OPENPI_CACHE_ROOT:-/sofia/projects/2026_start_025/cache/openpi_low_data_cache}
export SLURM_ACCOUNT=${SOFIA_SLURM_ACCOUNT:-zen4-h200-2026_start_025-1}
export SLURM_PARTITION=${SOFIA_SLURM_PARTITION:-zen4_h200}
export SLURM_CPU_ACCOUNT=${SOFIA_SLURM_CPU_ACCOUNT:-vsc}
export SLURM_CPU_PARTITION=${SOFIA_SLURM_CPU_PARTITION:-zen5_vis}
export SLURM_GPU_REQUEST_MODE=gpus_per_node
export SLURM_GPUS_PER_NODE=1
export SLURM_CPUS_PER_GPU=24
export SLURM_GPU_MEMORY=auto
export SLURM_FINALIZE_MEMORY=auto
export OPENPI_GPU_MODULES=${SOFIA_GPU_MODULES:-CUDA/12.8.0}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export REQUIRE_HF_AUTH=${REQUIRE_HF_AUTH:-1}

exec "${OPENPI_REPO_ROOT}/scripts/bootstrap_low_data_cluster.sh"
