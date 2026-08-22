#!/bin/bash
# Shared site defaults for low-data bootstrap and native-Slurm submission.

set -euo pipefail

load_low_data_cluster_profile() {
  local cluster=${1:?Pass a cluster name (sofia or leonardo)}
  case "${cluster}" in
    sofia)
      export OPENPI_CLUSTER=sofia
      export OPENPI_REPO_ROOT=${OPENPI_REPO_ROOT:-/sofia/projects/2026_start_025/openpi}
      export OPENPI_CACHE_ROOT=${OPENPI_CACHE_ROOT:-/sofia/projects/2026_start_025/cache/openpi_low_data_cache}
      export SLURM_ACCOUNT=${SLURM_ACCOUNT:-zen4-h200-2026_start_025-1}
      export SLURM_PARTITION=${SLURM_PARTITION:-zen4_h200}
      export SLURM_CPU_ACCOUNT=${SLURM_CPU_ACCOUNT:-vsc}
      export SLURM_CPU_PARTITION=${SLURM_CPU_PARTITION:-zen5_vis}
      export SLURM_GPU_REQUEST_MODE=${SLURM_GPU_REQUEST_MODE:-gpus_per_node}
      export SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}
      export SLURM_CPUS_PER_GPU=${SLURM_CPUS_PER_GPU:-24}
      export SLURM_GPU_MEMORY=${SLURM_GPU_MEMORY:-auto}
      export SLURM_FINALIZE_MEMORY=${SLURM_FINALIZE_MEMORY:-auto}
      export OPENPI_GPU_MODULES=${OPENPI_GPU_MODULES:-CUDA/12.8.0}
      export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
      export REQUIRE_HF_AUTH=${REQUIRE_HF_AUTH:-1}
      ;;
    leonardo)
      export OPENPI_CLUSTER=leonardo
      export OPENPI_REPO_ROOT=${OPENPI_REPO_ROOT:-/leonardo_work/EUHPC_D35_005/ze/openpi}
      export OPENPI_STORAGE_ROOT=${OPENPI_STORAGE_ROOT:-/leonardo_scratch/fast/EUHPC_D35_005/ze}
      export OPENPI_CACHE_ROOT=${OPENPI_CACHE_ROOT:-${OPENPI_STORAGE_ROOT}/cache/openpi_low_data_cache}
      export OPENPI_JOB_TMP_ROOT=${OPENPI_JOB_TMP_ROOT:-${OPENPI_STORAGE_ROOT}/job_tmp}
      export SLURM_ACCOUNT=${SLURM_ACCOUNT:-euhpc_d35_005}
      export SLURM_PARTITION=${SLURM_PARTITION:-boost_usr_prod}
      export SLURM_CPU_ACCOUNT=${SLURM_CPU_ACCOUNT:-${SLURM_ACCOUNT}}
      export SLURM_CPU_PARTITION=${SLURM_CPU_PARTITION:-lrd_all_serial}
      export SLURM_QOS=${SLURM_QOS:-normal}
      export SLURM_GPU_REQUEST_MODE=${SLURM_GPU_REQUEST_MODE:-gres}
      export SLURM_GPU_GRES=${SLURM_GPU_GRES:-gpu:1}
      export SLURM_CPUS_PER_GPU=${SLURM_CPUS_PER_GPU:-8}
      export SLURM_STAGE_A_GPU_REQUEST_MODE=${SLURM_STAGE_A_GPU_REQUEST_MODE:-gres}
      export SLURM_STAGE_A_GPU_GRES=${SLURM_STAGE_A_GPU_GRES:-gpu:2}
      export SLURM_STAGE_A_CPUS_PER_TASK=${SLURM_STAGE_A_CPUS_PER_TASK:-16}
      export OPENPI_SOURCE_FSDP_DEVICES=${OPENPI_SOURCE_FSDP_DEVICES:-2}
      export SLURM_GPU_MEMORY=${SLURM_GPU_MEMORY:-auto}
      export SLURM_FINALIZE_MEMORY=${SLURM_FINALIZE_MEMORY:-24G}
      export SLURM_STAGE_A_TIME=${SLURM_STAGE_A_TIME:-24:00:00}
      export SLURM_STAGE_B_TIME=${SLURM_STAGE_B_TIME:-12:00:00}
      export OPENPI_BOOTSTRAP_MODULES=${OPENPI_BOOTSTRAP_MODULES:-python/3.11.7}
      export OPENPI_GPU_MODULES=${OPENPI_GPU_MODULES:-python/3.11.7 cuda/12.2}
      export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-0}
      export REQUIRE_HF_AUTH=${REQUIRE_HF_AUTH:-1}
      export OPENPI_INSTALL_UV=${OPENPI_INSTALL_UV:-1}
      # LeRobot pulls rerun-sdk only for visualization. Its locked Linux wheel
      # requires glibc 2.31, while Leonardo provides glibc 2.28.
      export OPENPI_SKIP_RERUN_SDK=${OPENPI_SKIP_RERUN_SDK:-1}
      ;;
    *)
      echo "Unknown cluster '${cluster}'. Expected sofia or leonardo." >&2
      return 2
      ;;
  esac
  export LIBERO_VENV=${LIBERO_VENV:-${OPENPI_REPO_ROOT}/examples/libero/.venv}
}
