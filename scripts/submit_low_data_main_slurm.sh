#!/bin/bash
# Submit any schema-v3 low-data config as Stage A -> Stage B array -> finalizer on native Slurm.

set -euo pipefail

REPO_ROOT=${OPENPI_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
EXPERIMENT_CONFIG=${1:-${REPO_ROOT}/examples/low_data/configs/libero_main_18source_22target.json}
EXPERIMENT_CONFIG=$(realpath "${EXPERIMENT_CONFIG}")
OPENPI_CACHE_ROOT=${OPENPI_CACHE_ROOT:-${SCRATCH:-${HOME}/.cache}/openpi_low_data_cache}
LIBERO_VENV=${LIBERO_VENV:-${REPO_ROOT}/examples/libero/.venv}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}
STAGE_B_ARRAY_INDICES=${STAGE_B_ARRAY_INDICES:-}

SLURM_PARTITION=${SLURM_PARTITION:?Set SLURM_PARTITION for GPU jobs}
SLURM_CPU_PARTITION=${SLURM_CPU_PARTITION:-${SLURM_PARTITION}}
SLURM_CPU_ACCOUNT=${SLURM_CPU_ACCOUNT:-${SLURM_ACCOUNT:-}}
SLURM_GPU_REQUEST_MODE=${SLURM_GPU_REQUEST_MODE:-gres}
SLURM_GPU_GRES=${SLURM_GPU_GRES:-gpu:1}
SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}
SLURM_CPUS_PER_GPU=${SLURM_CPUS_PER_GPU:-12}
SLURM_STAGE_A_GPU_REQUEST_MODE=${SLURM_STAGE_A_GPU_REQUEST_MODE:-${SLURM_GPU_REQUEST_MODE}}
SLURM_STAGE_A_GPU_GRES=${SLURM_STAGE_A_GPU_GRES:-${SLURM_GPU_GRES}}
SLURM_STAGE_A_GPUS_PER_NODE=${SLURM_STAGE_A_GPUS_PER_NODE:-${SLURM_GPUS_PER_NODE}}
SLURM_STAGE_A_CPUS_PER_TASK=${SLURM_STAGE_A_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU}}
SLURM_GPU_MEMORY=${SLURM_GPU_MEMORY:-125G}
SLURM_FINALIZE_MEMORY=${SLURM_FINALIZE_MEMORY:-16G}
SLURM_STAGE_A_TIME=${SLURM_STAGE_A_TIME:-48:00:00}
SLURM_STAGE_B_TIME=${SLURM_STAGE_B_TIME:-12:00:00}
SLURM_FINALIZE_TIME=${SLURM_FINALIZE_TIME:-02:00:00}

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Native Slurm submission requires sbatch." >&2
  exit 2
fi
if ! [[ "${MAX_CONCURRENT}" =~ ^[1-8]$ ]]; then
  echo "MAX_CONCURRENT must be between 1 and 8 (got ${MAX_CONCURRENT})." >&2
  exit 2
fi
if [ "${OPENPI_SKIP_MODULES:-0}" != 1 ] && [ -z "${OPENPI_GPU_MODULES:-}" ]; then
  echo "Set OPENPI_GPU_MODULES to the new cluster's space-separated module list, or OPENPI_SKIP_MODULES=1." >&2
  exit 2
fi

cd "${REPO_ROOT}"
mkdir -p job_record
export OPENPI_REPO_ROOT="${REPO_ROOT}"
export OPENPI_CACHE_ROOT
export LIBERO_VENV
export HF_HOME=${HF_HOME:-${OPENPI_CACHE_ROOT}/huggingface}
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${OPENPI_CACHE_ROOT}/openpi}

read -r GRID_SIZE NUM_SOURCE NUM_TARGET METHODS EVAL_PROTOCOL_ID RESULTS_DIR SPLIT_ID <<< "$(uv run python - "${EXPERIMENT_CONFIG}" <<'PY'
import pathlib
import sys

from openpi.training.low_data.experiment import load_experiment_config, target_grid

config = load_experiment_config(sys.argv[1])
print(
    len(target_grid(config)),
    len(config.source_task_refs()),
    len(config.target_task_refs()),
    ",".join(config.adaptation.methods),
    config.evaluation.protocol_id,
    pathlib.Path(config.results_root) / config.split_id,
    config.split_id,
)
PY
)"

if [ "${NUM_SOURCE}" -lt 1 ] || [ "${NUM_TARGET}" -lt 1 ] || [ "${METHODS}" != lora ] || \
   [ "${GRID_SIZE}" -lt 1 ] || \
   [ "${EVAL_PROTOCOL_ID}" != sog_target50_l10_target25_retention25_seed0_no_l10_retention ]; then
  echo "Refusing invalid formal protocol: split=${SPLIT_ID}, source=${NUM_SOURCE}, target=${NUM_TARGET}, " \
       "methods=${METHODS}, cells=${GRID_SIZE}, evaluation_protocol=${EVAL_PROTOCOL_ID}" >&2
  exit 2
fi

ACCOUNT_ARGS=()
if [ -n "${SLURM_ACCOUNT:-}" ]; then ACCOUNT_ARGS+=(--account="${SLURM_ACCOUNT}"); fi
QOS_ARGS=()
if [ -n "${SLURM_QOS:-}" ]; then QOS_ARGS+=(--qos="${SLURM_QOS}"); fi
CONSTRAINT_ARGS=()
if [ -n "${SLURM_CONSTRAINT:-}" ]; then CONSTRAINT_ARGS+=(--constraint="${SLURM_CONSTRAINT}"); fi
GPU_REQUEST_ARGS=()
case "${SLURM_GPU_REQUEST_MODE}" in
  gres)
    GPU_REQUEST_ARGS+=(--ntasks=1 --gres="${SLURM_GPU_GRES}")
    ;;
  gpus_per_node)
    GPU_REQUEST_ARGS+=(--gpus-per-node="${SLURM_GPUS_PER_NODE}" --ntasks-per-gpu=1)
    ;;
  *)
    echo "SLURM_GPU_REQUEST_MODE must be gres or gpus_per_node." >&2
    exit 2
    ;;
esac
STAGE_A_GPU_REQUEST_ARGS=()
case "${SLURM_STAGE_A_GPU_REQUEST_MODE}" in
  gres)
    STAGE_A_GPU_REQUEST_ARGS+=(--ntasks=1 --gres="${SLURM_STAGE_A_GPU_GRES}")
    ;;
  gpus_per_node)
    STAGE_A_GPU_REQUEST_ARGS+=(--gpus-per-node="${SLURM_STAGE_A_GPUS_PER_NODE}" --ntasks-per-gpu=1)
    ;;
  *)
    echo "SLURM_STAGE_A_GPU_REQUEST_MODE must be gres or gpus_per_node." >&2
    exit 2
    ;;
esac
GPU_MEMORY_ARGS=()
if [ -n "${SLURM_GPU_MEMORY}" ] && [ "${SLURM_GPU_MEMORY}" != auto ]; then
  GPU_MEMORY_ARGS+=(--mem="${SLURM_GPU_MEMORY}")
fi
FINALIZE_MEMORY_ARGS=()
if [ -n "${SLURM_FINALIZE_MEMORY}" ] && [ "${SLURM_FINALIZE_MEMORY}" != auto ]; then
  FINALIZE_MEMORY_ARGS+=(--mem="${SLURM_FINALIZE_MEMORY}")
fi
CPU_ACCOUNT_ARGS=()
if [ -n "${SLURM_CPU_ACCOUNT}" ]; then CPU_ACCOUNT_ARGS+=(--account="${SLURM_CPU_ACCOUNT}"); fi

GPU_COMMON=(
  --nodes=1
  --cpus-per-task="${SLURM_CPUS_PER_GPU}"
  --partition="${SLURM_PARTITION}"
  "${GPU_REQUEST_ARGS[@]}"
  "${GPU_MEMORY_ARGS[@]}"
  "${ACCOUNT_ARGS[@]}"
  "${QOS_ARGS[@]}"
  "${CONSTRAINT_ARGS[@]}"
)
STAGE_A_GPU_COMMON=(
  --nodes=1
  --cpus-per-task="${SLURM_STAGE_A_CPUS_PER_TASK}"
  --partition="${SLURM_PARTITION}"
  "${STAGE_A_GPU_REQUEST_ARGS[@]}"
  "${GPU_MEMORY_ARGS[@]}"
  "${ACCOUNT_ARGS[@]}"
  "${QOS_ARGS[@]}"
  "${CONSTRAINT_ARGS[@]}"
)

if [ -n "${STAGE_B_ARRAY_INDICES}" ]; then
  if ! [[ "${STAGE_B_ARRAY_INDICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "STAGE_B_ARRAY_INDICES must be a comma-separated list of non-negative integers." >&2
    exit 2
  fi
  IFS=',' read -r -a SELECTED_STAGE_B_INDICES <<< "${STAGE_B_ARRAY_INDICES}"
  for index in "${SELECTED_STAGE_B_INDICES[@]}"; do
    if [ "${index}" -ge "${GRID_SIZE}" ]; then
      echo "Stage-B array index ${index} is outside the grid [0, $((GRID_SIZE - 1))]." >&2
      exit 2
    fi
  done
  ARRAY_RANGE="${STAGE_B_ARRAY_INDICES}%${MAX_CONCURRENT}"
  SUBMITTED_STAGE_B_CELLS=${#SELECTED_STAGE_B_INDICES[@]}
else
  ARRAY_RANGE="0-$((GRID_SIZE - 1))%${MAX_CONCURRENT}"
  SUBMITTED_STAGE_B_CELLS=${GRID_SIZE}
fi

echo "Submitting ${SPLIT_ID}: ${NUM_SOURCE} source tasks -> ${NUM_TARGET} targets / ${GRID_SIZE} Stage-B cells."
echo "Evaluation=${EVAL_PROTOCOL_ID}; concurrency=${MAX_CONCURRENT}."
echo "GPU partition=${SLURM_PARTITION} request_mode=${SLURM_GPU_REQUEST_MODE} CPUs/GPU=${SLURM_CPUS_PER_GPU}."
echo "Stage A request_mode=${SLURM_STAGE_A_GPU_REQUEST_MODE} GPU=${SLURM_STAGE_A_GPU_GRES:-${SLURM_STAGE_A_GPUS_PER_NODE}} CPUs=${SLURM_STAGE_A_CPUS_PER_TASK} FSDP=${OPENPI_SOURCE_FSDP_DEVICES:-1}."
echo "Stage-B submission contains ${SUBMITTED_STAGE_B_CELLS} cells: ${ARRAY_RANGE}."

STAGE_A_JOB_ID=$(sbatch --parsable \
  "${STAGE_A_GPU_COMMON[@]}" \
  --job-name="ld_A_${SPLIT_ID}" \
  --time="${SLURM_STAGE_A_TIME}" \
  --output="${REPO_ROOT}/job_record/%x.%j.out" \
  --error="${REPO_ROOT}/job_record/%x.%j.err" \
  --export="ALL,EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG}" \
  --wrap="exec ${REPO_ROOT}/scripts/job_low_data_stage_a.sh")

STAGE_B_JOB_ID=$(sbatch --parsable \
  "${GPU_COMMON[@]}" \
  --job-name="ld_B_${SPLIT_ID}" \
  --time="${SLURM_STAGE_B_TIME}" \
  --dependency="afterok:${STAGE_A_JOB_ID}" \
  --array="${ARRAY_RANGE}" \
  --output="${REPO_ROOT}/job_record/%x.%A_%a.out" \
  --error="${REPO_ROOT}/job_record/%x.%A_%a.err" \
  --export="ALL,EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG},DELETE_TARGET_CHECKPOINT=1,DELETE_TARGET_CHECKPOINT_ON_FAILURE=1" \
  --wrap="exec ${REPO_ROOT}/scripts/job_low_data_stage_b.sh")

FINAL_JOB_ID=$(sbatch --parsable \
  "${CPU_ACCOUNT_ARGS[@]}" \
  "${QOS_ARGS[@]}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  "${FINALIZE_MEMORY_ARGS[@]}" \
  --partition="${SLURM_CPU_PARTITION}" \
  --job-name="ld_F_${SPLIT_ID}" \
  --time="${SLURM_FINALIZE_TIME}" \
  --dependency="afterany:${STAGE_B_JOB_ID}" \
  --output="${REPO_ROOT}/job_record/%x.%j.out" \
  --error="${REPO_ROOT}/job_record/%x.%j.err" \
  --export="ALL,EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG}" \
  --wrap="exec ${REPO_ROOT}/scripts/job_low_data_finalize.sh")

mkdir -p "${RESULTS_DIR}"
uv run python - "${RESULTS_DIR}/submission_manifest.json" "${EXPERIMENT_CONFIG}" \
  "${STAGE_A_JOB_ID}" "${STAGE_B_JOB_ID}" "${FINAL_JOB_ID}" "${ARRAY_RANGE}" \
  "${GRID_SIZE}" "${MAX_CONCURRENT}" "${SLURM_PARTITION}" "${SLURM_ACCOUNT:-}" \
  "${SLURM_STAGE_A_TIME}" "${SLURM_STAGE_B_TIME}" "${SUBMITTED_STAGE_B_CELLS}" <<'PY'
import datetime
import json
import os
import pathlib
import sys

from openpi.training.low_data.experiment import evaluation_workload, load_experiment_config

path = pathlib.Path(sys.argv[1])
config = load_experiment_config(sys.argv[2])
path.write_text(
    json.dumps(
        {
            "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scheduler": "native_slurm",
            "experiment_config": sys.argv[2],
            "split_id": config.split_id,
            "num_source_tasks": len(config.source_task_refs()),
            "num_target_tasks": len(config.target_task_refs()),
            "stage_a_job_id": sys.argv[3],
            "stage_b_array_job_id": sys.argv[4],
            "finalize_job_id": sys.argv[5],
            "stage_b_array_range": sys.argv[6],
            "stage_b_grid_size": int(sys.argv[7]),
            "submitted_stage_b_cells": int(sys.argv[13]),
            "evaluation_protocol": config.evaluation.protocol_manifest(),
            "evaluation_workload": evaluation_workload(config),
            "max_concurrent": int(sys.argv[8]),
            "slurm_partition": sys.argv[9],
            "slurm_account": sys.argv[10] or None,
            "stage_a_walltime": sys.argv[11],
            "stage_b_walltime": sys.argv[12],
            "stage_a_gpu_request": os.environ.get("SLURM_STAGE_A_GPU_GRES")
            or os.environ.get("SLURM_STAGE_A_GPUS_PER_NODE"),
            "stage_a_cpus_per_task": int(os.environ.get("SLURM_STAGE_A_CPUS_PER_TASK", "0")) or None,
            "stage_a_fsdp_devices": int(os.environ.get("OPENPI_SOURCE_FSDP_DEVICES", "1")),
            "stage_b_gpu_request": os.environ.get("SLURM_GPU_GRES") or os.environ.get("SLURM_GPUS_PER_NODE"),
        },
        indent=2,
    )
    + "\n"
)
PY

echo "Stage A job:    ${STAGE_A_JOB_ID}"
echo "Stage B array:  ${STAGE_B_JOB_ID} (${ARRAY_RANGE}, after successful Stage A)"
echo "Finalize job:   ${FINAL_JOB_ID} (after Stage B terminates)"
echo "Results:        ${RESULTS_DIR}"
