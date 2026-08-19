#!/bin/bash
# Submit the formal Stage A -> Stage B array -> finalizer chain with native Slurm sbatch.

set -euo pipefail

REPO_ROOT=${OPENPI_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
EXPERIMENT_CONFIG=${1:-${REPO_ROOT}/examples/low_data/configs/libero_main_18source_22target.json}
EXPERIMENT_CONFIG=$(realpath "${EXPERIMENT_CONFIG}")
OPENPI_CACHE_ROOT=${OPENPI_CACHE_ROOT:-${SCRATCH:-${HOME}/.cache}/openpi_low_data_cache}
LIBERO_VENV=${LIBERO_VENV:-${REPO_ROOT}/examples/libero/.venv}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}

SLURM_PARTITION=${SLURM_PARTITION:?Set SLURM_PARTITION for GPU jobs}
SLURM_CPU_PARTITION=${SLURM_CPU_PARTITION:-${SLURM_PARTITION}}
SLURM_GPU_GRES=${SLURM_GPU_GRES:-gpu:1}
SLURM_CPUS_PER_GPU=${SLURM_CPUS_PER_GPU:-12}
SLURM_GPU_MEMORY=${SLURM_GPU_MEMORY:-125G}
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

read -r GRID_SIZE NUM_SOURCE NUM_TARGET NUM_SEEDS NUM_LIBERO10_SEEDS NUM_ALL_AVAILABLE_SEEDS METHODS EVAL_PROTOCOL_ID RESULTS_DIR <<< "$(uv run python - "${EXPERIMENT_CONFIG}" <<'PY'
import pathlib
import sys

from openpi.training.low_data.experiment import load_experiment_config, target_grid

config = load_experiment_config(sys.argv[1])
print(
    len(target_grid(config)),
    len(config.source_task_refs()),
    len(config.target_task_refs()),
    len(config.adaptation.seeds),
    len(config.adaptation.seeds_for("libero_10", "1")),
    len(config.adaptation.seeds_for("libero_spatial", "all_available")),
    ",".join(config.adaptation.methods),
    config.evaluation.protocol_id,
    pathlib.Path(config.results_root) / config.split_id,
)
PY
)"

if [ "${NUM_SOURCE}" -ne 18 ] || [ "${NUM_TARGET}" -ne 22 ] || [ "${METHODS}" != lora ] || \
   [ "${NUM_SEEDS}" -ne 3 ] || [ "${NUM_LIBERO10_SEEDS}" -ne 1 ] || \
   [ "${NUM_ALL_AVAILABLE_SEEDS}" -ne 1 ] || [ "${GRID_SIZE}" -ne 206 ] || \
   [ "${EVAL_PROTOCOL_ID}" != sog_target50_l10_target25_retention25_seed0_no_l10_retention ]; then
  echo "Refusing unexpected protocol: source=${NUM_SOURCE}, target=${NUM_TARGET}, methods=${METHODS}, " \
       "seeds=${NUM_SEEDS}, libero10_seeds=${NUM_LIBERO10_SEEDS}, all_available_seeds=${NUM_ALL_AVAILABLE_SEEDS}, " \
       "cells=${GRID_SIZE}, evaluation_protocol=${EVAL_PROTOCOL_ID}" >&2
  exit 2
fi

ACCOUNT_ARGS=()
if [ -n "${SLURM_ACCOUNT:-}" ]; then ACCOUNT_ARGS+=(--account="${SLURM_ACCOUNT}"); fi
QOS_ARGS=()
if [ -n "${SLURM_QOS:-}" ]; then QOS_ARGS+=(--qos="${SLURM_QOS}"); fi
CONSTRAINT_ARGS=()
if [ -n "${SLURM_CONSTRAINT:-}" ]; then CONSTRAINT_ARGS+=(--constraint="${SLURM_CONSTRAINT}"); fi

GPU_COMMON=(
  --nodes=1
  --ntasks=1
  --cpus-per-task="${SLURM_CPUS_PER_GPU}"
  --mem="${SLURM_GPU_MEMORY}"
  --gres="${SLURM_GPU_GRES}"
  --partition="${SLURM_PARTITION}"
  "${ACCOUNT_ARGS[@]}"
  "${QOS_ARGS[@]}"
  "${CONSTRAINT_ARGS[@]}"
)

echo "Submitting native-Slurm paper grid: ${GRID_SIZE} cells, evaluation=${EVAL_PROTOCOL_ID}, concurrency ${MAX_CONCURRENT}."
echo "GPU partition=${SLURM_PARTITION} gres=${SLURM_GPU_GRES} memory=${SLURM_GPU_MEMORY}"

STAGE_A_JOB_ID=$(sbatch --parsable \
  "${GPU_COMMON[@]}" \
  --job-name=ld_main_A \
  --time="${SLURM_STAGE_A_TIME}" \
  --output="${REPO_ROOT}/job_record/%x.%j.out" \
  --error="${REPO_ROOT}/job_record/%x.%j.err" \
  --export="ALL,EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG}" \
  scripts/job_low_data_stage_a.sh)

ARRAY_RANGE="0-$((GRID_SIZE - 1))%${MAX_CONCURRENT}"
STAGE_B_JOB_ID=$(sbatch --parsable \
  "${GPU_COMMON[@]}" \
  --job-name=ld_main_B \
  --time="${SLURM_STAGE_B_TIME}" \
  --dependency="afterok:${STAGE_A_JOB_ID}" \
  --array="${ARRAY_RANGE}" \
  --output="${REPO_ROOT}/job_record/%x.%A_%a.out" \
  --error="${REPO_ROOT}/job_record/%x.%A_%a.err" \
  --export="ALL,EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG},DELETE_TARGET_CHECKPOINT=1,DELETE_TARGET_CHECKPOINT_ON_FAILURE=1" \
  scripts/job_low_data_stage_b.sh)

FINAL_JOB_ID=$(sbatch --parsable \
  "${ACCOUNT_ARGS[@]}" \
  "${QOS_ARGS[@]}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --partition="${SLURM_CPU_PARTITION}" \
  --job-name=ld_finalize \
  --time="${SLURM_FINALIZE_TIME}" \
  --dependency="afterany:${STAGE_B_JOB_ID}" \
  --output="${REPO_ROOT}/job_record/%x.%j.out" \
  --error="${REPO_ROOT}/job_record/%x.%j.err" \
  --export="ALL,EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG}" \
  scripts/job_low_data_finalize.sh)

mkdir -p "${RESULTS_DIR}"
uv run python - "${RESULTS_DIR}/submission_manifest.json" "${EXPERIMENT_CONFIG}" \
  "${STAGE_A_JOB_ID}" "${STAGE_B_JOB_ID}" "${FINAL_JOB_ID}" "${ARRAY_RANGE}" \
  "${GRID_SIZE}" "${MAX_CONCURRENT}" "${SLURM_PARTITION}" "${SLURM_ACCOUNT:-}" \
  "${SLURM_STAGE_A_TIME}" "${SLURM_STAGE_B_TIME}" <<'PY'
import datetime
import json
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
            "stage_a_job_id": sys.argv[3],
            "stage_b_array_job_id": sys.argv[4],
            "finalize_job_id": sys.argv[5],
            "stage_b_array_range": sys.argv[6],
            "stage_b_grid_size": int(sys.argv[7]),
            "evaluation_protocol": config.evaluation.protocol_manifest(),
            "evaluation_workload": evaluation_workload(config),
            "max_concurrent": int(sys.argv[8]),
            "slurm_partition": sys.argv[9],
            "slurm_account": sys.argv[10] or None,
            "stage_a_walltime": sys.argv[11],
            "stage_b_walltime": sys.argv[12],
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
