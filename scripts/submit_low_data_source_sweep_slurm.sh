#!/bin/bash
# Submit the three 12-source leave-one-suite-out Stage-A checkpoints on native Slurm.

set -euo pipefail

REPO_ROOT=${OPENPI_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MAX_SOURCE_CONCURRENT=${MAX_SOURCE_CONCURRENT:-2}
SLURM_PARTITION=${SLURM_PARTITION:?Set SLURM_PARTITION}
SLURM_GPU_GRES=${SLURM_GPU_GRES:-gpu:1}
SLURM_CPUS_PER_GPU=${SLURM_CPUS_PER_GPU:-12}
SLURM_GPU_MEMORY=${SLURM_GPU_MEMORY:-125G}
SLURM_STAGE_A_TIME=${SLURM_STAGE_A_TIME:-48:00:00}

if ! [[ "${MAX_SOURCE_CONCURRENT}" =~ ^[1-3]$ ]]; then
  echo "MAX_SOURCE_CONCURRENT must be between 1 and 3." >&2
  exit 2
fi
if [ "${OPENPI_SKIP_MODULES:-0}" != 1 ] && [ -z "${OPENPI_GPU_MODULES:-}" ]; then
  echo "Set OPENPI_GPU_MODULES or OPENPI_SKIP_MODULES=1." >&2
  exit 2
fi

cd "${REPO_ROOT}"
mkdir -p job_record
for config in \
  examples/low_data/configs/libero_source_spatial_object_12source.json \
  examples/low_data/configs/libero_source_spatial_goal_12source.json \
  examples/low_data/configs/libero_source_object_goal_12source.json; do
  uv run python - "${config}" <<'PY'
import sys
from openpi.training.low_data.experiment import load_experiment_config

config = load_experiment_config(sys.argv[1])
assert len(config.source_task_refs()) == 12
assert len(config.target_task_refs()) == 22
PY
done

ACCOUNT_ARGS=()
if [ -n "${SLURM_ACCOUNT:-}" ]; then ACCOUNT_ARGS+=(--account="${SLURM_ACCOUNT}"); fi
QOS_ARGS=()
if [ -n "${SLURM_QOS:-}" ]; then QOS_ARGS+=(--qos="${SLURM_QOS}"); fi
CONSTRAINT_ARGS=()
if [ -n "${SLURM_CONSTRAINT:-}" ]; then CONSTRAINT_ARGS+=(--constraint="${SLURM_CONSTRAINT}"); fi

JOB_ID=$(sbatch --parsable \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_GPU}" \
  --mem="${SLURM_GPU_MEMORY}" \
  --gres="${SLURM_GPU_GRES}" \
  --partition="${SLURM_PARTITION}" \
  "${ACCOUNT_ARGS[@]}" \
  "${QOS_ARGS[@]}" \
  "${CONSTRAINT_ARGS[@]}" \
  --job-name=ld_source12 \
  --time="${SLURM_STAGE_A_TIME}" \
  --array="0-2%${MAX_SOURCE_CONCURRENT}" \
  --output="${REPO_ROOT}/job_record/%x.%A_%a.out" \
  --error="${REPO_ROOT}/job_record/%x.%A_%a.err" \
  --export=ALL \
  scripts/job_low_data_source_sweep.sh)

echo "12-source Stage-A array: ${JOB_ID} (0-2%${MAX_SOURCE_CONCURRENT})"
