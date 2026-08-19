#!/bin/bash
# Select one leave-one-suite-out Stage-A config from a native Slurm array.

set -euo pipefail

REPO_ROOT=${OPENPI_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONFIGS=(
  examples/low_data/configs/libero_source_spatial_object_12source.json
  examples/low_data/configs/libero_source_spatial_goal_12source.json
  examples/low_data/configs/libero_source_object_goal_12source.json
)
ARRAY_INDEX=${SLURM_ARRAY_TASK_ID:?Submit this wrapper as a Slurm array}
if [ "${ARRAY_INDEX}" -lt 0 ] || [ "${ARRAY_INDEX}" -ge "${#CONFIGS[@]}" ]; then
  echo "Source-sweep array index ${ARRAY_INDEX} is outside 0-$((${#CONFIGS[@]} - 1))." >&2
  exit 2
fi

export EXPERIMENT_CONFIG="${REPO_ROOT}/${CONFIGS[${ARRAY_INDEX}]}"
exec "${REPO_ROOT}/scripts/job_low_data_stage_a.sh"
