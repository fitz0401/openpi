#!/bin/bash
# Submit the full target grid as one throttled PBS job array.

set -euo pipefail

EXPERIMENT_CONFIG=${1:?Usage: $0 <experiment-config.json>}
EXPERIMENT_CONFIG=$(realpath "${EXPERIMENT_CONFIG}")
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MAX_CONCURRENT=${MAX_CONCURRENT:-2}

if ! [[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT must be a positive integer (got ${MAX_CONCURRENT})" >&2
  exit 2
fi

cd "${REPO_ROOT}"
GRID_SIZE=$(uv run python - "${EXPERIMENT_CONFIG}" <<'PY'
import sys
from openpi.training.low_data.experiment import load_experiment_config, target_grid

config = load_experiment_config(sys.argv[1])
print(len(target_grid(config)))
PY
)

if [ "${GRID_SIZE}" -lt 1 ]; then
  echo "Stage-B grid is empty: ${EXPERIMENT_CONFIG}" >&2
  exit 2
fi

ARRAY_RANGE="0-$((GRID_SIZE - 1))%${MAX_CONCURRENT}"
echo "Submitting ${GRID_SIZE} Stage-B cells as array ${ARRAY_RANGE}"
qsub -t "${ARRAY_RANGE}" \
  -v "EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG}" \
  scripts/job_low_data_stage_b.sh
