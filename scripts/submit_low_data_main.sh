#!/bin/bash
# Submit Stage A -> throttled Stage B -> final aggregation as one PBS dependency chain.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXPERIMENT_CONFIG=${1:-examples/low_data/configs/libero_main_18source_22target.json}
EXPERIMENT_CONFIG=$(realpath "${EXPERIMENT_CONFIG}")
MAX_CONCURRENT=${MAX_CONCURRENT:-2}

if ! [[ "${MAX_CONCURRENT}" =~ ^[12]$ ]]; then
  echo "MAX_CONCURRENT must be 1 or 2 for the unattended main run (got ${MAX_CONCURRENT})." >&2
  exit 2
fi

cd "${REPO_ROOT}"
read -r GRID_SIZE NUM_SOURCE NUM_TARGET RESULTS_DIR <<< "$(uv run python - "${EXPERIMENT_CONFIG}" <<'PY'
import pathlib
import sys
from openpi.training.low_data.experiment import load_experiment_config, target_grid

config = load_experiment_config(sys.argv[1])
print(
    len(target_grid(config)),
    len(config.source_task_refs()),
    len(config.target_task_refs()),
    pathlib.Path(config.results_root) / config.split_id,
)
PY
)"

if [ "${NUM_SOURCE}" -ne 18 ] || [ "${NUM_TARGET}" -ne 22 ] || [ "${GRID_SIZE}" -ne 176 ]; then
  echo "Refusing unexpected main grid: source=${NUM_SOURCE}, target=${NUM_TARGET}, cells=${GRID_SIZE}" >&2
  exit 2
fi

echo "Submitting unified ${NUM_SOURCE}-source / ${NUM_TARGET}-target experiment."
echo "Stage B: ${GRID_SIZE} independent cells, maximum ${MAX_CONCURRENT} concurrent."
my_dodrio_quota -p starting_2026_047 || true

STAGE_A_JOB_ID=$(qsub \
  -N ld_main_A \
  -v "EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG}" \
  scripts/job_low_data_stage_a.sh | tail -n 1)

ARRAY_RANGE="0-$((GRID_SIZE - 1))%${MAX_CONCURRENT}"
STAGE_B_JOB_ID=$(qsub \
  -N ld_main_B \
  -W "depend=afterok:${STAGE_A_JOB_ID}" \
  -t "${ARRAY_RANGE}" \
  -v "EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG},DELETE_TARGET_CHECKPOINT=1,DELETE_TARGET_CHECKPOINT_ON_FAILURE=1" \
  scripts/job_low_data_stage_b.sh | tail -n 1)

FINAL_JOB_ID=$(qsub \
  -N ld_finalize \
  -W "depend=afterany:${STAGE_B_JOB_ID}" \
  -v "EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG}" \
  scripts/job_low_data_finalize.sh | tail -n 1)

mkdir -p "${RESULTS_DIR}"
uv run python - "${RESULTS_DIR}/submission_manifest.json" "${EXPERIMENT_CONFIG}" \
  "${STAGE_A_JOB_ID}" "${STAGE_B_JOB_ID}" "${FINAL_JOB_ID}" "${ARRAY_RANGE}" <<'PY'
import datetime
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "experiment_config": sys.argv[2],
            "stage_a_job_id": sys.argv[3],
            "stage_b_array_job_id": sys.argv[4],
            "finalize_job_id": sys.argv[5],
            "stage_b_array_range": sys.argv[6],
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
