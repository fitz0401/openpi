#!/bin/bash
# Submit Stage A -> throttled Stage B -> final aggregation as one PBS dependency chain.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXPERIMENT_CONFIG=${1:-examples/low_data/configs/libero_main_18source_22target.json}
EXPERIMENT_CONFIG=$(realpath "${EXPERIMENT_CONFIG}")
MAX_CONCURRENT=${MAX_CONCURRENT:-4}

if ! [[ "${MAX_CONCURRENT}" =~ ^[1-4]$ ]]; then
  echo "MAX_CONCURRENT must be between 1 and 4 for the unattended main run (got ${MAX_CONCURRENT})." >&2
  exit 2
fi

cd "${REPO_ROOT}"
read -r GRID_SIZE NUM_SOURCE NUM_TARGET NUM_SEEDS NUM_LIBERO10_SEEDS NUM_ALL_AVAILABLE_SEEDS METHODS NUM_TRIALS RESULTS_DIR <<< "$(uv run python - "${EXPERIMENT_CONFIG}" <<'PY'
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
    config.evaluation.num_trials,
    pathlib.Path(config.results_root) / config.split_id,
)
PY
)"

if [ "${NUM_SOURCE}" -ne 18 ] || [ "${NUM_TARGET}" -ne 22 ]; then
  echo "Refusing unexpected Split-A task manifest: source=${NUM_SOURCE}, target=${NUM_TARGET}" >&2
  exit 2
fi
if [ "${METHODS}" != lora ] || [ "${NUM_SEEDS}" -ne 3 ] || [ "${NUM_LIBERO10_SEEDS}" -ne 1 ] || [ "${NUM_ALL_AVAILABLE_SEEDS}" -ne 1 ] || [ "${GRID_SIZE}" -ne 206 ] || [ "${NUM_TRIALS}" -ne 25 ]; then
  echo "Refusing unexpected paper grid: methods=${METHODS}, global_seeds=${NUM_SEEDS}, libero10_seeds=${NUM_LIBERO10_SEEDS}, all_available_seeds=${NUM_ALL_AVAILABLE_SEEDS}, cells=${GRID_SIZE}, num_trials=${NUM_TRIALS}" >&2
  exit 2
fi

echo "Submitting paper Split-A: ${NUM_SOURCE} source tasks, ${NUM_TARGET} target tasks, ${NUM_SEEDS} seeds."
echo "Stage B: ${GRID_SIZE} LoRA-only cells; LIBERO-10 and all_available use seed 0, ${NUM_TRIALS} trials, maximum ${MAX_CONCURRENT} concurrent."
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
  "${STAGE_A_JOB_ID}" "${STAGE_B_JOB_ID}" "${FINAL_JOB_ID}" "${ARRAY_RANGE}" \
  "${GRID_SIZE}" "${NUM_SEEDS}" "${NUM_LIBERO10_SEEDS}" "${NUM_ALL_AVAILABLE_SEEDS}" "${METHODS}" \
  "${NUM_TRIALS}" "${MAX_CONCURRENT}" <<'PY'
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
            "stage_b_grid_size": int(sys.argv[7]),
            "num_adaptation_seeds": int(sys.argv[8]),
            "num_libero_10_seeds": int(sys.argv[9]),
            "num_all_available_seeds": int(sys.argv[10]),
            "methods": sys.argv[11].split(","),
            "num_trials": int(sys.argv[12]),
            "max_concurrent": int(sys.argv[13]),
            "stage_b_walltime": "24:00:00",
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
