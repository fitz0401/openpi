#!/bin/bash
# Download one continual run's result files from Dodrio and render all plots locally.

set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-vsc39029@tier1.hpc.ugent.be}
REMOTE_REPO=${REMOTE_REPO:-/dodrio/scratch/projects/starting_2026_047/openpi}
RUN_NAME=slice_v0
BUDGET=50
SEED=0
DEST=""
PLOT=true

usage() {
  echo "Usage: $0 [--budget N] [--seed N] [--run-name NAME] [--dest DIR] [--no-plot]"
  echo "Environment overrides: REMOTE_HOST, REMOTE_REPO"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --budget) BUDGET=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --run-name) RUN_NAME=$2; shift 2 ;;
    --dest) DEST=$2; shift 2 ;;
    --no-plot) PLOT=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DEST=${DEST:-${REPO_ROOT}/experiments/budget${BUDGET}_seed${SEED}}
REMOTE_RUN_DIR="${REMOTE_REPO}/checkpoints/pi05_libero_object_continual/${RUN_NAME}/budget${BUDGET}/seed${SEED}"

mkdir -p "${DEST}"

RESULT_FILES=(
  manifest.json
  success_matrix.csv
  learning_curves.csv
  eval_results.json
  metrics.json
)

echo "Downloading ${REMOTE_HOST}:${REMOTE_RUN_DIR} -> ${DEST}"
for file in "${RESULT_FILES[@]}"; do
  scp "${REMOTE_HOST}:${REMOTE_RUN_DIR}/${file}" "${DEST}/${file}"
done

# Reproducibility metadata are useful but older runs may not contain every file.
scp "${REMOTE_HOST}:${REMOTE_RUN_DIR}/stage*_sampled_indices.json" "${DEST}/" || \
  echo "Warning: no sampled-index metadata downloaded." >&2

REMOTE_SUMMARY="${REMOTE_REPO}/results/${RUN_NAME}_budget${BUDGET}_seed${SEED}_summary.csv"
scp "${REMOTE_HOST}:${REMOTE_SUMMARY}" "${DEST}/metrics_summary.csv" || \
  echo "Warning: per-run summary CSV is not available." >&2

if [ "${PLOT}" = true ]; then
  cd "${REPO_ROOT}"
  uv run python scripts/plot_continual_results.py --run-dir "${DEST}"
fi

echo "Done: ${DEST}"
