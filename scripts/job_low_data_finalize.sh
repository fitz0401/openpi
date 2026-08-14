#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=02:00:00
#PBS -A starting_2026_047

# Aggregate, audit, plot, and package all results after the Stage-B array terminates.

set -uo pipefail

REPO_ROOT=/dodrio/scratch/projects/starting_2026_047/openpi
cd "${REPO_ROOT}"
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:?Set EXPERIMENT_CONFIG}

read -r RESULTS_DIR SPLIT_ID <<< "$(uv run python - "${EXPERIMENT_CONFIG}" <<'PY'
import pathlib
import sys
from openpi.training.low_data.experiment import load_experiment_config

config = load_experiment_config(sys.argv[1])
print(pathlib.Path(config.results_root) / config.split_id, config.split_id)
PY
)"
RESULTS_NAMESPACE=$(basename "$(dirname "${RESULTS_DIR}")")
ARCHIVE_PATH="$(dirname "${RESULTS_DIR}")/${RESULTS_NAMESPACE}__${SPLIT_ID}.tar.gz"

echo "===== LOW-DATA FINALIZE ====="
echo "EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG} RESULTS_DIR=${RESULTS_DIR}"

uv run scripts/audit_low_data_results.py --experiment-config "${EXPERIMENT_CONFIG}"

if find "${RESULTS_DIR}/runs" -name tidy_results.jsonl -print -quit 2>/dev/null | grep -q .; then
  uv run scripts/aggregate_low_data_results.py --results-dir "${RESULTS_DIR}"
  uv run scripts/plot_low_data_main_results.py --results-dir "${RESULTS_DIR}" || \
    echo "WARNING: plotting failed; tidy tables and workflow audit are still available." >&2
  tar -czf "${ARCHIVE_PATH}" -C "$(dirname "${RESULTS_DIR}")" "${SPLIT_ID}"
  echo "Download-ready archive: ${ARCHIVE_PATH}"
else
  echo "WARNING: no completed Stage-B result was found; see ${RESULTS_DIR}/workflow_status.json" >&2
fi
