#!/bin/bash
#PBS -l nodes=1:ppn=12:gpus=1
#PBS -l walltime=48:00:00
#PBS -A starting_2026_047

# Evaluate ONE (budget, seed) continual run: fills the success matrix + learning curves by running
# LIBERO rollouts against per-checkpoint policy servers, then computes metrics.
#
# The eval CLIENT runs in the LIBERO venv (examples/libero/.venv, per examples/libero/README.md --
# no conda on this cluster); it launches policy SERVERS via `uv run` (openpi env). Both use the
# single GPU on this node (server holds the model; the sim is CPU/EGL).
#
# One-time setup for the LIBERO venv (if not already done):
#   git submodule update --init --recursive
#   uv venv --python 3.8 examples/libero/.venv
#   source examples/libero/.venv/bin/activate
#   uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
#     --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
#   uv pip install -e packages/openpi-client
#   uv pip install -e third_party/libero
#
# Usage (mirror the train jobs):
#   qsub -v BUDGET=1  scripts/job_continual_eval.sh
#   qsub -v BUDGET=10 scripts/job_continual_eval.sh
#   qsub -v BUDGET=50 scripts/job_continual_eval.sh
# Optional overrides: SEED, RUN_NAME, NUM_TRIALS, NUM_TRIALS_LC, LIBERO_VENV,
# LIBERO_CONFIG_PATH.

set -euo pipefail

REPO_ROOT=/dodrio/scratch/projects/starting_2026_047/openpi
cd "${REPO_ROOT}"

module load cluster/dodrio/gpu_rome_a100_80_rhel9
module load CUDA

export HF_HOME=/dodrio/scratch/projects/starting_2026_047/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE=$HF_HOME
# Redirect the openpi checkpoint download cache (used if evaluating a gs:// baseline checkpoint)
# to project scratch; inherited by the serve_policy.py subprocess continual_eval.py launches.
export OPENPI_DATA_HOME=/dodrio/scratch/projects/starting_2026_047/cache/openpi
# Server uses GPU 0 on this node; headless mujoco for the sim client.
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=${MUJOCO_GL:-egl}

# Numba otherwise writes JIT cache files next to the installed modules in the LIBERO venv, which
# lives on quota-limited project scratch. Use this job's node-local scratch for disposable caches.
JOB_TMPDIR=${TMPDIR:-${VSC_SCRATCH_NODE:-/tmp}/${USER}/${PBS_JOBID:-$$}}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-${JOB_TMPDIR}/numba}
mkdir -p "${NUMBA_CACHE_DIR}"

BUDGET=${BUDGET:-10}
SEED=${SEED:-0}
RUN_NAME=${RUN_NAME:-slice_v0}
NUM_TRIALS=${NUM_TRIALS:-20}
NUM_TRIALS_LC=${NUM_TRIALS_LC:-10}
LIBERO_VENV=${LIBERO_VENV:-${REPO_ROOT}/examples/libero/.venv}

# LIBERO prompts for its config path on first import. Batch jobs have no stdin, so create the
# default config up front in job-local scratch (or at LIBERO_CONFIG_PATH when explicitly set).
# JSON is valid YAML and avoids importing LIBERO before the config exists.
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${JOB_TMPDIR}/libero}
LIBERO_PACKAGE_ROOT="${REPO_ROOT}/third_party/libero/libero/libero"
LIBERO_CONFIG_FILE="${LIBERO_CONFIG_PATH}/config.yaml"
mkdir -p "${LIBERO_CONFIG_PATH}"
if [ ! -f "${LIBERO_CONFIG_FILE}" ]; then
  "${LIBERO_VENV}/bin/python" - "${LIBERO_CONFIG_FILE}" "${LIBERO_PACKAGE_ROOT}" <<'PY'
import json
import os
import pathlib
import sys

config_file = pathlib.Path(sys.argv[1])
libero_root = pathlib.Path(sys.argv[2]).resolve()
dataset_root = config_file.parent / "datasets"
dataset_root.mkdir(parents=True, exist_ok=True)
config = {
    "benchmark_root": str(libero_root),
    "bddl_files": str(libero_root / "bddl_files"),
    "init_states": str(libero_root / "init_files"),
    "datasets": str(dataset_root),
    "assets": str(libero_root / "assets"),
}
tmp_file = config_file.with_name(f"{config_file.name}.{os.getpid()}.tmp")
tmp_file.write_text(json.dumps(config, indent=2) + "\n")
os.replace(tmp_file, config_file)
PY
fi

MANIFEST="${REPO_ROOT}/checkpoints/pi05_libero_object_continual/${RUN_NAME}/budget${BUDGET}/seed${SEED}/manifest.json"
RUN_DIR="$(dirname "${MANIFEST}")"

# Fail before loading a multi-GiB policy if project scratch cannot accept even a tiny result file.
WRITE_PROBE="${RUN_DIR}/.continual_eval_write_probe_${PBS_JOBID:-$$}"
if ! : > "${WRITE_PROBE}"; then
  echo "ERROR: Cannot write to ${RUN_DIR}; check project byte/inode quota with:" >&2
  echo "  my_dodrio_quota -p starting_2026_047" >&2
  exit 1
fi
rm -f "${WRITE_PROBE}"

echo "===== CONTINUAL EVAL ====="
hostname; date
echo "MANIFEST=${MANIFEST} NUM_TRIALS=${NUM_TRIALS} NUM_TRIALS_LC=${NUM_TRIALS_LC} LIBERO_VENV=${LIBERO_VENV} LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH} NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR}"
df -h "${RUN_DIR}" "${JOB_TMPDIR}" || true
df -i "${RUN_DIR}" "${JOB_TMPDIR}" || true
nvidia-smi

# Activate the LIBERO venv for the rollout client (no conda on this cluster).
source "${LIBERO_VENV}/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}/third_party/libero"

# Run rollouts -> success_matrix.csv + learning_curves.csv (servers launched via uv internally).
python scripts/continual_eval.py \
  --manifest "${MANIFEST}" \
  --repo-root "${REPO_ROOT}" \
  --num-trials "${NUM_TRIALS}" \
  --num-trials-lc "${NUM_TRIALS_LC}" \
  --server-gpu 0

# Compute metrics in the openpi (uv) env.
deactivate
source .venv/bin/activate
uv run scripts/continual_metrics.py \
  --run-dirs "${RUN_DIR}" \
  --summary-out "${REPO_ROOT}/results/${RUN_NAME}_budget${BUDGET}_seed${SEED}_summary.csv"
