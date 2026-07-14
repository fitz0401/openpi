#!/bin/bash
#PBS -l nodes=1:ppn=12:gpus=1
#PBS -l walltime=48:00:00
#PBS -A starting_2026_047

# Joint source adaptation followed by source and unseen-target zero-shot evaluation.

set -euo pipefail

REPO_ROOT=/dodrio/scratch/projects/starting_2026_047/openpi
cd "${REPO_ROOT}"
module load cluster/dodrio/gpu_rome_a100_80_rhel9
module load CUDA

EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:?Set EXPERIMENT_CONFIG to a low-data JSON config}
LIBERO_VENV=${LIBERO_VENV:-${REPO_ROOT}/examples/libero/.venv}
JOB_TMPDIR=${TMPDIR:-${VSC_SCRATCH_NODE:-/tmp}/${USER}/${PBS_JOBID:-$$}}
DESCRIPTOR="${JOB_TMPDIR}/low_data_source_train_manifest.json"
mkdir -p "${JOB_TMPDIR}"

export HF_HOME=/dodrio/scratch/projects/starting_2026_047/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE=$HF_HOME
export OPENPI_DATA_HOME=/dodrio/scratch/projects/starting_2026_047/cache/openpi
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=${MUJOCO_GL:-egl}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-${JOB_TMPDIR}/numba}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${JOB_TMPDIR}/libero}
mkdir -p "${NUMBA_CACHE_DIR}" "${LIBERO_CONFIG_PATH}"

"${LIBERO_VENV}/bin/python" - "${LIBERO_CONFIG_PATH}/config.yaml" "${REPO_ROOT}/third_party/libero/libero/libero" <<'PY'
import json
import pathlib
import sys
config_file = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2]).resolve()
datasets = config_file.parent / "datasets"
datasets.mkdir(parents=True, exist_ok=True)
config_file.write_text(json.dumps({
    "benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"), "datasets": str(datasets),
    "assets": str(root / "assets"),
}, indent=2) + "\n")
PY

echo "===== LOW-DATA STAGE A ====="
echo "EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG} DESCRIPTOR=${DESCRIPTOR}"
df -h "${REPO_ROOT}" "${JOB_TMPDIR}" || true
nvidia-smi

source .venv/bin/activate
uv run scripts/low_data_train.py \
  --experiment-config "${EXPERIMENT_CONFIG}" \
  --stage source \
  --seed 0 \
  --descriptor-out "${DESCRIPTOR}"
deactivate

source "${LIBERO_VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/libero:${PYTHONPATH:-}"
python scripts/low_data_eval.py \
  --experiment-config "${EXPERIMENT_CONFIG}" \
  --train-manifest "${DESCRIPTOR}" \
  --repo-root "${REPO_ROOT}" \
  --server-gpu 0
