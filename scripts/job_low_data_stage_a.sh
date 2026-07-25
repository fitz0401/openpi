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
JOB_TOKEN=${PBS_JOBID:-$$}
JOB_TOKEN=${JOB_TOKEN//[!a-zA-Z0-9_.-]/_}
JOB_TMPDIR=${JOB_TMPDIR:-${TMPDIR:-${VSC_SCRATCH_NODE:-/tmp}}/${USER}/openpi_low_data_${JOB_TOKEN}}
DESCRIPTOR="${JOB_TMPDIR}/low_data_source_train_manifest.json"
mkdir -p "${JOB_TMPDIR}"
SOURCE_RESULT_DIR=$(uv run python - "${EXPERIMENT_CONFIG}" <<'PY'
import pathlib
import sys
from openpi.training.low_data.experiment import load_experiment_config

config = load_experiment_config(sys.argv[1])
print(pathlib.Path(config.results_root) / config.split_id / "source")
PY
)
PERSISTENT_MANIFEST="${SOURCE_RESULT_DIR}/train_manifest.json"
if [ -s "${SOURCE_RESULT_DIR}/source_eval.json" ] && [ -s "${SOURCE_RESULT_DIR}/target_zero_shot_eval.json" ]; then
  echo "Stage A and zero-shot evaluation already complete; skipping ${SOURCE_RESULT_DIR}"
  exit 0
fi
PORT_SEED="${SLURM_JOB_ID:-${PBS_JOBID:-$$}}"
PORT_SEED=${PORT_SEED//[!0-9]/}
PORT=${PORT:-$((18000 + 10#${PORT_SEED:-1} % 20000))}

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

REUSE_CHECKPOINT=$(uv run python - "${PERSISTENT_MANIFEST}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    print(0)
else:
    manifest = json.loads(path.read_text())
    print(int((pathlib.Path(manifest["checkpoint_dir"]) / "params").is_dir()))
PY
)
if [ "${REUSE_CHECKPOINT}" = 1 ]; then
  echo "Reusing trained Stage-A checkpoint from ${PERSISTENT_MANIFEST}"
  DESCRIPTOR="${PERSISTENT_MANIFEST}"
else
  source .venv/bin/activate
  uv run scripts/low_data_train.py \
    --experiment-config "${EXPERIMENT_CONFIG}" \
    --stage source \
    --seed 0 \
    --descriptor-out "${DESCRIPTOR}"
  deactivate
fi

source "${LIBERO_VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/libero:${PYTHONPATH:-}"
python scripts/low_data_eval.py \
  --experiment-config "${EXPERIMENT_CONFIG}" \
  --train-manifest "${DESCRIPTOR}" \
  --repo-root "${REPO_ROOT}" \
  --port "${PORT}" \
  --server-gpu 0
