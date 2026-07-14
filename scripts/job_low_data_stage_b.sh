#!/bin/bash
#PBS -l nodes=1:ppn=12:gpus=1
#PBS -l walltime=48:00:00
#PBS -A starting_2026_047

# Train and evaluate ONE independent (target, method, num_demos, seed) adaptation run.

set -euo pipefail

REPO_ROOT=/dodrio/scratch/projects/starting_2026_047/openpi
cd "${REPO_ROOT}"
module load cluster/dodrio/gpu_rome_a100_80_rhel9
module load CUDA

EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:?Set EXPERIMENT_CONFIG}

# A throttled PBS array supplies only an index. Resolve it deterministically using the same
# product order as the experiment config. Explicit variables still support submitting one cell.
ARRAY_INDEX=${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-${SLURM_ARRAY_TASK_ID:-}}}
if [ -n "${ARRAY_INDEX}" ]; then
  GRID_ENTRY=$(uv run python - "${EXPERIMENT_CONFIG}" "${ARRAY_INDEX}" <<'PY'
import sys

from openpi.training.low_data.experiment import load_experiment_config, target_grid

config = load_experiment_config(sys.argv[1])
try:
    print(*target_grid(config)[int(sys.argv[2])])
except IndexError as exc:
    raise SystemExit(f"Array index {sys.argv[2]} is outside the Stage-B grid") from exc
PY
  )
  read -r TARGET_TASK_ID METHOD NUM_DEMOS SEED BUDGET_NAME <<< "${GRID_ENTRY}"
fi

TARGET_TASK_ID=${TARGET_TASK_ID:?Set TARGET_TASK_ID or submit as a PBS array}
METHOD=${METHOD:?Set METHOD to full or lora}
NUM_DEMOS=${NUM_DEMOS:?Set NUM_DEMOS}
SEED=${SEED:-0}
BUDGET_NAME=${BUDGET_NAME:-}
DELETE_TARGET_CHECKPOINT=${DELETE_TARGET_CHECKPOINT:-1}
LIBERO_VENV=${LIBERO_VENV:-${REPO_ROOT}/examples/libero/.venv}
JOB_TMPDIR=${TMPDIR:-${VSC_SCRATCH_NODE:-/tmp}/${USER}/${PBS_JOBID:-$$}}
DESCRIPTOR="${JOB_TMPDIR}/low_data_target_train_manifest.json"
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

echo "===== LOW-DATA STAGE B ====="
echo "CONFIG=${EXPERIMENT_CONFIG} TARGET=${TARGET_TASK_ID} METHOD=${METHOD} DEMOS=${NUM_DEMOS} SEED=${SEED} BUDGET=${BUDGET_NAME:-auto} DELETE=${DELETE_TARGET_CHECKPOINT}"
df -h "${REPO_ROOT}" "${JOB_TMPDIR}" || true
nvidia-smi

source .venv/bin/activate
BUDGET_ARGS=()
if [ -n "${BUDGET_NAME}" ]; then
  BUDGET_ARGS+=(--budget-name "${BUDGET_NAME}")
fi
uv run scripts/low_data_train.py \
  --experiment-config "${EXPERIMENT_CONFIG}" \
  --stage target \
  --target-task-id "${TARGET_TASK_ID}" \
  --method "${METHOD}" \
  --num-demos "${NUM_DEMOS}" \
  "${BUDGET_ARGS[@]}" \
  --seed "${SEED}" \
  --descriptor-out "${DESCRIPTOR}"
deactivate

DELETE_ARGS=()
if [ "${DELETE_TARGET_CHECKPOINT}" = 1 ]; then
  DELETE_ARGS+=(--delete-checkpoint-after-eval)
fi
source "${LIBERO_VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/libero:${PYTHONPATH:-}"
python scripts/low_data_eval.py \
  --experiment-config "${EXPERIMENT_CONFIG}" \
  --train-manifest "${DESCRIPTOR}" \
  --repo-root "${REPO_ROOT}" \
  --server-gpu 0 \
  "${DELETE_ARGS[@]}"
