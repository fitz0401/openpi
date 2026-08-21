#!/bin/bash
#PBS -l nodes=1:ppn=12:gpus=1
#PBS -l walltime=12:00:00
#PBS -A starting_2026_047

# Train and evaluate ONE independent (target, method, data_budget, seed) adaptation run.

set -euo pipefail

REPO_ROOT=${OPENPI_REPO_ROOT:-/dodrio/scratch/projects/starting_2026_047/openpi}
cd "${REPO_ROOT}"
if [ "${OPENPI_SKIP_MODULES:-0}" != 1 ]; then
  read -r -a OPENPI_MODULE_LIST <<< "${OPENPI_GPU_MODULES:-cluster/dodrio/gpu_rome_a100_80_rhel9 CUDA}"
  module load "${OPENPI_MODULE_LIST[@]}"
fi

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
  read -r TARGET_SUITE TARGET_TASK_ID METHOD DATA_BUDGET SEED <<< "${GRID_ENTRY}"
fi

TARGET_SUITE=${TARGET_SUITE:?Set TARGET_SUITE or submit as a PBS array}
TARGET_TASK_ID=${TARGET_TASK_ID:?Set TARGET_TASK_ID or submit as a PBS array}
METHOD=${METHOD:?Set METHOD to full or lora}
DATA_BUDGET=${DATA_BUDGET:?Set DATA_BUDGET to 1, 5, 10, 25, or all_available}
SEED=${SEED:-0}
DELETE_TARGET_CHECKPOINT=${DELETE_TARGET_CHECKPOINT:-1}
DELETE_TARGET_CHECKPOINT_ON_FAILURE=${DELETE_TARGET_CHECKPOINT_ON_FAILURE:-1}
EVAL_MAX_ATTEMPTS=${EVAL_MAX_ATTEMPTS:-3}
if ! [[ "${EVAL_MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_MAX_ATTEMPTS must be a positive integer; got ${EVAL_MAX_ATTEMPTS}" >&2
  exit 2
fi
EVAL_MAX_CONCURRENT_PER_NODE=${EVAL_MAX_CONCURRENT_PER_NODE:-4}
if ! [[ "${EVAL_MAX_CONCURRENT_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_MAX_CONCURRENT_PER_NODE must be a positive integer; got ${EVAL_MAX_CONCURRENT_PER_NODE}" >&2
  exit 2
fi
LIBERO_VENV=${LIBERO_VENV:-${REPO_ROOT}/examples/libero/.venv}
JOB_TOKEN="${SLURM_JOB_ID:-${PBS_JOBID:-$$}}_${ARRAY_INDEX:-0}"
JOB_TOKEN=${JOB_TOKEN//[!a-zA-Z0-9_.-]/_}
JOB_TMPDIR=${JOB_TMPDIR:-${SLURM_TMPDIR:-${TMPDIR:-${VSC_SCRATCH_NODE:-/tmp}}}/${USER}/openpi_low_data_${JOB_TOKEN}}
DESCRIPTOR="${JOB_TMPDIR}/low_data_target_train_manifest.json"
mkdir -p "${JOB_TMPDIR}"

RESULT_DIR=$(uv run python - "${EXPERIMENT_CONFIG}" "${TARGET_SUITE}" "${TARGET_TASK_ID}" \
  "${METHOD}" "${DATA_BUDGET}" "${SEED}" <<'PY'
import sys
from openpi.training.low_data.experiment import load_experiment_config, target_result_dir

config = load_experiment_config(sys.argv[1])
print(target_result_dir(config, sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])))
PY
)
PERSISTENT_MANIFEST="${RESULT_DIR}/train_manifest.json"
RESULT_COMPLETE=$(uv run python - "${EXPERIMENT_CONFIG}" "${RESULT_DIR}" <<'PY'
import json
import pathlib
import sys

from openpi.training.low_data.experiment import load_experiment_config

config = load_experiment_config(sys.argv[1])
result_dir = pathlib.Path(sys.argv[2])
try:
    matches = (
        (result_dir / "tidy_results.jsonl").stat().st_size > 0
        and json.loads((result_dir / "eval_results.json").read_text()).get("evaluation_protocol")
        == config.evaluation.protocol_manifest()
    )
except (FileNotFoundError, json.JSONDecodeError):
    matches = False
print(int(matches))
PY
)
if [ "${RESULT_COMPLETE}" = 1 ]; then
  echo "Stage-B result already matches the configured trial count; skipping ${RESULT_DIR}"
  exit 0
fi

cleanup_failed_checkpoint() {
  status=$?
  if [ "${status}" -ne 0 ] && [ "${DELETE_TARGET_CHECKPOINT_ON_FAILURE}" = 1 ]; then
    uv run python - "${REPO_ROOT}" "${JOB_TMPDIR}" "${PERSISTENT_MANIFEST}" \
      "${RESULT_DIR}/resolved_train_config.json" <<'PY' || true
import json
import pathlib
import shutil
import sys

repo_root = pathlib.Path(sys.argv[1]).resolve()
job_tmpdir = pathlib.Path(sys.argv[2]).resolve()
manifest_path = pathlib.Path(sys.argv[3])
config_path = pathlib.Path(sys.argv[4])
run_dir = None
if manifest_path.exists():
    run_dir = pathlib.Path(json.loads(manifest_path.read_text())["checkpoint_run_dir"])
elif config_path.exists():
    config = json.loads(config_path.read_text())
    run_dir = pathlib.Path(config["checkpoint_base_dir"]) / config["name"] / config["exp_name"]
if run_dir is not None:
    run_dir = (repo_root / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
    allowed_roots = ((repo_root / "checkpoints").resolve(), job_tmpdir)
    if (
        any(root == run_dir or root in run_dir.parents for root in allowed_roots)
        and "targets" in run_dir.parts
        and run_dir.is_dir()
    ):
        print(f"Removing failed transient target checkpoint: {run_dir}")
        shutil.rmtree(run_dir)
PY
  fi
  exit "${status}"
}
trap cleanup_failed_checkpoint EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Array cells may share a host, so port 8000 is not safe. Derive a stable per-job port unless
# explicitly overridden. Include both the scheduler job ID and array index when available.
PORT_SEED="${SLURM_JOB_ID:-${PBS_JOBID:-$$}}${ARRAY_INDEX:-0}"
PORT_SEED=${PORT_SEED//[!0-9]/}
PORT=${PORT:-$((18000 + 10#${PORT_SEED:-1} % 20000))}

OPENPI_CACHE_ROOT=${OPENPI_CACHE_ROOT:-/dodrio/scratch/projects/starting_2026_047/cache}
export HF_HOME=${HF_HOME:-${OPENPI_CACHE_ROOT}/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${OPENPI_CACHE_ROOT}/openpi}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
# Do not derive MUJOCO_EGL_DEVICE_ID from SLURM_JOB_GPUS. NVIDIA's EGL device ordering is not
# guaranteed to match Slurm's physical GPU numbering. With no explicit override MuJoCo probes
# the EGL devices and selects the one made accessible by the job's device cgroup.
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
echo "CONFIG=${EXPERIMENT_CONFIG} TARGET=${TARGET_SUITE}:${TARGET_TASK_ID} METHOD=${METHOD} DATA_BUDGET=${DATA_BUDGET} SEED=${SEED} PORT=${PORT} DELETE=${DELETE_TARGET_CHECKPOINT} EVAL_MAX_ATTEMPTS=${EVAL_MAX_ATTEMPTS} EVAL_MAX_CONCURRENT_PER_NODE=${EVAL_MAX_CONCURRENT_PER_NODE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-auto}"
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
  echo "Reusing trained target checkpoint from ${PERSISTENT_MANIFEST}"
  DESCRIPTOR="${PERSISTENT_MANIFEST}"
else
  source .venv/bin/activate
  uv run scripts/low_data_train.py \
    --experiment-config "${EXPERIMENT_CONFIG}" \
    --stage target \
    --target-suite "${TARGET_SUITE}" \
    --target-task-id "${TARGET_TASK_ID}" \
    --method "${METHOD}" \
    --data-budget "${DATA_BUDGET}" \
    --seed "${SEED}" \
    --checkpoint-base-dir "${JOB_TMPDIR}/checkpoints" \
    --descriptor-out "${DESCRIPTOR}"
  deactivate
fi

DELETE_ARGS=()
if [ "${DELETE_TARGET_CHECKPOINT}" = 1 ]; then
  DELETE_ARGS+=(--delete-checkpoint-after-eval)
fi
source "${LIBERO_VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/libero:${PYTHONPATH:-}"

acquire_eval_slot() {
  # Sofia H200 nodes are stable with several independent trainers, but too many simultaneous EGL
  # readbacks can abort inside the NVIDIA driver. Keep training parallel while bounding only the
  # rendering phase. /tmp is node-local, so the semaphore is independent on every allocated node.
  local lock_dir="/tmp/${USER}/openpi_low_data_eval_locks"
  local candidate_fd slot
  mkdir -p "${lock_dir}"
  while true; do
    for ((slot = 0; slot < EVAL_MAX_CONCURRENT_PER_NODE; slot++)); do
      exec {candidate_fd}>"${lock_dir}/slot${slot}.lock"
      if flock -n "${candidate_fd}"; then
        EVAL_LOCK_FD=${candidate_fd}
        EVAL_LOCK_SLOT=${slot}
        echo "Acquired node-local evaluation slot ${slot}/${EVAL_MAX_CONCURRENT_PER_NODE}"
        return
      fi
      exec {candidate_fd}>&-
    done
    echo "Waiting for a node-local evaluation slot (${EVAL_MAX_CONCURRENT_PER_NODE} allowed)"
    sleep 5
  done
}

cleanup_orphaned_policy_server() {
  # A native MuJoCo/EGL abort bypasses Python's context-manager cleanup and can leave the policy
  # server alive. The port is unique to this array cell, so this pattern cannot affect peers.
  pkill -TERM -f "[s]cripts/serve_policy.py --port ${PORT}" 2>/dev/null || true
  sleep 2
  pkill -KILL -f "[s]cripts/serve_policy.py --port ${PORT}" 2>/dev/null || true
}

eval_status=1
acquire_eval_slot
for ((attempt = 1; attempt <= EVAL_MAX_ATTEMPTS; attempt++)); do
  echo "Starting evaluation attempt ${attempt}/${EVAL_MAX_ATTEMPTS}"
  if python scripts/low_data_eval.py \
    --experiment-config "${EXPERIMENT_CONFIG}" \
    --train-manifest "${DESCRIPTOR}" \
    --repo-root "${REPO_ROOT}" \
    --port "${PORT}" \
    --server-gpu 0 \
    "${DELETE_ARGS[@]}"; then
    eval_status=0
    break
  else
    eval_status=$?
  fi
  echo "Evaluation attempt ${attempt} failed with status ${eval_status}" >&2
  cleanup_orphaned_policy_server
done

if [ "${eval_status}" -ne 0 ]; then
  echo "Evaluation failed after ${EVAL_MAX_ATTEMPTS} attempts" >&2
  exit "${eval_status}"
fi
