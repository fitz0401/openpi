#!/bin/bash
# Prepare a shared-filesystem OpenPI + LIBERO installation and prefetch the formal-run inputs.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CLUSTER=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --cluster)
      CLUSTER=${2:?--cluster requires sofia or leonardo}
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--cluster sofia|leonardo]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
if [ -n "${CLUSTER}" ]; then
  # shellcheck source=scripts/low_data_cluster_profiles.sh
  source "${SCRIPT_DIR}/low_data_cluster_profiles.sh"
  load_low_data_cluster_profile "${CLUSTER}"
fi

if [ "${OPENPI_SKIP_MODULES:-0}" != 1 ] && [ -n "${OPENPI_BOOTSTRAP_MODULES:-}" ]; then
  read -r -a OPENPI_BOOTSTRAP_MODULE_LIST <<< "${OPENPI_BOOTSTRAP_MODULES}"
  module load "${OPENPI_BOOTSTRAP_MODULE_LIST[@]}"
fi

REPO_ROOT=${OPENPI_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DEFAULT_CACHE_PARENT=${SCRATCH:-${HOME}/.cache}
OPENPI_CACHE_ROOT=${OPENPI_CACHE_ROOT:-${DEFAULT_CACHE_PARENT}/openpi_low_data_cache}
LIBERO_VENV=${LIBERO_VENV:-${REPO_ROOT}/examples/libero/.venv}
PREFETCH_CHECKPOINT=${PREFETCH_CHECKPOINT:-1}
PREFETCH_DATASET=${PREFETCH_DATASET:-1}

for value in "${REPO_ROOT}" "${OPENPI_CACHE_ROOT}" "${LIBERO_VENV}"; do
  if [[ "${value}" =~ [[:space:]] ]]; then
    echo "Paths containing whitespace are unsupported by the batch orchestration: ${value}" >&2
    exit 2
  fi
done

if ! command -v uv >/dev/null 2>&1 && [ "${OPENPI_INSTALL_UV:-0}" = 1 ]; then
  UV_INSTALL_DIR=${UV_INSTALL_DIR:-${OPENPI_CACHE_ROOT}/tools/bin}
  mkdir -p "${UV_INSTALL_DIR}"
  echo "===== UV ====="
  echo "Installing uv into ${UV_INSTALL_DIR}"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${UV_INSTALL_DIR}" sh
  export PATH="${UV_INSTALL_DIR}:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Re-run with OPENPI_INSTALL_UV=1 or install it first." >&2
  exit 2
fi

cd "${REPO_ROOT}"
mkdir -p "${OPENPI_CACHE_ROOT}/huggingface" "${OPENPI_CACHE_ROOT}/openpi" .cluster
if [ -n "${OPENPI_STORAGE_ROOT:-}" ]; then
  mkdir -p "${OPENPI_STORAGE_ROOT}/checkpoints" "${OPENPI_STORAGE_ROOT}/results" \
    "${OPENPI_JOB_TMP_ROOT:-${OPENPI_STORAGE_ROOT}/job_tmp}"
  for name in checkpoints results; do
    destination="${OPENPI_STORAGE_ROOT}/${name}"
    if [ -L "${REPO_ROOT}/${name}" ]; then
      current=$(readlink -f "${REPO_ROOT}/${name}")
      if [ "${current}" != "$(readlink -f "${destination}")" ]; then
        echo "${REPO_ROOT}/${name} points to ${current}, expected ${destination}" >&2
        exit 2
      fi
    elif [ -e "${REPO_ROOT}/${name}" ]; then
      if [ -n "$(find "${REPO_ROOT}/${name}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "Refusing to replace non-empty ${REPO_ROOT}/${name}; move it manually first." >&2
        exit 2
      fi
      rmdir "${REPO_ROOT}/${name}"
      ln -s "${destination}" "${REPO_ROOT}/${name}"
    else
      ln -s "${destination}" "${REPO_ROOT}/${name}"
    fi
  done
fi

export OPENPI_REPO_ROOT="${REPO_ROOT}"
export OPENPI_CACHE_ROOT
export HF_HOME=${HF_HOME:-${OPENPI_CACHE_ROOT}/huggingface}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-0}
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${OPENPI_CACHE_ROOT}/openpi}
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}}

echo "===== REPOSITORY ====="
git submodule update --init --recursive
SYNC_ARGS=(--frozen)
if [ "${OPENPI_SKIP_RERUN_SDK:-0}" = 1 ]; then
  SYNC_ARGS+=(--no-install-package rerun-sdk)
  # Prevent later `uv run` calls from trying to restore the deliberately
  # omitted, unused visualization dependency.
  export UV_NO_SYNC=1
fi
if [ -n "${OPENPI_PROJECT_PYTHON:-}" ]; then
  PROJECT_PYTHON=${OPENPI_PROJECT_PYTHON}
else
  PROJECT_PYTHON=$(command -v python3)
fi
GIT_LFS_SKIP_SMUDGE=1 uv sync --python "${PROJECT_PYTHON}" "${SYNC_ARGS[@]}"
GIT_LFS_SKIP_SMUDGE=1 uv pip install --no-deps -e .

echo "===== LIBERO CLIENT ENVIRONMENT ====="
uv venv --python 3.8 --allow-existing "${LIBERO_VENV}"
uv pip sync --python "${LIBERO_VENV}/bin/python" \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install --python "${LIBERO_VENV}/bin/python" -e packages/openpi-client
uv pip install --python "${LIBERO_VENV}/bin/python" -e third_party/libero

NORM_STATS=assets/pi05_libero_low_data_full/libero_low_data/norm_stats.json
if [ ! -s "${NORM_STATS}" ]; then
  echo "Missing versioned formal-protocol norm stats: ${REPO_ROOT}/${NORM_STATS}" >&2
  exit 2
fi

if [ "${PREFETCH_CHECKPOINT}" = 1 ]; then
  echo "===== PI0.5 RUNTIME ASSETS ====="
  uv run python - <<'PY'
from openpi.shared.download import maybe_download

print(maybe_download("gs://openpi-assets/checkpoints/pi05_base"))
print(maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}))
PY
fi

if [ "${PREFETCH_DATASET}" = 1 ]; then
  if [ "${REQUIRE_HF_AUTH:-0}" = 1 ] && [ -z "${HF_TOKEN:-}" ] && [ ! -s "${HF_TOKEN_PATH}" ]; then
    echo "Hugging Face authentication is required before dataset prefetch." >&2
    echo "Set HF_HOME/HF_TOKEN_PATH, then run: uv run huggingface-cli login" >&2
    exit 2
  fi
  echo "===== LIBERO LEROBOT DATASET ====="
  uv run python - <<'PY'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("physical-intelligence/libero", download_videos=True)
print(f"dataset_root={dataset.root}")
print(f"num_frames={len(dataset)}")
print(f"num_episodes={len(dataset.meta.episodes)}")
PY
fi

ENV_FILE=${REPO_ROOT}/.cluster/low_data.env
{
  printf 'export OPENPI_REPO_ROOT=%q\n' "${REPO_ROOT}"
  printf 'export OPENPI_CACHE_ROOT=%q\n' "${OPENPI_CACHE_ROOT}"
  printf 'export HF_HOME=%q\n' "${HF_HOME}"
  printf 'export HF_TOKEN_PATH=%q\n' "${HF_TOKEN_PATH}"
  printf 'export HF_HUB_DISABLE_XET=%q\n' "${HF_HUB_DISABLE_XET}"
  printf 'export OPENPI_DATA_HOME=%q\n' "${OPENPI_DATA_HOME}"
  printf 'export LIBERO_VENV=%q\n' "${LIBERO_VENV}"
  if [ -n "${OPENPI_CLUSTER:-}" ]; then printf 'export OPENPI_CLUSTER=%q\n' "${OPENPI_CLUSTER}"; fi
  if [ -n "${OPENPI_STORAGE_ROOT:-}" ]; then printf 'export OPENPI_STORAGE_ROOT=%q\n' "${OPENPI_STORAGE_ROOT}"; fi
  if [ -n "${OPENPI_JOB_TMP_ROOT:-}" ]; then printf 'export OPENPI_JOB_TMP_ROOT=%q\n' "${OPENPI_JOB_TMP_ROOT}"; fi
  if [ -n "${OPENPI_SKIP_RERUN_SDK:-}" ]; then printf 'export OPENPI_SKIP_RERUN_SDK=%q\n' "${OPENPI_SKIP_RERUN_SDK}"; fi
  if [ -n "${UV_NO_SYNC:-}" ]; then printf 'export UV_NO_SYNC=%q\n' "${UV_NO_SYNC}"; fi
  printf 'export PATH=%q\n' "${PATH}"
  if [ -n "${SLURM_ACCOUNT:-}" ]; then printf 'export SLURM_ACCOUNT=%q\n' "${SLURM_ACCOUNT}"; fi
  if [ -n "${SLURM_PARTITION:-}" ]; then printf 'export SLURM_PARTITION=%q\n' "${SLURM_PARTITION}"; fi
  if [ -n "${SLURM_CPU_PARTITION:-}" ]; then printf 'export SLURM_CPU_PARTITION=%q\n' "${SLURM_CPU_PARTITION}"; fi
  if [ -n "${SLURM_CPU_ACCOUNT:-}" ]; then printf 'export SLURM_CPU_ACCOUNT=%q\n' "${SLURM_CPU_ACCOUNT}"; fi
  if [ -n "${SLURM_QOS:-}" ]; then printf 'export SLURM_QOS=%q\n' "${SLURM_QOS}"; fi
  if [ -n "${OPENPI_GPU_MODULES:-}" ]; then printf 'export OPENPI_GPU_MODULES=%q\n' "${OPENPI_GPU_MODULES}"; fi
  if [ -n "${OPENPI_BOOTSTRAP_MODULES:-}" ]; then printf 'export OPENPI_BOOTSTRAP_MODULES=%q\n' "${OPENPI_BOOTSTRAP_MODULES}"; fi
  if [ -n "${OPENPI_SKIP_MODULES:-}" ]; then printf 'export OPENPI_SKIP_MODULES=%q\n' "${OPENPI_SKIP_MODULES}"; fi
  if [ -n "${SLURM_GPU_GRES:-}" ]; then printf 'export SLURM_GPU_GRES=%q\n' "${SLURM_GPU_GRES}"; fi
  if [ -n "${SLURM_GPU_REQUEST_MODE:-}" ]; then printf 'export SLURM_GPU_REQUEST_MODE=%q\n' "${SLURM_GPU_REQUEST_MODE}"; fi
  if [ -n "${SLURM_GPUS_PER_NODE:-}" ]; then printf 'export SLURM_GPUS_PER_NODE=%q\n' "${SLURM_GPUS_PER_NODE}"; fi
  if [ -n "${SLURM_CPUS_PER_GPU:-}" ]; then printf 'export SLURM_CPUS_PER_GPU=%q\n' "${SLURM_CPUS_PER_GPU}"; fi
  if [ -n "${SLURM_STAGE_A_GPU_REQUEST_MODE:-}" ]; then printf 'export SLURM_STAGE_A_GPU_REQUEST_MODE=%q\n' "${SLURM_STAGE_A_GPU_REQUEST_MODE}"; fi
  if [ -n "${SLURM_STAGE_A_GPU_GRES:-}" ]; then printf 'export SLURM_STAGE_A_GPU_GRES=%q\n' "${SLURM_STAGE_A_GPU_GRES}"; fi
  if [ -n "${SLURM_STAGE_A_GPUS_PER_NODE:-}" ]; then printf 'export SLURM_STAGE_A_GPUS_PER_NODE=%q\n' "${SLURM_STAGE_A_GPUS_PER_NODE}"; fi
  if [ -n "${SLURM_STAGE_A_CPUS_PER_TASK:-}" ]; then printf 'export SLURM_STAGE_A_CPUS_PER_TASK=%q\n' "${SLURM_STAGE_A_CPUS_PER_TASK}"; fi
  if [ -n "${OPENPI_SOURCE_FSDP_DEVICES:-}" ]; then printf 'export OPENPI_SOURCE_FSDP_DEVICES=%q\n' "${OPENPI_SOURCE_FSDP_DEVICES}"; fi
  if [ -n "${EVAL_MUJOCO_EGL_DEVICE_ID:-}" ]; then printf 'export EVAL_MUJOCO_EGL_DEVICE_ID=%q\n' "${EVAL_MUJOCO_EGL_DEVICE_ID}"; fi
  if [ -n "${SLURM_GPU_MEMORY:-}" ]; then printf 'export SLURM_GPU_MEMORY=%q\n' "${SLURM_GPU_MEMORY}"; fi
  if [ -n "${SLURM_FINALIZE_MEMORY:-}" ]; then printf 'export SLURM_FINALIZE_MEMORY=%q\n' "${SLURM_FINALIZE_MEMORY}"; fi
  if [ -n "${SLURM_STAGE_A_TIME:-}" ]; then printf 'export SLURM_STAGE_A_TIME=%q\n' "${SLURM_STAGE_A_TIME}"; fi
  if [ -n "${SLURM_STAGE_B_TIME:-}" ]; then printf 'export SLURM_STAGE_B_TIME=%q\n' "${SLURM_STAGE_B_TIME}"; fi
} > "${ENV_FILE}"

echo "===== READY ====="
echo "Environment file: ${ENV_FILE}"
echo "Before submission: source ${ENV_FILE}"
du -sh "${OPENPI_CACHE_ROOT}" "${LIBERO_VENV}" 2>/dev/null || true
