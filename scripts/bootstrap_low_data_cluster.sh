#!/bin/bash
# Prepare a shared-filesystem OpenPI + LIBERO installation and prefetch the formal-run inputs.

set -euo pipefail

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

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 2
fi

cd "${REPO_ROOT}"
mkdir -p "${OPENPI_CACHE_ROOT}/huggingface" "${OPENPI_CACHE_ROOT}/openpi" .cluster

export OPENPI_REPO_ROOT="${REPO_ROOT}"
export OPENPI_CACHE_ROOT
export HF_HOME=${HF_HOME:-${OPENPI_CACHE_ROOT}/huggingface}
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${OPENPI_CACHE_ROOT}/openpi}
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}}

echo "===== REPOSITORY ====="
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

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
  echo "===== PI0.5 BASE CHECKPOINT ====="
  uv run python - <<'PY'
from openpi.shared.download import maybe_download

print(maybe_download("gs://openpi-assets/checkpoints/pi05_base"))
PY
fi

if [ "${PREFETCH_DATASET}" = 1 ]; then
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
  printf 'export OPENPI_DATA_HOME=%q\n' "${OPENPI_DATA_HOME}"
  printf 'export LIBERO_VENV=%q\n' "${LIBERO_VENV}"
  if [ -n "${SLURM_ACCOUNT:-}" ]; then printf 'export SLURM_ACCOUNT=%q\n' "${SLURM_ACCOUNT}"; fi
  if [ -n "${SLURM_PARTITION:-}" ]; then printf 'export SLURM_PARTITION=%q\n' "${SLURM_PARTITION}"; fi
  if [ -n "${SLURM_CPU_PARTITION:-}" ]; then printf 'export SLURM_CPU_PARTITION=%q\n' "${SLURM_CPU_PARTITION}"; fi
  if [ -n "${OPENPI_GPU_MODULES:-}" ]; then printf 'export OPENPI_GPU_MODULES=%q\n' "${OPENPI_GPU_MODULES}"; fi
  if [ -n "${OPENPI_SKIP_MODULES:-}" ]; then printf 'export OPENPI_SKIP_MODULES=%q\n' "${OPENPI_SKIP_MODULES}"; fi
  if [ -n "${SLURM_GPU_GRES:-}" ]; then printf 'export SLURM_GPU_GRES=%q\n' "${SLURM_GPU_GRES}"; fi
  if [ -n "${SLURM_CPUS_PER_GPU:-}" ]; then printf 'export SLURM_CPUS_PER_GPU=%q\n' "${SLURM_CPUS_PER_GPU}"; fi
  if [ -n "${SLURM_GPU_MEMORY:-}" ]; then printf 'export SLURM_GPU_MEMORY=%q\n' "${SLURM_GPU_MEMORY}"; fi
} > "${ENV_FILE}"

echo "===== READY ====="
echo "Environment file: ${ENV_FILE}"
echo "Before submission: source ${ENV_FILE}"
du -sh "${OPENPI_CACHE_ROOT}" "${LIBERO_VENV}" 2>/dev/null || true
