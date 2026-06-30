#!/bin/bash
#PBS -l nodes=1:ppn=24:gpus=4
#PBS -l walltime=48:00:00
#PBS -A starting_2026_047

# Usage:
#   qsub scripts/job_train.sh                           # default: pi05_rlbench_lora
#   qsub -v CONFIG=pi05_rlbench scripts/job_train.sh    # full fine-tune

set -euo pipefail

CONFIG=${CONFIG:-pi05_rlbench_lora}
EXP_NAME=${EXP_NAME:-${CONFIG}}

module load cluster/dodrio/gpu_rome_a100_80_rhel9
module load CUDA

cd /dodrio/scratch/projects/starting_2026_047/openpi
source .venv/bin/activate

export HF_HOME=/dodrio/scratch/projects/starting_2026_047/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE=$HF_HOME
# Redirect the openpi checkpoint download cache (gs://openpi-assets/...) to project scratch;
# it defaults to ~/.cache/openpi, which sits on the quota-limited user home and truncates
# large checkpoint downloads (e.g. pi05_base, ~11.6 GiB) mid-transfer.
export OPENPI_DATA_HOME=/dodrio/scratch/projects/starting_2026_047/cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "===== JOB INFO ====="
hostname
date
pwd
echo "PBS_JOBID=${PBS_JOBID:-local}"
echo "CONFIG=${CONFIG}"
echo "EXP_NAME=${EXP_NAME}"

echo "===== GPU INFO ====="
nvidia-smi

echo "===== ENV ====="
which python
python -V

echo "===== DATA CHECK ====="
ls ../dataset/peract_lerobot_train -al || true
ls assets/${CONFIG} -al || true

echo "===== START TRAIN ====="

uv run scripts/train.py ${CONFIG} \
  --exp-name=${EXP_NAME} \
  --resume
