#!/bin/bash
#PBS -l nodes=1:ppn=12:gpus=1
#PBS -l walltime=24:00:00
#PBS -A starting_2026_047

set -euo pipefail

module load cluster/dodrio/gpu_rome_a100_80_rhel9
module load CUDA

cd /dodrio/scratch/projects/starting_2026_047/openpi
source .venv/bin/activate

export HF_HOME=/dodrio/scratch/projects/starting_2026_047/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE=$HF_HOME
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "===== JOB INFO ====="
hostname
date
pwd
echo "PBS_JOBID=$PBS_JOBID"

echo "===== GPU INFO ====="
nvidia-smi

echo "===== ENV ====="
which python
python -V

echo "===== DATA CHECK ====="
ls ../dataset/peract_lerobot_train -al || true
ls assets/pi05_rlbench_lora -al || true

echo "===== START TRAIN ====="

uv run scripts/train.py pi05_rlbench_lora \
  --exp-name=pi05_rlbench_lora \
  --resume
