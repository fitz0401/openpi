#!/bin/bash
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l walltime=05:00:00
#PBS -A starting_2026_047

cd /dodrio/scratch/projects/starting_2026_047/openpi

module load cluster/dodrio/gpu_rome_a100_80_rhel9
module load CUDA

source .venv/bin/activate

export HF_HOME=/dodrio/scratch/projects/starting_2026_047/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_DOWNLOAD_THREADS=2
nvidia-smi
uv run scripts/compute_norm_stats.py --config-name pi05_rlbench_lora