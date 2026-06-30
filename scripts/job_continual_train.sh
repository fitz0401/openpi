#!/bin/bash
#PBS -l nodes=1:ppn=12:gpus=1
#PBS -l walltime=48:00:00
#PBS -A starting_2026_047

# Continual (sequential) finetuning for ONE (budget, seed) over the first N LIBERO-Object tasks.
# Reuses the standard training loop per task; task K initializes from task K-1's checkpoint.
#
# Usage (run one per budget; validation slice = budgets 1, 10, 50 at seed 0):
#   qsub -v BUDGET=1  scripts/job_continual_train.sh
#   qsub -v BUDGET=10 scripts/job_continual_train.sh
#   qsub -v BUDGET=50 scripts/job_continual_train.sh
# Optional overrides: SEED, RUN_NAME, N_TASKS, STEPS, SAVE, CHECKPOINT_MODE.

set -euo pipefail

cd /dodrio/scratch/projects/starting_2026_047/openpi

module load cluster/dodrio/gpu_rome_a100_80_rhel9
module load CUDA
source .venv/bin/activate

export HF_HOME=/dodrio/scratch/projects/starting_2026_047/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_CACHE=$HF_HOME
# Redirect the openpi checkpoint download cache (gs://openpi-assets/...) to project scratch;
# it defaults to ~/.cache/openpi, which sits on the quota-limited user home and truncates
# large checkpoint downloads (e.g. pi05_base, ~11.6 GiB) mid-transfer.
export OPENPI_DATA_HOME=/dodrio/scratch/projects/starting_2026_047/cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

BUDGET=${BUDGET:-10}
SEED=${SEED:-0}
RUN_NAME=${RUN_NAME:-slice_v0}
N_TASKS=${N_TASKS:-5}
# eval (default): final params only; learning_curves: interval params; resumable: full train state.
CHECKPOINT_MODE=${CHECKPOINT_MODE:-eval}

# Per-budget defaults for steps / checkpoint (= learning-curve) interval. Few-shot: more demos -> more steps.
if [ -z "${STEPS:-}" ]; then
  case "${BUDGET}" in
    1)  STEPS=200 ;;
    5)  STEPS=400 ;;
    10) STEPS=600 ;;
    25) STEPS=1000 ;;
    50) STEPS=1500 ;;
    100) STEPS=2000 ;;
    *)  STEPS=800 ;;
  esac
fi
SAVE=${SAVE:-$(( STEPS / 4 ))}

echo "===== CONTINUAL TRAIN ====="
hostname; date
echo "RUN_NAME=${RUN_NAME} BUDGET=${BUDGET} SEED=${SEED} N_TASKS=${N_TASKS} STEPS=${STEPS} SAVE=${SAVE} CHECKPOINT_MODE=${CHECKPOINT_MODE}"
nvidia-smi

uv run scripts/continual_finetune.py \
  --run-name "${RUN_NAME}" \
  --budget "${BUDGET}" \
  --seed "${SEED}" \
  --n-tasks "${N_TASKS}" \
  --num-train-steps "${STEPS}" \
  --save-interval "${SAVE}" \
  --checkpoint-mode "${CHECKPOINT_MODE}"
