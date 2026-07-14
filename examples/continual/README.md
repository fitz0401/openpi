# Continual Finetuning Benchmark — pi0.5 on LIBERO-Object

Stage-1 benchmark that characterizes how pretrained **π₀.₅** behaves under different expert-demo
budgets when adapted **sequentially** to downstream tasks. **No new algorithm, no architecture
change** — it only measures behavior (success matrix, forgetting, transfer, recovery curves) so a
later stage can add inference-time guidance.

## What it does

- **Tasks:** first `N=5` tasks of **LIBERO-Object** (10 available, ~45 demos each), learned T1→T2→…→T5.
- **Demo budgets:** {1, 5, 10, 25, 50, 100}; budgets above the ~45 available demos use *all* of them.
- **Seeds:** {0, 1, 2}; reported metrics are averaged across seeds.
- **Sequential finetuning without weight reset:** task K initializes from task K-1's checkpoint via
  the existing `CheckpointWeightLoader`; a fresh optimizer is used per task. Task 1 starts from the
  pretrained pi0.5 checkpoint. The standard training loop (`scripts/train.py`) is reused unchanged.
- **Reproducible subsampling:** `data.subsample_spec` selects exactly `budget` episodes of one task
  for a given seed; the chosen episode indices are stored as `*_sampled_indices.json`.

## Components

| Piece | Env | What |
|---|---|---|
| `src/openpi/training/continual/subsample.py` | uv | reproducible per-task demo selection (episode whitelist) |
| `src/openpi/training/continual/metrics.py` | uv | success-matrix metrics (acc / forgetting / BWT / FWT) |
| `scripts/continual_finetune.py` | uv | orchestrates the sequential finetuning for one (budget, seed) |
| `scripts/continual_eval.py` | **LIBERO venv** | runs rollouts, fills success matrix + learning curves |
| `scripts/continual_metrics.py` | uv | computes metrics.json + cross-seed summary from the matrix |
| `pi05_libero_object_continual` (config) | uv | base template (filled per stage by the orchestrator) |

The config `pi05_libero_object_continual` trains on `physical-intelligence/libero` filtered to the
Object task strings (canonical list in `continual/libero_object_tasks.py`).

## Running (cluster, 48h-per-job)

```bash
# 1) Shared normalization stats (once).
qsub scripts/job_continual_norm.sh

# 2) Sequential finetuning — one job per budget (validation slice = 1, 10, 50 at seed 0).
qsub -v BUDGET=1  scripts/job_continual_train.sh
qsub -v BUDGET=10 scripts/job_continual_train.sh
qsub -v BUDGET=50 scripts/job_continual_train.sh

# 3) Evaluation + metrics — mirror the train jobs. Needs the LIBERO venv (examples/libero/.venv);
#    see the one-time setup instructions in scripts/job_continual_eval.sh if not built yet.
qsub -v BUDGET=1  scripts/job_continual_eval.sh
qsub -v BUDGET=10 scripts/job_continual_eval.sh
qsub -v BUDGET=50 scripts/job_continual_eval.sh
```

Overrides via `qsub -v`: `SEED`, `RUN_NAME`, `N_TASKS`, `STEPS`, `SAVE`, `CHECKPOINT_MODE` (train);
`NUM_TRIALS`, `NUM_TRIALS_LC`, `LIBERO_VENV`, `MUJOCO_GL` (eval).

`CHECKPOINT_MODE=eval` is the storage-efficient default: it retains only each stage's final
`params/assets`, skips periodic checkpoint writes, and removes optimizer state after the stage
completes. Use
`CHECKPOINT_MODE=learning_curves` to also retain interval inference checkpoints, or
`CHECKPOINT_MODE=resumable` only when optimizer-state resume is required. For pi0.5, the resumable
mode can consume hundreds of GiB per `(budget, seed)` run.

### Scaling to the full grid
Submit the train/eval jobs for every `BUDGET` in {1,5,10,25,50,100} and every `SEED` in {0,1,2}.
Then aggregate all seeds of a budget:

```bash
uv run scripts/continual_metrics.py \
  --run-dirs checkpoints/pi05_libero_object_continual/<run>/budget10/seed0 \
             checkpoints/pi05_libero_object_continual/<run>/budget10/seed1 \
             checkpoints/pi05_libero_object_continual/<run>/budget10/seed2 \
  --summary-out results/budget10_summary.csv
```

## Output layout

```
checkpoints/pi05_libero_object_continual/<run>/budget{b}/seed{s}/
  stage{k}_task{tid}/<step>/{params,assets}       # default: final inference checkpoint only
  manifest.json                  # task order, checkpoint dirs, learning-curve steps
  stage*_sampled_indices.json    # reproducible demo selection
  success_matrix.csv             # rows = train stage (0 = optional baseline), cols = task
  learning_curves.csv            # (stage, task_id, step, success_rate)
  eval_results.json              # raw results
  metrics.json                   # avg accuracy / forgetting / BWT / FWT / newest-task success
results/*_summary.csv            # mean/std across seeds
```

## Downloading and plotting results

Download one run's JSON/CSV outputs, manifest, sampled-demo indices, and summary from Dodrio, then
generate a dashboard and individual plots locally:

```bash
scripts/download_continual_results.sh --budget 50 --seed 0 --run-name slice_v0
```

The default destination is `experiments/budget{budget}_seed{seed}`. Override the login or remote
repository with `REMOTE_HOST` and `REMOTE_REPO`. To only replot files already downloaded:

```bash
uv run python scripts/plot_continual_results.py \
  --run-dir experiments/budget50_seed0
```

## Metrics (from the success matrix `R[i][j]`)

Rows `i = 0..N` (0 = pretrained baseline), cols `j = 1..N` (tasks in training order). The full row
is evaluated after every stage, so both backward and forward transfer are available.

- **Average accuracy** = mean over tasks of final-stage success.
- **Average forgetting** = mean over earlier tasks of (peak success while/after learned − final success).
- **Backward transfer (BWT)** = mean over earlier tasks of (final − success right after that task).
- **Forward transfer (FWT)** = mean over later tasks of (zero-shot-before-training − pretrained baseline).
  Requires the optional stage-0 baseline (`--baseline-config/--baseline-dir` in `continual_eval.py`); else `None`.
- **Newest-task success** = final-stage success on the last-trained task.

## Notes

- Norm stats are computed once and shared (avoids unstable few-shot statistics).
- Default `CHECKPOINT_MODE=eval` keeps only final inference checkpoints. The optional
  `learning_curves` mode retains intermediate params without optimizer state.
- EMA is disabled so the *actually trained* weights are evaluated.
- `serve_policy.py` now respects a pre-set `CUDA_VISIBLE_DEVICES` (the eval job exports `0`).
- LIBERO sim runs headless via `MUJOCO_GL=egl`; switch to `osmesa`/`glx` if you hit EGL errors.
