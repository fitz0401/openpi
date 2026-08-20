# Progress Regression Lite v3

This is the paper's lightweight source-to-target behavioral-compatibility probe. It is standalone:
it never changes Pi0.5, the adaptation dataloader, or historical FT/LoRA results.

For every successful source trajectory of length `T`, frame `t` receives the training label
`t / (T - 1)`. The probe sees RGB and the task instruction only. Frozen CLIP ViT-B/32 image/text
features are projected and fused by a small shared MLP with a sigmoid scalar output. Actions,
frame indices, progress labels, proprioception, temporal models, and task-specific heads are not
model inputs.

At evaluation time the source-trained probe is frozen. Per-demonstration R² against normalized
target demonstration progress is the primary compatibility score; MAE, Spearman rho, temporal
pair accuracy, and wrong-language R² are diagnostics.

## Local dataset location

The maintained local launchers read the LIBERO LeRobot dataset from:

```text
/mnt/data/ze/datasets/lerobot/physical-intelligence/libero
```

Both training and evaluation pass this location through `OPENPI_LIBERO_DATA_ROOT`; the resolved
probe config records the resulting absolute path. Generated Hugging Face Arrow data is also kept
off the system disk at `/mnt/data/ze/cache/huggingface/datasets`. To use another machine or mount,
override either location without editing the experiment configs:

```bash
OPENPI_LIBERO_DATA_ROOT=/path/to/physical-intelligence/libero \
OPENPI_HF_DATASETS_CACHE=/path/to/huggingface/datasets \
  GPU_INDEX=1 scripts/train_progress_regression_probe.sh "$CONFIG"
```

## Source compositions

The V3 matrix reuses the frozen Split-A manifest and includes:

- Spatial-only, Object-only, Goal-only;
- Spatial+Object, Spatial+Goal, Object+Goal;
- all 18 controlled source tasks.

All probes evaluate the same controlled Spatial/Object/Goal targets. The all-source config also
contains LIBERO-10 for diagnostic continuity; it is not part of the paper's primary 12-target
compatibility analysis.

Run the complete source-composition matrix sequentially on local GPU 1:

```bash
GPU_INDEX=1 scripts/run_progress_regression_v3_matrix.sh all
```

Or train/evaluate one probe:

```bash
CONFIG=examples/progress_probe/configs/progress_regression_v3_all_source.json
GPU_INDEX=1 scripts/train_progress_regression_probe.sh "$CONFIG"
GPU_INDEX=1 scripts/eval_progress_regression_probe.sh "$CONFIG"
```

Aggregate already completed evaluations without training:

```bash
scripts/run_progress_regression_v3_matrix.sh aggregate
```

Outputs are written under `results/progress_probe/progress_regression_lite_v3/` and include the
resolved config, source/target manifests, curves, lightweight checkpoint without CLIP weights,
per-demo/per-task/per-suite metrics, and source-to-target compatibility matrix.

The smoke config exercises one source task, one target task, and two optimizer steps:

```bash
CONFIG=examples/progress_probe/configs/progress_regression_v3_smoke.json
GPU_INDEX=1 scripts/train_progress_regression_probe.sh "$CONFIG"
GPU_INDEX=1 scripts/eval_progress_regression_probe.sh "$CONFIG"
```

Historical ranking-v1 and proprio-v2 experiments predate the paper protocol and are intentionally
not part of the maintained launch surface. Their ignored local results remain untouched.
