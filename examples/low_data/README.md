# Low-data post-pretraining adaptation

This pipeline reuses OpenPI's JAX trainer, LIBERO LeRobot dataloader, checkpoint format, policy
server, and rollout evaluator. It is independent of the sequential continual-learning benchmark.

## Final protocol freeze

The final protocol is defined by `configs/libero_main_18source_22target.json` (schema version 3).
Stage A creates one source-adapted checkpoint. Stage B is always:

```text
immutable Stage-A source checkpoint
  -> adapt to exactly one target task and one data budget
  -> evaluate
  -> finish
```

Every Stage-B run reloads the exact same Stage-A checkpoint. Target tasks are never trained
jointly, no adapted checkpoint is passed to another target, and source demonstrations are
unavailable to the Stage-B dataloader. Both LoRA and Full FT are supported. Guidance and replay
are outside this protocol.

The primary data budgets are complete trajectories:

```text
D1 ⊂ D5 ⊂ D10 ⊂ D25
```

For each `(target, subset_seed)`, the code creates one deterministic trajectory ordering and uses
prefixes. LoRA uses `1/5/10/25/all_available`; Full FT retains sparse anchors
`1/10/all_available`. The shared `1`, `10`, and `all_available` subsets are identical across LoRA
and Full FT, so method comparisons are paired. `all_available` records its actual trajectory count
and is never called "50 demos".

All adaptation conditions use the development-validated `effective_epochs=10.0`. There is no
minimum-step floor and no normal optimizer-step cap. The scientific optimizer budget is:

```text
global_batch_size =
    per_device_batch_size * world_size * gradient_accumulation_steps

calculated_optimizer_steps = ceil(
    effective_epochs * num_training_examples / global_batch_size
)
```

`adaptation.hard_max_steps` is `null` in the scientific config. If an emergency guard is supplied
and the calculated budget exceeds it, the run fails as a protocol violation; it is never silently
truncated. The existing trainer currently requires `gradient_accumulation_steps=1`.

Evaluation uses a distinct `rollout_horizons` setting:

```yaml
libero_spatial: 220
libero_object:  280
libero_goal:    300
libero_10:      520
```

Each target and each source-retention task uses the horizon of its own suite. Nonstandard values
are rejected unless `allow_nonstandard_rollout_horizons=true` explicitly marks a debug run.

## Frozen Split A

The task split is unchanged:

- Spatial source `[4, 2, 9, 7, 6, 8]`, target `[5, 1, 3, 0]`
- Object source `[4, 2, 9, 7, 6, 8]`, target `[5, 1, 3, 0]`
- Goal source `[4, 2, 9, 7, 6, 8]`, target `[5, 1, 3, 0]`
- LIBERO-10 source `[]`, target `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

This is one 18-source checkpoint and 22 independent targets. No Split B is defined. With two
LoRA's five data budgets, Full FT's three anchors, and seed 0, a complete future Stage-B grid is
`22 × (5 + 3) × 1 = 176` runs.

The final protocol writes to `checkpoints/low_data_final` and `results/low_data_final`, separated
from historical outputs. A future complete dependency chain can be submitted with:

```bash
MAX_CONCURRENT=2 scripts/submit_low_data_main.sh
```

One explicit Stage-B cell uses a data-budget label, not `NUM_DEMOS`:

```bash
qsub -v EXPERIMENT_CONFIG=examples/low_data/configs/libero_main_18source_22target.json,\
TARGET_SUITE=libero_goal,TARGET_TASK_ID=0,METHOD=lora,DATA_BUDGET=all_available,SEED=0 \
  scripts/job_low_data_stage_b.sh
```

## Auditable future results

Future train manifests and tidy results record:

```text
source_split_id, source_checkpoint
target_suite, target_task_id, method
requested_data_budget, actual_num_demos, total_available_demos
selected_trajectory_ids, subset_seed
effective_epochs, num_training_examples, global_batch_size
calculated_optimizer_steps, actual_optimizer_steps, samples_seen
rollout_horizon
zero_shot_target_success, adapted_target_success, target_gain
source_success_before, source_success_after, source_forgetting, source_retention
```

Training asserts that `actual_optimizer_steps == calculated_optimizer_steps`. Target checkpoints
remain transient; manifests and evaluations persist.

## Historical exploratory experiments

Experiments produced before this freeze are exploratory historical results. In particular, the
8-source/2-target pilots and `libero_main_18s22t_v0` used capped-effective-epochs behavior,
`min_steps`, a 500-step truncation, sparse Full-FT anchors, a nominal 50-demo condition, and/or a
shared 280-step evaluation horizon. Their result files are retained unchanged and must not be
mixed with schema-v3 results.

The historical configs remain in `configs/libero_spatial_8source_2target.json` and
`configs/libero_goal_8source_2target.json` for auditability, but schema versions 1 and 2 are rejected
by the future-run loader so they cannot accidentally launch under the frozen protocol. Existing
aggregation paths retain compatibility with historical result rows.
