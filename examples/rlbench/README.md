# RLBench Evaluation Guide

Evaluate a fine-tuned π₀.₅ LoRA model on RLBench tasks using a server/client architecture:

- **Policy server** runs in the openpi `uv` environment (GPU node, handles JAX inference).
- **Evaluation client** runs in the conda environment that has RLBench installed (manages CoppeliaSim / PyRep).

---

## Prerequisites

### 1. Trained checkpoint

Fine-tune using the `pi05_rlbench_lora` config. Checkpoints land at:
```
checkpoints/pi05_rlbench_lora/<exp_name>/<step>/
```

### 2. RLBench in the conda environment

```bash
conda activate guide   # or whichever env has RLBench
pip install rlbench
```

### 3. openpi-client

```bash
pip install openpi-client   # provides WebsocketClientPolicy and image_tools
```

---

## Running evaluation

### Step 1 — start the policy server (openpi uv env)

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_rlbench_lora \
    --policy.dir checkpoints/pi05_rlbench_lora/<exp_name>/<step>

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_rlbench_lora \
    --policy.dir checkpoints/pi05_rlbench_lora/4999
```

The server listens on `0.0.0.0:8000` by default. Override with `--port <N>`.

### Step 2 — run the evaluation client (conda env with RLBench)

```bash
conda activate guide
python examples/rlbench/main.py \
    --task_names push_button open_drawer \
    --num_trials_per_task 10


nohup sudo X &
export DISPLAY=:0.0
export DISPLAY=:0.1 # second GPU
```
For running in headless mode, ref: https://github.com/MohitShridhar/RLBench/tree/peract#running-headless

If the server is on a different machine, pass `--host <server_ip>`.

---

## Configuration reference

| Argument | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Policy server host |
| `--port` | `8000` | Policy server port |
| `--task_names` | `push_button` | Space-separated list of RLBench task names |
| `--num_trials_per_task` | `10` | Episodes per task |
| `--max_steps_per_episode` | `500` | Step budget per episode |
| `--replan_steps` | `1` | Query policy every N env steps |
| `--resize_size` | `224` | Image resize target (px) |
| `--dataset_root` | `None` | Path to peract dataset (e.g. `/home/ze/data/peract`) |
| `--use_demos` | `False` | Load initial states from stored peract demos |
| `--video_out_path` | `data/rlbench/videos` | Directory for episode videos |
| `--seed` | `7` | Random seed |

---

## Multi-task evaluation

Evaluate all 18 peract tasks (import the constant from the script):

```bash
python examples/rlbench/main.py \
    --task_names \
        close_jar insert_onto_square_peg light_bulb_in meat_off_grill \
        open_drawer place_cups place_shape_in_sorter push_buttons \
        put_groceries_in_cupboard put_item_in_drawer put_money_in_safe \
        reach_and_drag slide_block_to_color_target stack_blocks stack_cups \
        turn_tap place_wine_at_rack_location sweep_to_dustpan_of_size \
    --num_trials_per_task 25
```

---

## Demo-based evaluation (peract test split)

If you have the peract dataset at `/home/ze/data/peract`:

```bash
python examples/rlbench/main.py \
    --task_names push_button open_drawer \
    --dataset_root /home/ze/data/peract \
    --use_demos \
    --num_trials_per_task 25
```

Expected peract directory layout:
```
/home/ze/data/peract/
  test/
    push_button/
      variation0/
        episodes/
          episode0/
            low_dim_obs.pkl
            front_rgb/   (or front_rgb.pkl)
            wrist_rgb/
          episode1/ ...
    open_drawer/ ...
```

If demo loading fails for any reason, the script falls back to random resets automatically.

---

## Action dimensions

The policy (`pi05_rlbench_lora`) outputs **7-dimensional** joint velocity actions via `RLBenchOutputs`.
The RLBench environment is configured with `ArmActionMode.ABS_JOINT_VELOCITY` and expects the same 7-dim vector.

State fed to the policy is **8-dimensional**: `[joint_positions (7), gripper_open (1)]`.

If you retrain with 8-dim actions (joint vel + gripper), update `RLBenchOutputs` in
[src/openpi/policies/rlbench_policy.py](../../src/openpi/policies/rlbench_policy.py)
to return `[:, :8]`.

---

## Output

- **Console**: per-episode success/failure, running success rate, final per-task and overall summary.
- **Videos**: saved as `<video_out_path>/<task_name>_<episode_idx>_<success|failure>.mp4`.

---

## Troubleshooting

**`ConnectionRefusedError`** — policy server is not running or wrong host/port.

**`RLBench ImportError`** — activate the conda env that has RLBench installed.

**Slow first episode** — JAX JIT-compiles on the first call; subsequent episodes are faster.

**Dimension mismatch in `task.step`** — check that `RLBenchOutputs` action slice matches your trained action space.
