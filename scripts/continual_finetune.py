"""Continual (sequential) finetuning orchestrator for the LIBERO-Object benchmark.

Runs ONE ``(demo_budget, seed)`` configuration of the continual benchmark: it finetunes pretrained
pi0.5 on the first ``n_tasks`` LIBERO-Object tasks one after another, WITHOUT resetting model
weights between tasks. Each stage reuses the standard training loop (``scripts/train.py:main``)
unchanged -- the only continual-specific pieces are:

  * task K initializes from task K-1's checkpoint (via the existing ``CheckpointWeightLoader``);
    task 1 initializes from the pretrained pi0.5 checkpoint. A fresh optimizer is used per task,
    so weights carry over but the optimizer does not (this is the "no weight reset" requirement).
  * the dataset is subsampled to a single task and ``budget`` reproducibly-chosen demos
    (``data.subsample_spec``).

This script runs in the openpi (uv) environment. Evaluation is a separate step (continual_eval.py)
that needs the LIBERO simulator.

Example (one budget, one seed):
    uv run scripts/continual_finetune.py \
        --run-name slice_v0 --budget 10 --seed 0 \
        --n-tasks 5 --num-train-steps 800 --save-interval 200
"""

import dataclasses
import json
import logging
import pathlib
import shutil
import sys
from typing import Literal

# Ensure the repo root is importable so we can reuse scripts/train.py's training loop.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tyro

import openpi.training.config as _config
from openpi.training.continual.libero_object_tasks import LIBERO_OBJECT_TASKS
from openpi.training.continual.subsample import SubsampleSpec
from openpi.training.continual.subsample import resolve_task_string
import openpi.training.weight_loaders as weight_loaders


@dataclasses.dataclass
class Args:
    # Name for this benchmark run (groups all budgets/seeds under one tree).
    run_name: str = "slice_v0"
    # Demo budget for every task in the sequence (capped at available demos per task).
    budget: int = 10
    # Reproducibility seed: controls both demo subsampling and training RNG.
    seed: int = 0
    # Number of sequential tasks (first N of LIBERO-Object).
    n_tasks: int = 5
    # Base template config (filled in per stage). Usually leave as default.
    base_config: str = "pi05_libero_object_continual"
    # Training steps and checkpoint interval per task. save_interval also sets the
    # learning-curve granularity (each saved checkpoint becomes a learning-curve point).
    num_train_steps: int = 1_000
    save_interval: int = 250
    # Checkpoint storage policy:
    #   eval: retain only each stage's final params/assets (~60 GiB for five stages), no optimizer.
    #   learning_curves: retain interval params/assets for recovery curves, no optimizer.
    #   resumable: retain interval params/assets/train_state (very large; supports training resume).
    checkpoint_mode: Literal["eval", "learning_curves", "resumable"] = "eval"
    # Override batch size (must divide device count). Defaults to the template's value if None.
    batch_size: int | None = None


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    base = _config.get_config(args.base_config)
    effective_save_interval = args.num_train_steps if args.checkpoint_mode == "eval" else args.save_interval

    # Resolve the canonical task strings against the actual training dataset so we filter on exactly
    # the strings present in the dataset (fails loudly on mismatch).
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    repo_id = base.data.repo_id
    dataset_root = base.data.dataset_root
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=dataset_root)
    task_strings = [resolve_task_string(meta, LIBERO_OBJECT_TASKS[i]) for i in range(args.n_tasks)]
    logging.info("Continual task order (%d tasks): %s", args.n_tasks, task_strings)

    # Derive the run root directly from checkpoint_base_dir/name (the template's exp_name is unset,
    # so we cannot use base.checkpoint_dir here). This matches each stage's cfg.checkpoint_dir parent.
    run_root = (
        pathlib.Path(base.checkpoint_base_dir).resolve()
        / base.name
        / args.run_name
        / f"budget{args.budget}"
        / f"seed{args.seed}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_name": args.run_name,
        "base_config": args.base_config,
        "budget": args.budget,
        "seed": args.seed,
        "n_tasks": args.n_tasks,
        "num_train_steps": args.num_train_steps,
        "save_interval": effective_save_interval,
        "requested_save_interval": args.save_interval,
        "checkpoint_mode": args.checkpoint_mode,
        "task_strings": task_strings,
        "stages": [],
    }

    final_step = args.num_train_steps - 1
    prev_stage_dir: pathlib.Path | None = None

    for k, task in enumerate(task_strings, start=1):
        exp_name = f"{args.run_name}/budget{args.budget}/seed{args.seed}/stage{k}_task{k - 1}"

        if k == 1:
            weight_loader = base.weight_loader  # pretrained pi0.5
        else:
            params_path = str(prev_stage_dir / str(final_step) / "params")
            weight_loader = weight_loaders.CheckpointWeightLoader(params_path)
            logging.info("Stage %d initializing from %s", k, params_path)

        indices_path = str(run_root / f"stage{k}_task{k - 1}_sampled_indices.json")
        data_factory = dataclasses.replace(
            base.data,
            subsample_spec=SubsampleSpec(task=task, budget=args.budget, seed=args.seed),
            subsample_indices_path=indices_path,
        )

        overrides = dict(
            exp_name=exp_name,
            data=data_factory,
            weight_loader=weight_loader,
            seed=args.seed,
            num_train_steps=args.num_train_steps,
            save_interval=effective_save_interval,
            # In eval mode Orbax's max_to_keep=1 retains only the final checkpoint. The other modes
            # preserve interval checkpoints for within-stage recovery curves / resume respectively.
            keep_period=None if args.checkpoint_mode == "eval" else effective_save_interval,
            overwrite=True,
            resume=False,
        )
        if args.batch_size is not None:
            overrides["batch_size"] = args.batch_size
        cfg = dataclasses.replace(base, **overrides)

        logging.info("=" * 80)
        logging.info("STAGE %d/%d  task=%r  exp=%s", k, args.n_tasks, task, exp_name)
        logging.info("=" * 80)

        # Reuse the standard training loop unchanged.
        import scripts.train as train_main  # local import: heavy JAX deps

        train_main.main(cfg)

        stage_dir = cfg.checkpoint_dir
        if args.checkpoint_mode != "resumable":
            # All later stages and policy evaluation load only <step>/params. Removing optimizer
            # state after a completed stage saves ~19 GiB per checkpoint for pi0.5.
            for checkpoint_dir in stage_dir.iterdir():
                train_state_dir = checkpoint_dir / "train_state"
                if train_state_dir.is_dir():
                    logging.info("Removing non-resumable optimizer state: %s", train_state_dir)
                    shutil.rmtree(train_state_dir)

        if args.checkpoint_mode == "eval":
            learning_curve_steps = [final_step]
        else:
            learning_curve_steps = sorted(
                {s for s in range(0, args.num_train_steps, args.save_interval) if s > 0}
                | {final_step}
            )
        manifest["stages"].append(
            {
                "stage": k,
                "task_id": k - 1,
                "task": task,
                "exp_name": exp_name,
                "checkpoint_dir": str(stage_dir),
                "final_step": final_step,
                "final_checkpoint": str(stage_dir / str(final_step)),
                "sampled_indices_path": indices_path,
                "learning_curve_steps": learning_curve_steps,
            }
        )
        prev_stage_dir = stage_dir

        # Write/refresh the manifest after each stage so partial runs are still usable.
        (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logging.info("Continual finetuning complete. Manifest: %s", run_root / "manifest.json")


if __name__ == "__main__":
    main(tyro.cli(Args))
