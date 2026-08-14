"""Stage-A source adaptation and independent Stage-B target adaptation.

This script only assembles existing TrainConfig/DataConfig objects and calls scripts.train.main;
the repository trainer, checkpoint format, and dataloader remain unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import shutil
import sys

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

import openpi.training.config as _config
from openpi.training.continual.subsample import EpisodeSubsetSpec
from openpi.training.low_data.experiment import TaskRef
from openpi.training.low_data.experiment import jsonable
from openpi.training.low_data.experiment import load_experiment_config
from openpi.training.low_data.experiment import nested_episode_subsets
from openpi.training.low_data.experiment import resolve_task_refs
from openpi.training.low_data.experiment import subset_statistics
from openpi.training.low_data.experiment import target_result_dir
import openpi.training.weight_loaders as weight_loaders

# ``uv run scripts/low_data_train.py`` puts scripts/ (not the repository root) on sys.path.
# Add the root so the existing trainer can be imported as ``scripts.train`` on cluster jobs.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Method selection is intentionally isolated from experiment orchestration so a future method can
# register its own model/config builder without changing the target grid logic.
METHOD_CONFIGS = {
    "full": "pi05_libero_low_data_full",
    "lora": "pi05_libero_low_data_lora",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True, type=pathlib.Path)
    parser.add_argument("--stage", required=True, choices=("source", "target"))
    parser.add_argument("--target-suite")
    parser.add_argument("--target-task-id", type=int)
    parser.add_argument("--method", choices=tuple(METHOD_CONFIGS))
    parser.add_argument("--data-budget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-base-dir", type=pathlib.Path)
    parser.add_argument("--descriptor-out", type=pathlib.Path, default=None)
    return parser.parse_args()


def _params_path(checkpoint: str) -> str:
    checkpoint = checkpoint.rstrip("/")
    return checkpoint if checkpoint.endswith("/params") else f"{checkpoint}/params"


def _remove_train_state(checkpoint_run_dir: pathlib.Path) -> None:
    for step_dir in checkpoint_run_dir.iterdir():
        train_state = step_dir / "train_state"
        if train_state.is_dir():
            logging.info("Removing non-resumable optimizer state: %s", train_state)
            shutil.rmtree(train_state)


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n")


def _build_source(args: argparse.Namespace, experiment, meta: LeRobotDatasetMetadata):
    base = _config.get_config(METHOD_CONFIGS["full"])
    source_refs = experiment.source_task_refs()
    task_strings = resolve_task_refs(meta, experiment, source_refs)
    episode_indices = sorted(
        ep_idx for ep_idx, episode in meta.episodes.items() if any(task in episode["tasks"] for task in task_strings)
    )
    stats = subset_statistics(
        meta,
        episode_indices,
        optimizer_steps=experiment.source.optimizer_steps,
        global_batch_size=experiment.source.batch_size,
    )
    result_dir = pathlib.Path(experiment.results_root) / experiment.split_id / "source"
    exp_name = f"{experiment.split_id}/source"
    subset_manifest = result_dir / "source_subset_manifest.json"
    data = dataclasses.replace(
        base.data,
        subsample_spec=EpisodeSubsetSpec(tasks=task_strings),
        subsample_indices_path=str(subset_manifest),
    )
    cfg = dataclasses.replace(
        base,
        checkpoint_base_dir=str(args.checkpoint_base_dir or experiment.checkpoint_root),
        exp_name=exp_name,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(_params_path(experiment.base_checkpoint)),
        seed=args.seed,
        batch_size=experiment.source.batch_size,
        num_train_steps=experiment.source.optimizer_steps,
        save_interval=experiment.source.optimizer_steps,
        keep_period=None,
        overwrite=True,
        resume=False,
    )
    metadata = {
        "schema_version": 1,
        "stage": "source",
        "split_id": experiment.split_id,
        "task_order_index": experiment.task_order_index,
        "source_task_ids": experiment.source_task_ids_by_suite(),
        "source_tasks": [dataclasses.asdict(ref) for ref in source_refs],
        "source_task_strings": list(task_strings),
        "target_task_ids": experiment.target_task_ids_by_suite(),
        "target_tasks": [dataclasses.asdict(ref) for ref in experiment.target_task_refs()],
        "base_checkpoint": experiment.base_checkpoint,
        "method": "full",
        "seed": args.seed,
        "episode_indices": episode_indices,
        **stats,
        "result_dir": str(result_dir.resolve()),
        "serve_config": base.name,
    }
    return cfg, result_dir, metadata


def _build_target(args: argparse.Namespace, experiment, meta: LeRobotDatasetMetadata):
    missing = [
        name
        for name, value in (
            ("--target-suite", args.target_suite),
            ("--target-task-id", args.target_task_id),
            ("--method", args.method),
            ("--data-budget", args.data_budget),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"Target stage requires: {', '.join(missing)}")
    target_ref = TaskRef(args.target_suite, args.target_task_id)
    if target_ref not in experiment.target_task_refs():
        raise ValueError(f"Target must be one of {experiment.target_task_refs()}; got {target_ref}")
    adaptation = experiment.adaptation
    if args.method not in adaptation.methods:
        raise ValueError(f"method must be one of {adaptation.methods}")
    method_data_budgets = adaptation.data_budgets_for_method(args.method)
    if args.data_budget not in method_data_budgets:
        raise ValueError(f"data_budget for {args.method} must be one of {method_data_budgets}")
    allowed_seeds = adaptation.seeds_for(args.target_suite, args.data_budget)
    if args.seed not in allowed_seeds:
        raise ValueError(f"seed for {args.target_suite} must be one of {allowed_seeds}")

    base = _config.get_config(METHOD_CONFIGS[args.method])
    task_string = resolve_task_refs(meta, experiment, (target_ref,))[0]
    trajectory_order, nested = nested_episode_subsets(
        meta,
        suite=args.target_suite,
        task_id=args.target_task_id,
        task=task_string,
        seed=args.seed,
        data_budgets=adaptation.data_budgets,
    )
    selected = nested[args.data_budget]
    c1_trajectory_id = nested["1"][0]
    if not selected or selected[0] != c1_trajectory_id:
        raise AssertionError("Protocol violation: every target subset must begin with its seed-matched D1 trajectory.")
    num_training_examples = sum(int(meta.episodes[index]["length"]) for index in selected)
    calculated_optimizer_steps = adaptation.calculate_optimizer_steps(
        num_training_examples=num_training_examples,
    )
    stats = subset_statistics(
        meta,
        selected,
        optimizer_steps=calculated_optimizer_steps,
        global_batch_size=adaptation.global_batch_size,
    )
    source_manifest_path = (
        pathlib.Path(experiment.results_root) / experiment.split_id / "source" / "train_manifest.json"
    )
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"Run and evaluate Stage A first; missing {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text())
    source_checkpoint = source_manifest["checkpoint_dir"]

    result_dir = target_result_dir(
        experiment,
        args.target_suite,
        args.target_task_id,
        args.method,
        args.data_budget,
        args.seed,
    )
    exp_name = (
        f"{experiment.split_id}/targets/{args.method}/{args.target_suite}/"
        f"task{args.target_task_id}/budget{args.data_budget}"
    )
    exp_name += f"/seed{args.seed}"
    subset_manifest = result_dir / "trajectory_subset_manifest.json"
    data = dataclasses.replace(
        base.data,
        subsample_spec=EpisodeSubsetSpec(tasks=(task_string,), episode_indices=tuple(selected)),
        subsample_indices_path=str(subset_manifest),
    )
    cfg = dataclasses.replace(
        base,
        checkpoint_base_dir=str(args.checkpoint_base_dir or experiment.checkpoint_root),
        exp_name=exp_name,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(_params_path(source_checkpoint)),
        seed=args.seed,
        batch_size=adaptation.global_batch_size,
        num_train_steps=calculated_optimizer_steps,
        save_interval=calculated_optimizer_steps,
        keep_period=None,
        overwrite=True,
        resume=False,
    )
    metadata = {
        "schema_version": 2,
        "stage": "target",
        "split_id": experiment.split_id,
        "source_split_id": experiment.split_id,
        "suite": args.target_suite,
        "target_suite": args.target_suite,
        "task_order_index": experiment.task_order_index,
        "source_task_ids": experiment.source_task_ids_by_suite(),
        "source_tasks": [dataclasses.asdict(ref) for ref in experiment.source_task_refs()],
        "target_task_id": args.target_task_id,
        "target_task_string": task_string,
        "method": args.method,
        "requested_data_budget": args.data_budget,
        "actual_num_demos": len(selected),
        "total_available_demos": len(trajectory_order),
        "adaptation": jsonable(adaptation),
        "seed": args.seed,
        "subset_seed": args.seed,
        "source_checkpoint": source_checkpoint,
        "selected_trajectory_ids": selected,
        "c1_trajectory_id": c1_trajectory_id,
        "c1_definition": "progress_r2_on_seed_matched_D1",
        "deterministic_trajectory_order": trajectory_order,
        "nested_trajectory_subsets": {str(budget): indices for budget, indices in nested.items()},
        "effective_epochs": adaptation.effective_epochs,
        "num_training_examples": num_training_examples,
        "global_batch_size": adaptation.global_batch_size,
        "calculated_optimizer_steps": calculated_optimizer_steps,
        "actual_optimizer_steps": None,
        **stats,
        "result_dir": str(result_dir.resolve()),
        "serve_config": base.name,
    }
    return cfg, result_dir, metadata


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    experiment = load_experiment_config(args.experiment_config)
    if args.stage == "target":
        import jax

        actual_world_size = jax.device_count()
        if actual_world_size != experiment.adaptation.world_size:
            raise RuntimeError(
                f"Configured adaptation.world_size={experiment.adaptation.world_size}, "
                f"but JAX sees {actual_world_size} devices."
            )
    metadata = LeRobotDatasetMetadata("physical-intelligence/libero")
    cfg, result_dir, run_metadata = (
        _build_source(args, experiment, metadata)
        if args.stage == "source"
        else _build_target(args, experiment, metadata)
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result_dir / "resolved_train_config.json", jsonable(cfg))
    _write_json(result_dir / "train_manifest.pending.json", run_metadata)
    if run_metadata["stage"] == "target":
        logging.info(
            "Scientific adaptation budget: effective_epochs=%s num_training_examples=%d "
            "global_batch_size=%d calculated_optimizer_steps=%d actual_optimizer_steps=pending "
            "samples_seen=%d",
            run_metadata["effective_epochs"],
            run_metadata["num_training_examples"],
            run_metadata["global_batch_size"],
            run_metadata["calculated_optimizer_steps"],
            run_metadata["samples_seen"],
        )

    import scripts.train as train_main

    train_main.main(cfg)
    checkpoint_run_dir = pathlib.Path(cfg.checkpoint_dir)
    final_step = cfg.num_train_steps - 1
    actual_optimizer_steps = final_step + 1
    if run_metadata["stage"] == "target":
        calculated_optimizer_steps = run_metadata["calculated_optimizer_steps"]
        if actual_optimizer_steps != calculated_optimizer_steps:
            raise AssertionError(
                f"Protocol violation: actual_optimizer_steps={actual_optimizer_steps} != "
                f"calculated_optimizer_steps={calculated_optimizer_steps}."
            )
        run_metadata["actual_optimizer_steps"] = actual_optimizer_steps
        logging.info(
            "Completed scientific adaptation budget: calculated_optimizer_steps=%d actual_optimizer_steps=%d",
            calculated_optimizer_steps,
            actual_optimizer_steps,
        )
    checkpoint_dir = checkpoint_run_dir / str(final_step)
    if not (checkpoint_dir / "params").exists():
        raise FileNotFoundError(f"Final params were not saved: {checkpoint_dir / 'params'}")
    _remove_train_state(checkpoint_run_dir)

    run_metadata.update(
        {
            "checkpoint_run_dir": str(checkpoint_run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "final_step": final_step,
            "resolved_train_config": str((result_dir / "resolved_train_config.json").resolve()),
        }
    )
    _write_json(result_dir / "train_manifest.json", run_metadata)
    (result_dir / "train_manifest.pending.json").unlink(missing_ok=True)
    if args.descriptor_out is not None:
        _write_json(args.descriptor_out, run_metadata)
    logging.info("Training complete: %s", result_dir / "train_manifest.json")


if __name__ == "__main__":
    main()
