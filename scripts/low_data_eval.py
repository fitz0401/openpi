"""Evaluate a Stage-A source checkpoint or one independent Stage-B target run."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import pathlib
import shutil

import continual_eval
from libero.libero import benchmark
from openpi_client import websocket_client_policy

from openpi.training.low_data.experiment import TaskRef
from openpi.training.low_data.experiment import load_experiment_config

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True, type=pathlib.Path)
    parser.add_argument("--train-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--repo-root", type=pathlib.Path, default=_REPO_ROOT)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-gpu", default="0")
    parser.add_argument("--server-timeout-s", type=float, default=900.0)
    parser.add_argument("--delete-checkpoint-after-eval", action="store_true")
    parser.add_argument("--retention-epsilon", type=float, default=1e-8)
    return parser.parse_args()


def _write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _evaluate_tasks(args, experiment, train_manifest, task_refs: list[TaskRef]) -> dict[TaskRef, float]:
    rates: dict[TaskRef, float] = {}
    with continual_eval.policy_server(
        str(args.repo_root),
        train_manifest["serve_config"],
        train_manifest["checkpoint_dir"],
        args.port,
        args.server_gpu,
        args.server_timeout_s,
    ):
        client = websocket_client_policy.WebsocketClientPolicy("0.0.0.0", args.port)
        for suite in dict.fromkeys(ref.suite for ref in task_refs):
            task_suite = benchmark.get_benchmark_dict()[suite](task_order_index=experiment.task_order_index)
            rollout_args = continual_eval.Args(
                manifest="unused",
                repo_root=str(args.repo_root),
                serve_config=train_manifest["serve_config"],
                task_suite_name=suite,
                num_trials=experiment.evaluation.num_trials,
                max_steps=experiment.evaluation.rollout_horizon(suite),
                replan_steps=experiment.evaluation.replan_steps,
                port=args.port,
                server_gpu=args.server_gpu,
                server_timeout_s=args.server_timeout_s,
                seed=7,
            )
            for ref in (ref for ref in task_refs if ref.suite == suite):
                rates[ref] = continual_eval.eval_task(
                    client,
                    task_suite,
                    ref.task_id,
                    rollout_args,
                    experiment.evaluation.num_trials,
                )
    return rates


def _target_rows(
    experiment,
    train_manifest,
    rates: dict[TaskRef, float],
    source_before: dict[TaskRef, float],
    target_success_zero_shot: float,
    epsilon: float,
):
    source_refs = experiment.source_task_refs()
    source_after = {ref: rates[ref] for ref in source_refs}
    macro_before = sum(source_before.values()) / len(source_before)
    macro_after = sum(source_after.values()) / len(source_after)
    target_ref = TaskRef(train_manifest["target_suite"], train_manifest["target_task_id"])
    adapted_target_success = rates[target_ref]
    common = {
        "source_split_id": train_manifest["source_split_id"],
        "source_checkpoint": train_manifest["source_checkpoint"],
        "target_suite": train_manifest["target_suite"],
        "split_id": experiment.split_id,
        "suite": train_manifest["target_suite"],  # Legacy alias for historical analysis tools.
        "source_task_ids": experiment.source_task_ids_by_suite(),
        "source_tasks": [dataclasses.asdict(ref) for ref in source_refs],
        "target_task_id": train_manifest["target_task_id"],
        "method": train_manifest["method"],
        "requested_data_budget": train_manifest["requested_data_budget"],
        "actual_num_demos": train_manifest["actual_num_demos"],
        "total_available_demos": train_manifest["total_available_demos"],
        "selected_trajectory_ids": train_manifest["selected_trajectory_ids"],
        "c1_trajectory_id": train_manifest["c1_trajectory_id"],
        "c1_definition": train_manifest["c1_definition"],
        "subset_seed": train_manifest["subset_seed"],
        "effective_epochs": train_manifest["effective_epochs"],
        "num_training_examples": train_manifest["num_training_examples"],
        "global_batch_size": train_manifest["global_batch_size"],
        "calculated_optimizer_steps": train_manifest["calculated_optimizer_steps"],
        "actual_optimizer_steps": train_manifest["actual_optimizer_steps"],
        "samples_seen": train_manifest["samples_seen"],
        "num_trials": experiment.evaluation.num_trials,
        "zero_shot_target_success": target_success_zero_shot,
        "adapted_target_success": adapted_target_success,
        "target_gain": adapted_target_success - target_success_zero_shot,
        "source_success_before_macro": macro_before,
        "source_success_after_macro": macro_after,
        "source_forgetting_macro": macro_before - macro_after,
        "source_retention_macro": macro_after / max(macro_before, epsilon),
        # Compatibility aliases that do not relabel all_available as a nominal demo budget.
        "num_selected_trajectories": train_manifest["actual_num_demos"],
        "seed": train_manifest["subset_seed"],
        "num_transitions": train_manifest["num_transitions"],
        "num_training_windows": train_manifest["num_training_windows"],
        "optimizer_steps": train_manifest["actual_optimizer_steps"],
        "target_success_zero_shot": target_success_zero_shot,
        "target_success_gain": adapted_target_success - target_success_zero_shot,
    }
    rows = []
    for ref in source_refs:
        before = source_before[ref]
        after = source_after[ref]
        rows.append(
            {
                **common,
                "evaluated_task_suite": ref.suite,
                "evaluated_task_id": ref.task_id,
                "evaluated_task_role": "source",
                "rollout_horizon": experiment.evaluation.rollout_horizon(ref.suite),
                "success_rate": after,
                "source_success_before": before,
                "source_success_after": after,
                "source_forgetting": before - after,
                "source_retention": after / max(before, epsilon),
                "source_metric_scope": "task",
            }
        )
    rows.append(
        {
            **common,
            "evaluated_task_suite": target_ref.suite,
            "evaluated_task_id": target_ref.task_id,
            "evaluated_task_role": "target",
            "rollout_horizon": experiment.evaluation.rollout_horizon(target_ref.suite),
            "success_rate": rates[target_ref],
            "target_success_gain": rates[target_ref] - target_success_zero_shot,
            "source_success_before": macro_before,
            "source_success_after": macro_after,
            "source_forgetting": macro_before - macro_after,
            "source_retention": macro_after / max(macro_before, epsilon),
            "source_metric_scope": "macro",
        }
    )
    return rows


def _write_tidy(result_dir: pathlib.Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with (result_dir / "tidy_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (result_dir / "tidy_results.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_target_zero_shot(experiment, train_manifest: dict, rates: dict[TaskRef, float]) -> pathlib.Path:
    result_dir = pathlib.Path(train_manifest["result_dir"])
    path = result_dir / "target_zero_shot_eval.json"
    _write_json(
        path,
        {
            "split_id": experiment.split_id,
            "target_task_ids": experiment.target_task_ids_by_suite(),
            "target_tasks": [dataclasses.asdict(ref) for ref in experiment.target_task_refs()],
            "target_success_zero_shot": {ref.key: rates[ref] for ref in experiment.target_task_refs()},
            "rollout_horizons": experiment.evaluation.rollout_horizons,
            "num_trials": experiment.evaluation.num_trials,
            "checkpoint_dir": train_manifest["checkpoint_dir"],
        },
    )
    return path


def _delete_checkpoint(train_manifest: dict) -> None:
    run_dir = pathlib.Path(train_manifest["checkpoint_run_dir"]).resolve()
    checkpoint_dir = pathlib.Path(train_manifest["checkpoint_dir"]).resolve()
    if run_dir not in checkpoint_dir.parents:
        raise ValueError(f"Refusing unsafe cleanup: {checkpoint_dir} is not inside {run_dir}")
    if run_dir.is_dir():
        logging.info("Deleting evaluated target checkpoint tree: %s", run_dir)
        shutil.rmtree(run_dir)


def _read_task_rates(payload: dict, field: str, refs: tuple[TaskRef, ...]) -> dict[TaskRef, float]:
    values = payload[field]
    rates = {}
    for ref in refs:
        key = ref.key if ref.key in values else str(ref.task_id)
        rates[ref] = float(values[key])
    return rates


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    experiment = load_experiment_config(args.experiment_config)
    train_manifest = json.loads(args.train_manifest.read_text())
    result_dir = pathlib.Path(train_manifest["result_dir"])

    if train_manifest["stage"] == "source":
        source_refs = experiment.source_task_refs()
        target_refs = experiment.target_task_refs()
        rates = _evaluate_tasks(args, experiment, train_manifest, [*source_refs, *target_refs])
        payload = {
            "split_id": experiment.split_id,
            "source_task_ids": experiment.source_task_ids_by_suite(),
            "source_tasks": [dataclasses.asdict(ref) for ref in source_refs],
            "source_success_before": {ref.key: rates[ref] for ref in source_refs},
            "source_success_before_macro": sum(rates[ref] for ref in source_refs) / len(source_refs),
            "target_task_ids": experiment.target_task_ids_by_suite(),
            "target_tasks": [dataclasses.asdict(ref) for ref in target_refs],
            "target_success_zero_shot": {ref.key: rates[ref] for ref in target_refs},
            "rollout_horizons": experiment.evaluation.rollout_horizons,
            "num_trials": experiment.evaluation.num_trials,
            "checkpoint_dir": train_manifest["checkpoint_dir"],
        }
        _write_json(result_dir / "source_eval.json", payload)
        _write_target_zero_shot(experiment, train_manifest, rates)
    elif train_manifest["stage"] == "target":
        source_eval_path = pathlib.Path(experiment.results_root) / experiment.split_id / "source" / "source_eval.json"
        if not source_eval_path.exists():
            raise FileNotFoundError(f"Stage-A evaluation is required first: {source_eval_path}")
        source_eval = json.loads(source_eval_path.read_text())
        source_before = _read_task_rates(source_eval, "source_success_before", experiment.source_task_refs())
        zero_shot_path = source_eval_path.parent / "target_zero_shot_eval.json"
        if not zero_shot_path.exists():
            raise FileNotFoundError(
                f"Target zero-shot evaluation is required; run scripts/job_low_data_zero_shot.sh: {zero_shot_path}"
            )
        zero_shot = json.loads(zero_shot_path.read_text())
        target_ref = TaskRef(train_manifest["target_suite"], train_manifest["target_task_id"])
        target_success_zero_shot = _read_task_rates(zero_shot, "target_success_zero_shot", (target_ref,))[target_ref]
        rates = _evaluate_tasks(args, experiment, train_manifest, [target_ref, *experiment.source_task_refs()])
        rows = _target_rows(
            experiment,
            train_manifest,
            rates,
            source_before,
            target_success_zero_shot,
            args.retention_epsilon,
        )
        _write_tidy(result_dir, rows)
        _write_json(
            result_dir / "eval_results.json",
            {
                "train_manifest": train_manifest,
                "success_rates": {ref.key: value for ref, value in rates.items()},
                "source_eval_path": str(source_eval_path),
                "num_trials": experiment.evaluation.num_trials,
            },
        )
    else:
        raise ValueError(f"Unknown manifest stage: {train_manifest['stage']!r}")

    if args.delete_checkpoint_after_eval:
        if train_manifest["stage"] != "target":
            raise ValueError("Automatic checkpoint deletion is allowed only for Stage-B target runs.")
        _delete_checkpoint(train_manifest)
    logging.info("Evaluation complete: %s", result_dir)


if __name__ == "__main__":
    main()
