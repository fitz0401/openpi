"""Evaluate a Stage-A source checkpoint or one independent Stage-B target run."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import shutil

import continual_eval
from libero.libero import benchmark
from openpi_client import websocket_client_policy

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


def _evaluate_tasks(args, experiment, train_manifest, task_ids: list[int]) -> dict[int, float]:
    suite_cls = benchmark.get_benchmark_dict()[experiment.suite]
    task_suite = suite_cls(task_order_index=experiment.task_order_index)
    rollout_args = continual_eval.Args(
        manifest="unused",
        repo_root=str(args.repo_root),
        serve_config=train_manifest["serve_config"],
        task_suite_name=experiment.suite,
        num_trials=experiment.evaluation.num_trials,
        max_steps=experiment.evaluation.max_steps,
        replan_steps=experiment.evaluation.replan_steps,
        port=args.port,
        server_gpu=args.server_gpu,
        server_timeout_s=args.server_timeout_s,
        seed=7,
    )
    rates: dict[int, float] = {}
    with continual_eval.policy_server(
        str(args.repo_root),
        train_manifest["serve_config"],
        train_manifest["checkpoint_dir"],
        args.port,
        args.server_gpu,
        args.server_timeout_s,
    ):
        client = websocket_client_policy.WebsocketClientPolicy("0.0.0.0", args.port)
        for task_id in task_ids:
            rates[task_id] = continual_eval.eval_task(
                client,
                task_suite,
                task_id,
                rollout_args,
                experiment.evaluation.num_trials,
            )
    return rates


def _target_rows(
    experiment,
    train_manifest,
    rates: dict[int, float],
    source_before: dict[int, float],
    target_success_zero_shot: float,
    epsilon: float,
):
    source_after = {task_id: rates[task_id] for task_id in experiment.source_task_ids}
    macro_before = sum(source_before.values()) / len(source_before)
    macro_after = sum(source_after.values()) / len(source_after)
    common = {
        "suite": experiment.suite,
        "split_id": experiment.split_id,
        "source_task_ids": list(experiment.source_task_ids),
        "target_task_id": train_manifest["target_task_id"],
        "method": train_manifest["method"],
        "num_demos": train_manifest["requested_num_demos"],
        # Defaults are only for pre-protocol manifests; new runs always record capped10epochs.
        "budget_name": train_manifest.get("budget_name", "legacy_fixed500"),
        "budget_mode": train_manifest.get("budget_mode", "legacy_fixed_steps"),
        "num_selected_trajectories": train_manifest["num_selected_trajectories"],
        "seed": train_manifest["seed"],
        "num_transitions": train_manifest["num_transitions"],
        "num_training_windows": train_manifest["num_training_windows"],
        "samples_seen": train_manifest["samples_seen"],
        "optimizer_steps": train_manifest["optimizer_steps"],
        "effective_epochs": train_manifest["effective_epochs"],
        "target_success_zero_shot": target_success_zero_shot,
        "target_success_gain": None,
    }
    rows = []
    for task_id in experiment.source_task_ids:
        before = source_before[task_id]
        after = source_after[task_id]
        rows.append(
            {
                **common,
                "evaluated_task_id": task_id,
                "evaluated_task_role": "source",
                "success_rate": after,
                "source_success_before": before,
                "source_success_after": after,
                "source_forgetting": before - after,
                "source_retention": after / max(before, epsilon),
                "source_metric_scope": "task",
            }
        )
    target_id = train_manifest["target_task_id"]
    rows.append(
        {
            **common,
            "evaluated_task_id": target_id,
            "evaluated_task_role": "target",
            "success_rate": rates[target_id],
            "target_success_gain": rates[target_id] - target_success_zero_shot,
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


def _write_target_zero_shot(experiment, train_manifest: dict, rates: dict[int, float]) -> pathlib.Path:
    result_dir = pathlib.Path(train_manifest["result_dir"])
    path = result_dir / "target_zero_shot_eval.json"
    _write_json(
        path,
        {
            "suite": experiment.suite,
            "split_id": experiment.split_id,
            "target_task_ids": list(experiment.target_task_ids),
            "target_success_zero_shot": {str(task_id): rates[task_id] for task_id in experiment.target_task_ids},
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


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    experiment = load_experiment_config(args.experiment_config)
    train_manifest = json.loads(args.train_manifest.read_text())
    result_dir = pathlib.Path(train_manifest["result_dir"])

    if train_manifest["stage"] == "source":
        eval_ids = [*experiment.source_task_ids, *experiment.target_task_ids]
        rates = _evaluate_tasks(args, experiment, train_manifest, eval_ids)
        payload = {
            "suite": experiment.suite,
            "split_id": experiment.split_id,
            "source_task_ids": list(experiment.source_task_ids),
            "source_success_before": {str(task_id): rates[task_id] for task_id in experiment.source_task_ids},
            "source_success_before_macro": (
                sum(rates[task_id] for task_id in experiment.source_task_ids) / len(experiment.source_task_ids)
            ),
            "target_success_zero_shot": {str(task_id): rates[task_id] for task_id in experiment.target_task_ids},
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
        source_before = {int(key): float(value) for key, value in source_eval["source_success_before"].items()}
        zero_shot_path = source_eval_path.parent / "target_zero_shot_eval.json"
        if not zero_shot_path.exists():
            raise FileNotFoundError(
                f"Target zero-shot evaluation is required; run scripts/job_low_data_zero_shot.sh: {zero_shot_path}"
            )
        zero_shot = json.loads(zero_shot_path.read_text())
        target_success_zero_shot = float(zero_shot["target_success_zero_shot"][str(train_manifest["target_task_id"])])
        eval_ids = [train_manifest["target_task_id"], *experiment.source_task_ids]
        rates = _evaluate_tasks(args, experiment, train_manifest, eval_ids)
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
                "success_rates": {str(key): value for key, value in rates.items()},
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
