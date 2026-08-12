"""Audit expected Stage-B cells and write a machine-readable workflow status."""

from __future__ import annotations

import argparse
import json
import pathlib

from openpi.training.low_data.experiment import load_experiment_config
from openpi.training.low_data.experiment import target_grid
from openpi.training.low_data.experiment import target_result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True, type=pathlib.Path)
    args = parser.parse_args()
    config = load_experiment_config(args.experiment_config)
    result_root = pathlib.Path(config.results_root) / config.split_id
    missing = []
    completed = []
    for suite, task_id, method, data_budget, seed in target_grid(config):
        run_dir = target_result_dir(config, suite, task_id, method, data_budget, seed)
        cell = {
            "suite": suite,
            "target_task_id": task_id,
            "method": method,
            "requested_data_budget": data_budget,
            "subset_seed": seed,
        }
        if (run_dir / "tidy_results.jsonl").is_file():
            completed.append(cell)
        else:
            missing.append(cell)
    source_dir = result_root / "source"
    payload = {
        "split_id": config.split_id,
        "source_train_complete": (source_dir / "train_manifest.json").is_file(),
        "source_eval_complete": (source_dir / "source_eval.json").is_file(),
        "zero_shot_complete": (source_dir / "target_zero_shot_eval.json").is_file(),
        "expected_stage_b_runs": len(completed) + len(missing),
        "completed_stage_b_runs": len(completed),
        "missing_stage_b_runs": missing,
        "complete": not missing,
    }
    result_root.mkdir(parents=True, exist_ok=True)
    status_path = result_root / "workflow_status.json"
    status_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Stage B: {len(completed)}/{len(completed) + len(missing)} complete; status written to {status_path}")


if __name__ == "__main__":
    main()
