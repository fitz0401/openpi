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
    protocol_violations = []
    c1_by_target_seed: dict[tuple[str, int, int], int] = {}
    for suite, task_id, method, data_budget, seed in target_grid(config):
        run_dir = target_result_dir(config, suite, task_id, method, data_budget, seed)
        cell = {
            "suite": suite,
            "target_task_id": task_id,
            "method": method,
            "requested_data_budget": data_budget,
            "subset_seed": seed,
        }
        if not (run_dir / "tidy_results.jsonl").is_file():
            missing.append(cell)
            continue
        manifest_path = run_dir / "train_manifest.json"
        eval_path = run_dir / "eval_results.json"
        errors = []
        if not manifest_path.is_file():
            errors.append(f"missing {manifest_path}")
        else:
            manifest = json.loads(manifest_path.read_text())
            nested = manifest.get("nested_trajectory_subsets", {})
            d1 = nested.get("1", [])
            selected = manifest.get("selected_trajectory_ids", [])
            c1_trajectory_id = manifest.get("c1_trajectory_id")
            if len(d1) != 1:
                errors.append(f"nested D1 must contain exactly one trajectory, got {d1}")
            elif c1_trajectory_id != d1[0]:
                errors.append(f"c1_trajectory_id={c1_trajectory_id} does not match D1={d1[0]}")
            if selected != nested.get(data_budget):
                errors.append(f"selected trajectories do not match nested subset {data_budget}")
            if not selected or selected[0] != c1_trajectory_id:
                errors.append("selected subset does not begin with c1_trajectory_id")
            for field, expected in (
                ("target_suite", suite),
                ("target_task_id", task_id),
                ("method", method),
                ("requested_data_budget", data_budget),
                ("subset_seed", seed),
            ):
                if manifest.get(field) != expected:
                    errors.append(f"manifest {field}={manifest.get(field)!r}, expected {expected!r}")
            if len(d1) == 1:
                key = (suite, task_id, seed)
                previous = c1_by_target_seed.setdefault(key, d1[0])
                if previous != d1[0]:
                    errors.append(f"D1 changed across budgets: expected {previous}, got {d1[0]}")
        if not eval_path.is_file():
            errors.append(f"missing {eval_path}")
        else:
            eval_payload = json.loads(eval_path.read_text())
            expected_protocol = config.evaluation.protocol_manifest()
            if eval_payload.get("evaluation_protocol") != expected_protocol:
                errors.append("evaluation protocol does not match the configured formal protocol")
            expected_retention = config.evaluation.should_evaluate_source_retention(
                target_suite=suite, subset_seed=seed
            )
            if eval_payload.get("target_num_trials") != config.evaluation.target_num_trials(suite):
                errors.append("target trial count does not match target suite protocol")
            if eval_payload.get("source_retention_evaluated") is not expected_retention:
                errors.append(f"source_retention_evaluated must be {expected_retention}")
            rows = [json.loads(line) for line in (run_dir / "tidy_results.jsonl").read_text().splitlines()]
            target_rows = [row for row in rows if row["evaluated_task_role"] == "target"]
            source_rows = [row for row in rows if row["evaluated_task_role"] == "source"]
            if len(target_rows) != 1 or target_rows[0].get("num_trials") != config.evaluation.target_num_trials(suite):
                errors.append("tidy output must contain one target row with the suite-specific trial count")
            expected_source_rows = len(config.source_task_refs()) if expected_retention else 0
            if len(source_rows) != expected_source_rows:
                errors.append(f"expected {expected_source_rows} source-retention rows, got {len(source_rows)}")
            if any(row.get("num_trials") != config.evaluation.source_retention_num_trials for row in source_rows):
                errors.append("source-retention row uses an incorrect trial count")
        if errors:
            protocol_violations.append({**cell, "errors": errors})
        else:
            completed.append(cell)
    source_dir = result_root / "source"
    baseline_paths = (source_dir / "source_eval.json", source_dir / "target_zero_shot_eval.json")
    baseline_protocols = {}
    for path in baseline_paths:
        try:
            baseline_protocols[path.name] = json.loads(path.read_text()).get("evaluation_protocol")
        except (FileNotFoundError, json.JSONDecodeError):
            baseline_protocols[path.name] = None
    expected_protocol = config.evaluation.protocol_manifest()
    baseline_protocol_valid = all(value == expected_protocol for value in baseline_protocols.values())
    payload = {
        "split_id": config.split_id,
        "source_train_complete": (source_dir / "train_manifest.json").is_file(),
        "source_eval_complete": (source_dir / "source_eval.json").is_file(),
        "zero_shot_complete": (source_dir / "target_zero_shot_eval.json").is_file(),
        "expected_evaluation_protocol": expected_protocol,
        "baseline_evaluation_protocols": baseline_protocols,
        "baseline_protocol_valid": baseline_protocol_valid,
        "expected_stage_b_runs": len(completed) + len(missing) + len(protocol_violations),
        "completed_stage_b_runs": len(completed),
        "missing_stage_b_runs": missing,
        "c1_binding": "progress R2 must use the exact c1_trajectory_id / D1 episode for each target and seed",
        "verified_c1_bindings": len(c1_by_target_seed),
        "protocol_violations": protocol_violations,
        "complete": baseline_protocol_valid and not missing and not protocol_violations,
    }
    result_root.mkdir(parents=True, exist_ok=True)
    status_path = result_root / "workflow_status.json"
    status_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Stage B: {len(completed)}/{len(completed) + len(missing) + len(protocol_violations)} valid; "
        f"protocol violations={len(protocol_violations)}; status written to {status_path}"
    )


if __name__ == "__main__":
    main()
