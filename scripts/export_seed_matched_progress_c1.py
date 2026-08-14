#!/usr/bin/env python3
"""Export one strict V3 C1 value for every formal (target, subset_seed) D1."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

from openpi.training.low_data.experiment import load_experiment_config
from openpi.training.low_data.experiment import target_result_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True, type=pathlib.Path)
    parser.add_argument("--probe-per-demo-csv", required=True, type=pathlib.Path)
    parser.add_argument("--probe-source-subset", default="all_source")
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.experiment_config)
    with args.probe_per_demo_csv.open() as file:
        probe_rows = list(csv.DictReader(file))
    probe_rows = [row for row in probe_rows if row["probe_source_subset"] == args.probe_source_subset]
    if not probe_rows:
        raise ValueError(f"No probe rows use probe_source_subset={args.probe_source_subset!r}.")

    records = []
    for target in config.target_task_refs():
        for seed in config.adaptation.seeds_for(target.suite, "1"):
            run_dir = target_result_dir(config, target.suite, target.task_id, "lora", "1", seed)
            manifest_path = run_dir / "train_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing seed-matched D1 manifest: {manifest_path}")
            manifest = json.loads(manifest_path.read_text())
            selected = manifest["selected_trajectory_ids"]
            c1_trajectory_id = int(manifest["c1_trajectory_id"])
            if selected != [c1_trajectory_id] or manifest["nested_trajectory_subsets"]["1"] != selected:
                raise ValueError(f"Invalid D1/C1 binding in {manifest_path}")
            matches = [
                row
                for row in probe_rows
                if row["target_suite"] == target.suite
                and int(row["target_task_id"]) == target.task_id
                and int(row["episode_id"]) == c1_trajectory_id
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one probe row for {target.key}/seed{seed}/episode{c1_trajectory_id}, "
                    f"found {len(matches)}."
                )
            probe = matches[0]
            records.append(
                {
                    "target_suite": target.suite,
                    "target_task_id": target.task_id,
                    "subset_seed": seed,
                    "c1_trajectory_id": c1_trajectory_id,
                    "C1": float(probe["r2"]),
                    "probe_source_subset": args.probe_source_subset,
                    "probe_checkpoint": probe["checkpoint"],
                    "probe_seed": int(probe["seed"]),
                    "d1_manifest": str(manifest_path),
                }
            )

    output = args.output or (
        pathlib.Path(config.results_root) / config.split_id / "progress_compatibility/c1_by_target_seed.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    output.with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Exported {len(records)} strictly seed-matched C1 values to {output}")


if __name__ == "__main__":
    main()
