"""Aggregate per-run low-data tidy JSONL files without concurrent appends."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--csv-out", type=pathlib.Path, default=None)
    parser.add_argument("--jsonl-out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    paths = sorted(args.results_dir.glob("runs/**/tidy_results.jsonl"))
    if not paths:
        raise SystemExit(f"No per-run tidy results found under {args.results_dir}")
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    zero_shot_path = args.results_dir / "source" / "target_zero_shot_eval.json"
    zero_shot_rates = {}
    if zero_shot_path.exists():
        payload = json.loads(zero_shot_path.read_text())
        zero_shot_rates = {str(key): float(value) for key, value in payload["target_success_zero_shot"].items()}
    for row in rows:
        # Preserve readability of pilot rows written before budget metadata was introduced.
        row.setdefault("budget_name", "legacy_fixed500")
        row.setdefault("budget_mode", "legacy_fixed_steps")
        target_key = f"{row['suite']}:{row['target_task_id']}"
        legacy_key = str(row["target_task_id"])
        if target_key in zero_shot_rates or legacy_key in zero_shot_rates:
            baseline = zero_shot_rates.get(target_key, zero_shot_rates.get(legacy_key))
            row["target_success_zero_shot"] = baseline
            row["target_success_gain"] = (
                row["success_rate"] - baseline if row["evaluated_task_role"] == "target" else None
            )
    rows.sort(
        key=lambda row: (
            row["suite"],
            row["target_task_id"],
            row["method"],
            row["num_demos"],
            row["budget_name"],
            row["seed"],
            row.get("evaluated_task_suite", row["suite"]),
            row["evaluated_task_id"],
        )
    )

    csv_out = args.csv_out or args.results_dir / "tidy_results.csv"
    jsonl_out = args.jsonl_out or args.results_dir / "tidy_results.jsonl"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    fieldnames.extend(sorted(set().union(*(row.keys() for row in rows)) - set(fieldnames)))
    with csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Aggregated {len(paths)} runs / {len(rows)} rows -> {csv_out}, {jsonl_out}")


if __name__ == "__main__":
    main()
