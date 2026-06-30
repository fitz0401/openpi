"""Compute continual-learning metrics from evaluation outputs (openpi uv environment).

Reads one or more ``success_matrix.csv`` files (written by ``continual_eval.py``) and computes, per
run, the average accuracy / forgetting / backward & forward transfer / newest-task success via
``openpi.training.continual.metrics``. When multiple runs (seeds) of the same budget are given,
it also writes an aggregated ``metrics_summary.csv`` (mean/std across seeds).

Examples:
    # Single run:
    uv run scripts/continual_metrics.py --run-dirs <run_dir>

    # Aggregate seeds for several runs, write a summary table:
    uv run scripts/continual_metrics.py \
        --run-dirs <budget10/seed0> <budget10/seed1> <budget10/seed2> \
        --summary-out results/metrics_summary.csv
"""

import collections
import csv
import dataclasses
import json
import pathlib
from typing import Sequence

import tyro

from openpi.training.continual import metrics as _metrics


@dataclasses.dataclass
class Args:
    # Run directories, each containing success_matrix.csv (+ optionally manifest.json).
    run_dirs: Sequence[str] = ()
    # Where to write the cross-seed summary CSV (grouped by budget).
    summary_out: str = "results/metrics_summary.csv"


def _read_matrix(run_dir: pathlib.Path) -> tuple[dict[int, dict[int, float]], int, dict]:
    meta = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text())
    matrix: dict[int, dict[int, float]] = collections.defaultdict(dict)
    with (run_dir / "success_matrix.csv").open() as f:
        reader = csv.reader(f)
        header = next(reader)
        n_tasks = len(header) - 1
        for row in reader:
            stage = int(row[0])
            for c in range(1, n_tasks + 1):
                cell = row[c].strip()
                if cell != "":
                    matrix[stage][c] = float(cell)
    return matrix, n_tasks, meta


def main(args: Args) -> None:
    if not args.run_dirs:
        raise SystemExit("Provide at least one --run-dirs entry.")

    by_budget: dict[int, list[_metrics.ContinualMetrics]] = collections.defaultdict(list)

    for rd in args.run_dirs:
        run_dir = pathlib.Path(rd)
        matrix, n_tasks, meta = _read_matrix(run_dir)
        m = _metrics.compute_metrics(matrix, n_tasks)
        out = m.to_dict()
        out.update({"budget": meta.get("budget"), "seed": meta.get("seed"), "run_dir": str(run_dir)})
        (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))
        print(f"[{run_dir}] budget={meta.get('budget')} seed={meta.get('seed')}: {out}")
        if meta.get("budget") is not None:
            by_budget[meta["budget"]].append(m)

    if by_budget:
        summary_path = pathlib.Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"budget": b, **_metrics.aggregate(ms)} for b, ms in sorted(by_budget.items())]
        fieldnames = sorted({k for r in rows for k in r})
        # Put budget first.
        fieldnames = ["budget"] + [k for k in fieldnames if k != "budget"]
        with summary_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
