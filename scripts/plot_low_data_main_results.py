"""Create compact multi-suite plots for the formal low-data experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")

METHOD_COLORS = {"lora": "#4C78A8", "full": "#E45756"}
DATA_BUDGET_POSITIONS = {"1": 1, "5": 5, "10": 10, "25": 25, "all_available": 50}


def _mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _budget_position(data_budget: str) -> int:
    return DATA_BUDGET_POSITIONS[data_budget] if data_budget in DATA_BUDGET_POSITIONS else int(data_budget)


def _read_target_rows(path: pathlib.Path) -> list[dict]:
    with path.open(newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["evaluated_task_role"] == "target"]
    for row in rows:
        row["_frozen_protocol"] = bool(row.get("requested_data_budget"))
        row["requested_data_budget"] = row.get("requested_data_budget") or row["num_demos"]
        row["data_budget_position"] = _budget_position(row["requested_data_budget"])
        for field in ("target_task_id", "seed", "optimizer_steps"):
            row[field] = int(row[field])
        for field in ("success_rate", "target_success_zero_shot"):
            row[field] = float(row[field])
        for field in ("source_success_after", "source_forgetting", "source_retention"):
            row[field] = float(row[field]) if row.get(field) not in (None, "") else float("nan")
    return rows


def _plot_suite_curves(rows: list[dict], out_dir: pathlib.Path, field: str, ylabel: str, filename: str) -> None:
    suites = list(dict.fromkeys(row["suite"] for row in rows))
    final_protocol = any(row["_frozen_protocol"] for row in rows)
    fig, axes = plt.subplots(1, len(suites), figsize=(6 * len(suites), 5), constrained_layout=True)
    for ax, suite in zip(np.atleast_1d(axes), suites, strict=True):
        suite_rows = [row for row in rows if row["suite"] == suite]
        for method in ("lora", "full"):
            selected = [row for row in suite_rows if row["method"] == method]
            budgets = sorted({row["requested_data_budget"] for row in selected}, key=_budget_position)
            values = [
                _mean([row[field] for row in selected if row["requested_data_budget"] == budget]) for budget in budgets
            ]
            positions = [_budget_position(budget) for budget in budgets]
            ax.plot(positions, values, marker="o", linewidth=2, color=METHOD_COLORS[method], label=method.upper())
        ax.set_xscale("log")
        ax.set_xticks(
            [1, 5, 10, 25, 50],
            ["1", "5", "10", "25", "all" if final_protocol else "50"],
        )
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(suite)
        ax.set_xlabel("Complete target trajectories")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(out_dir / filename, dpi=220)
    plt.close(fig)


def _plot_zero_shot(rows: list[dict], out_dir: pathlib.Path) -> None:
    targets = list(dict.fromkeys((row["suite"], row["target_task_id"]) for row in rows))
    rates = []
    for suite, task_id in targets:
        rates.append(
            next(
                row["target_success_zero_shot"]
                for row in rows
                if row["suite"] == suite and row["target_task_id"] == task_id
            )
        )
    labels = [f"{suite.replace('libero_', '')}:{task_id}" for suite, task_id in targets]
    colors = [f"C{list(dict.fromkeys(suite for suite, _ in targets)).index(suite)}" for suite, _ in targets]
    fig, ax = plt.subplots(figsize=(max(12, len(targets) * 0.65), 5), constrained_layout=True)
    ax.bar(np.arange(len(targets)), rates, color=colors)
    ax.set_xticks(np.arange(len(targets)), labels, rotation=55, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Stage-A zero-shot success")
    ax.set_title("Unified 18-source checkpoint: zero-shot performance on 22 targets")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "zero_shot_all_targets.png", dpi=220)
    plt.close(fig)


def _plot_target_heatmaps(rows: list[dict], out_dir: pathlib.Path) -> None:
    targets = list(dict.fromkeys((row["suite"], row["target_task_id"]) for row in rows))
    final_protocol = any(row["_frozen_protocol"] for row in rows)
    budgets = ["1", "5", "10", "25", "all_available"] if final_protocol else ["1", "5", "10", "25", "50"]
    fig, axes = plt.subplots(1, 2, figsize=(15, max(8, len(targets) * 0.42)), constrained_layout=True)
    for ax, method in zip(axes, ("lora", "full"), strict=True):
        matrix = np.full((len(targets), len(budgets) + 1), np.nan)
        for row_index, (suite, task_id) in enumerate(targets):
            target_rows = [row for row in rows if row["suite"] == suite and row["target_task_id"] == task_id]
            matrix[row_index, 0] = target_rows[0]["target_success_zero_shot"]
            for column, budget in enumerate(budgets, start=1):
                values = [
                    row["success_rate"]
                    for row in target_rows
                    if row["method"] == method and row["requested_data_budget"] == budget
                ]
                if values:
                    matrix[row_index, column] = _mean(values)
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(
            range(len(budgets) + 1),
            ["zero", "1", "5", "10", "25", "all" if final_protocol else "50"],
        )
        ax.set_yticks(
            range(len(targets)),
            [f"{suite.replace('libero_', '')}:{task_id}" for suite, task_id in targets],
        )
        ax.set_xlabel("requested_data_budget")
        ax.set_title(method.upper())
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Target success", shrink=0.75)
    fig.savefig(out_dir / "target_success_heatmaps.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--out-dir", type=pathlib.Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_target_rows(args.results_dir / "tidy_results.csv")
    if not rows:
        raise SystemExit("No completed target rows to plot.")

    _plot_zero_shot(rows, out_dir)
    _plot_suite_curves(rows, out_dir, "success_rate", "Mean target success", "target_success_by_suite.png")
    _plot_suite_curves(
        rows,
        out_dir,
        "source_success_after",
        "Mean success over all 18 source tasks",
        "source_retention_by_target_suite.png",
    )
    _plot_target_heatmaps(rows, out_dir)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["method"], row["requested_data_budget"])].append(row)
    summary = {
        "split_id": rows[0]["split_id"],
        "num_completed_runs": len(rows),
        "num_targets_with_results": len({(row["suite"], row["target_task_id"]) for row in rows}),
        "by_suite_method_demos": [
            {
                "suite": suite,
                "method": method,
                "requested_data_budget": data_budget,
                "num_runs": len(group),
                "mean_target_success": _mean([row["success_rate"] for row in group]),
                "mean_source_success_after": _mean([row["source_success_after"] for row in group]),
                "mean_source_forgetting": _mean([row["source_forgetting"] for row in group]),
                "mean_source_retention": _mean([row["source_retention"] for row in group]),
            }
            for (suite, method, data_budget), group in sorted(
                grouped.items(), key=lambda item: (item[0][0], item[0][1], _budget_position(item[0][2]))
            )
        ],
    }
    (out_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote formal multi-suite plots to {out_dir}")


if __name__ == "__main__":
    main()
