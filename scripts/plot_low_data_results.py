"""Plot and summarize an aggregated low-data adaptation result table."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


RUN_FIELDS = ("target_task_id", "method", "num_demos", "seed", "budget_name")
METHOD_COLORS = {"full": "#E45756", "lora": "#4C78A8"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    parser.add_argument("--num-trials", type=int, default=20)
    return parser.parse_args()


def _read_rows(path: pathlib.Path) -> list[dict]:
    integer_fields = {
        "target_task_id",
        "num_demos",
        "num_selected_trajectories",
        "seed",
        "num_transitions",
        "num_training_windows",
        "samples_seen",
        "optimizer_steps",
        "evaluated_task_id",
    }
    float_fields = {
        "effective_epochs",
        "success_rate",
        "source_success_before",
        "source_success_after",
        "source_forgetting",
        "source_retention",
        "target_success_zero_shot",
        "target_success_gain",
    }
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.setdefault("budget_name", "fixed500")
        row.setdefault("budget_mode", "fixed_steps")
        for field in integer_fields:
            row[field] = int(row[field])
        for field in float_fields:
            if field in row and row[field] not in ("", None):
                row[field] = float(row[field])
    return rows


def _run_key(row: dict) -> tuple:
    return tuple(row[field] for field in RUN_FIELDS)


def _validate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    target_rows = [row for row in rows if row["evaluated_task_role"] == "target"]
    source_rows = [row for row in rows if row["evaluated_task_role"] == "source"]
    if not target_rows:
        raise ValueError("No target rows found")
    if len({_run_key(row) for row in target_rows}) != len(target_rows):
        raise ValueError("Expected exactly one target row per adaptation run")
    source_ids = sorted({row["evaluated_task_id"] for row in source_rows})
    expected_source_rows = len(target_rows) * len(source_ids)
    if len(source_rows) != expected_source_rows:
        raise ValueError(f"Incomplete source evaluations: found {len(source_rows)}, expected {expected_source_rows}")
    return target_rows, source_rows


def _wilson_interval(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if math.isnan(rate):
        return math.nan, math.nan
    if n <= 0:
        return rate, rate
    denominator = 1.0 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _method_demo_summary(target_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for row in target_rows:
        grouped[(row["method"], row["num_demos"], row["budget_name"], row["budget_mode"])].append(row)
    summary = []
    for (method, demos, budget_name, budget_mode), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][2], item[0][1])
    ):
        item = {
            "method": method,
            "num_demos": demos,
            "budget_name": budget_name,
            "budget_mode": budget_mode,
            "num_runs": len(group),
            "target_success": float(np.mean([row["success_rate"] for row in group])),
            "source_success_before": float(np.mean([row["source_success_before"] for row in group])),
            "source_success_after": float(np.mean([row["source_success_after"] for row in group])),
            "source_forgetting": float(np.mean([row["source_forgetting"] for row in group])),
            "source_retention": float(np.mean([row["source_retention"] for row in group])),
            "effective_epochs": float(np.mean([row["effective_epochs"] for row in group])),
        }
        if all("target_success_zero_shot" in row for row in group):
            item["target_success_zero_shot"] = float(np.mean([row["target_success_zero_shot"] for row in group]))
            item["target_success_gain"] = item["target_success"] - item["target_success_zero_shot"]
        summary.append(item)
    return summary


def _series(rows: list[dict]) -> list[tuple[str, str]]:
    return sorted({(row["method"], row["budget_name"]) for row in rows})


def _series_label(series: tuple[str, str], all_series: list[tuple[str, str]]) -> str:
    method, budget_name = series
    return method.upper() if len({budget for _, budget in all_series}) == 1 else f"{method.upper()} / {budget_name}"


def _line_for(rows: list[dict], target: int, series: tuple[str, str], field: str, demos: list[int]) -> np.ndarray:
    method, budget_name = series
    lookup = {
        row["num_demos"]: row[field]
        for row in rows
        if row["target_task_id"] == target and row["method"] == method and row["budget_name"] == budget_name
    }
    return np.array([lookup.get(demo, np.nan) for demo in demos], dtype=np.float64)


def _setup_demo_axis(ax: plt.Axes, demos: list[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(demos, [str(demo) for demo in demos])
    ax.set_xlabel("Complete target trajectories (num_demos)")
    ax.grid(alpha=0.25)


def _plot_target_success(ax: plt.Axes, rows: list[dict], target: int, demos: list[int], n: int) -> None:
    all_series = _series(rows)
    target_rows = [row for row in rows if row["target_task_id"] == target]
    zero_shot = [row["target_success_zero_shot"] for row in target_rows if "target_success_zero_shot" in row]
    if zero_shot:
        ax.axhline(
            zero_shot[0],
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"Stage-A zero-shot ({zero_shot[0]:.2f})",
        )
    for series in all_series:
        method, budget_name = series
        values = _line_for(rows, target, series, "success_rate", demos)
        intervals = [_wilson_interval(value, n) for value in values]
        errors = np.array(
            [
                [value - low for value, (low, _) in zip(values, intervals, strict=True)],
                [high - value for value, (_, high) in zip(values, intervals, strict=True)],
            ]
        )
        ax.errorbar(
            demos,
            values,
            yerr=errors,
            marker="o",
            linewidth=2,
            capsize=3,
            color=METHOD_COLORS.get(method),
            linestyle="--" if "capped" in budget_name else "-",
            label=_series_label(series, all_series),
        )
    _setup_demo_axis(ax, demos)
    ax.set_ylim(-0.03, 1.04)
    ax.set_ylabel("Target success rate")
    ax.set_title(f"Unseen target task {target}")
    ax.legend()


def _plot_source_after(ax: plt.Axes, rows: list[dict], target: int, demos: list[int]) -> None:
    baseline = next(row["source_success_before"] for row in rows if row["target_task_id"] == target)
    ax.axhline(baseline, color="black", linestyle="--", linewidth=1.5, label=f"Stage-A baseline ({baseline:.3f})")
    all_series = _series(rows)
    for series in all_series:
        method, budget_name = series
        values = _line_for(rows, target, series, "source_success_after", demos)
        ax.plot(
            demos,
            values,
            marker="o",
            linewidth=2,
            color=METHOD_COLORS.get(method),
            linestyle="--" if "capped" in budget_name else "-",
            label=_series_label(series, all_series),
        )
    _setup_demo_axis(ax, demos)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Mean source success after adaptation")
    ax.set_title(f"Source retention after adapting to task {target}")
    ax.legend(fontsize=8)


def _plot_tradeoff(ax: plt.Axes, rows: list[dict], target: int) -> None:
    markers = {"full": "o", "lora": "s"}
    all_series = _series(rows)
    for series in all_series:
        method, budget_name = series
        selected = sorted(
            (
                row
                for row in rows
                if row["target_task_id"] == target and row["method"] == method and row["budget_name"] == budget_name
            ),
            key=lambda row: row["num_demos"],
        )
        x = [row["source_success_after"] for row in selected]
        y = [row["success_rate"] for row in selected]
        ax.plot(
            x,
            y,
            alpha=0.45,
            color=METHOD_COLORS.get(method),
            linestyle="--" if "capped" in budget_name else "-",
        )
        ax.scatter(
            x,
            y,
            s=65,
            marker=markers[method],
            color=METHOD_COLORS.get(method),
            label=_series_label(series, all_series),
        )
        for row in selected:
            ax.annotate(
                str(row["num_demos"]),
                (row["source_success_after"], row["success_rate"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Mean source success after adaptation")
    ax.set_ylabel("Target success rate")
    ax.set_title(f"Plasticity-retention trade-off, task {target}\n(labels are num_demos)")
    ax.grid(alpha=0.25)
    ax.legend()


def _plot_forgetting_heatmaps(
    figure: plt.Figure,
    axes: np.ndarray,
    source_rows: list[dict],
    targets: list[int],
    series_values: list[tuple[str, str]],
    demos: list[int],
) -> None:
    source_ids = sorted({row["evaluated_task_id"] for row in source_rows})
    image = None
    for ax, (target, series) in zip(
        axes.flat, [(target, series) for target in targets for series in series_values], strict=True
    ):
        method, budget_name = series
        lookup = {
            (row["evaluated_task_id"], row["num_demos"]): row["source_forgetting"]
            for row in source_rows
            if row["target_task_id"] == target and row["method"] == method and row["budget_name"] == budget_name
        }
        matrix = np.array([[lookup.get((task, demo), np.nan) for demo in demos] for task in source_ids])
        image = ax.imshow(matrix, vmin=-0.15, vmax=0.9, cmap="RdYlBu_r", aspect="auto")
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if np.isnan(value):
                    continue
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > 0.58 else "black",
                )
        ax.set_xticks(range(len(demos)), demos)
        ax.set_yticks(range(len(source_ids)), source_ids)
        ax.set_xlabel("num_demos")
        ax.set_ylabel("Source task ID")
        ax.set_title(f"Target {target} - {_series_label(series, series_values)}")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Source forgetting (before - after)", shrink=0.85)


def _save_condition_summary(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir.resolve()
    out_dir = (args.out_dir or results_dir / "plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(results_dir / "tidy_results.csv")
    target_rows, source_rows = _validate(rows)
    targets = sorted({row["target_task_id"] for row in target_rows})
    methods = sorted({row["method"] for row in target_rows})
    series_values = _series(target_rows)
    demos = sorted({row["num_demos"] for row in target_rows})
    seeds = sorted({row["seed"] for row in target_rows})

    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 5), constrained_layout=True)
    for ax, target in zip(np.atleast_1d(axes), targets, strict=True):
        _plot_target_success(ax, target_rows, target, demos, args.num_trials)
    fig.suptitle(f"Low-shot target adaptation ({args.num_trials} rollouts per point)", fontsize=15)
    fig.savefig(out_dir / "target_success_vs_demos.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 5), constrained_layout=True)
    for ax, target in zip(np.atleast_1d(axes), targets, strict=True):
        _plot_source_after(ax, target_rows, target, demos)
    fig.suptitle("Source capability retained after independent target adaptation", fontsize=15)
    fig.savefig(out_dir / "source_retention_vs_demos.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 5), constrained_layout=True)
    for ax, target in zip(np.atleast_1d(axes), targets, strict=True):
        _plot_tradeoff(ax, target_rows, target)
    fig.suptitle("Target plasticity versus source retention", fontsize=15)
    fig.savefig(out_dir / "plasticity_retention_tradeoff.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(
        len(targets),
        len(series_values),
        figsize=(7 * len(series_values), 5 * len(targets)),
        constrained_layout=True,
    )
    _plot_forgetting_heatmaps(
        fig,
        np.asarray(axes).reshape(len(targets), len(series_values)),
        source_rows,
        targets,
        series_values,
        demos,
    )
    fig.suptitle("Per-source-task forgetting", fontsize=16)
    fig.savefig(out_dir / "source_forgetting_heatmaps.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(targets), figsize=(7 * len(targets), 10), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, len(targets))
    for col, target in enumerate(targets):
        _plot_target_success(axes[0, col], target_rows, target, demos, args.num_trials)
        _plot_source_after(axes[1, col], target_rows, target, demos)
    split_id = target_rows[0]["split_id"]
    fig.suptitle(f"π₀.₅ low-data pilot — {split_id}", fontsize=17)
    fig.savefig(out_dir / "low_data_dashboard.png", dpi=220)
    fig.savefig(out_dir / "low_data_dashboard.pdf")
    plt.close(fig)

    condition_summary = _method_demo_summary(target_rows)
    _save_condition_summary(out_dir / "condition_summary.csv", condition_summary)
    overall = {}
    for method in methods:
        selected = [row for row in target_rows if row["method"] == method]
        overall[method] = {
            "mean_target_success": float(np.mean([row["success_rate"] for row in selected])),
            "mean_source_success_after": float(np.mean([row["source_success_after"] for row in selected])),
            "mean_source_forgetting": float(np.mean([row["source_forgetting"] for row in selected])),
            "mean_source_retention": float(np.mean([row["source_retention"] for row in selected])),
        }
    best_retention = max(condition_summary, key=lambda row: row["source_success_after"])
    summary = {
        "split_id": target_rows[0]["split_id"],
        "suite": target_rows[0]["suite"],
        "coverage": {
            "num_rows": len(rows),
            "num_adaptation_runs": len(target_rows),
            "target_task_ids": targets,
            "methods": methods,
            "training_budgets": sorted({row["budget_name"] for row in target_rows}),
            "num_demos": demos,
            "seeds": seeds,
            "complete": True,
        },
        "evaluation": {"num_trials_assumed": args.num_trials},
        "stage_a_source_success_macro": target_rows[0]["source_success_before"],
        "method_overall": overall,
        "by_method_and_num_demos": condition_summary,
        "best_mean_source_retention_condition": best_retention,
        "target_ceiling": {
            "runs_at_100_percent": sum(row["success_rate"] == 1.0 for row in target_rows),
            "runs_at_or_above_90_percent": sum(row["success_rate"] >= 0.9 for row in target_rows),
            "total_runs": len(target_rows),
        },
        "target_zero_shot": {
            str(target): next(
                (
                    row["target_success_zero_shot"]
                    for row in target_rows
                    if row["target_task_id"] == target and "target_success_zero_shot" in row
                ),
                None,
            )
            for target in targets
        },
        "caveats": [
            "Only one seed is present; differences cannot be treated as seed-level uncertainty.",
            "Target success is near ceiling, limiting conclusions about demo scaling.",
            "Wilson intervals in target plots represent rollout uncertainty only.",
        ],
    }
    (out_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote plots and summary to {out_dir}")


if __name__ == "__main__":
    main()
