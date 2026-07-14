"""Plot and summarize one continual LIBERO evaluation directory.

Expected inputs are the files produced by ``scripts/continual_eval.py`` and
``scripts/continual_metrics.py``: success_matrix.csv, learning_curves.csv,
eval_results.json, and metrics.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    parser.add_argument("--num-trials", type=int, default=20, help="Rollouts per success-matrix cell.")
    parser.add_argument("--num-trials-lc", type=int, default=10, help="Rollouts per intermediate LC point.")
    return parser.parse_args()


def _read_matrix(path: pathlib.Path) -> tuple[list[int], list[str], np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError(f"No matrix rows found in {path}")
    headers = rows[0][1:]
    stages = [int(row[0]) for row in rows[1:]]
    matrix = np.array(
        [[float(cell) if cell.strip() else np.nan for cell in row[1:]] for row in rows[1:]],
        dtype=np.float64,
    )
    return stages, headers, matrix


def _read_learning_curves(path: pathlib.Path) -> list[dict[str, int | float]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [
            {
                "stage": int(row["stage"]),
                "task_id": int(row["task_id"]),
                "step": int(row["step"]),
                "success_rate": float(row["success_rate"]),
            }
            for row in csv.DictReader(f)
        ]


def _short_task_name(task: str, fallback: str) -> str:
    match = re.match(r"pick up the (.+?) and place it in the basket", task.lower())
    return match.group(1).title() if match else fallback


def _binomial_error(rate: np.ndarray | float, n: int) -> np.ndarray:
    values = np.asarray(rate, dtype=np.float64)
    if n <= 0:
        return np.zeros_like(values)
    return 1.96 * np.sqrt(np.clip(values * (1.0 - values) / n, 0.0, None))


def _annotate_heatmap(ax: plt.Axes, matrix: np.ndarray) -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isnan(value):
                continue
            color = "white" if value >= 0.58 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)


def _plot_heatmap(ax: plt.Axes, stages: list[int], labels: list[str], matrix: np.ndarray) -> None:
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    _annotate_heatmap(ax, matrix)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(range(len(stages)), [f"After T{s}" for s in stages])
    ax.set_xlabel("Evaluation task")
    ax.set_ylabel("Training stage")
    ax.set_title("Continual success matrix")
    plt.colorbar(image, ax=ax, label="Success rate", fraction=0.046, pad=0.04)


def _plot_trajectories(
    ax: plt.Axes, stages: list[int], labels: list[str], matrix: np.ndarray, num_trials: int
) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for col, (label, color) in enumerate(zip(labels, colors)):
        values = matrix[:, col]
        ax.plot(stages, values, marker="o", linewidth=2, color=color, label=f"T{col + 1}: {label}")
        learned_row = next((idx for idx, stage in enumerate(stages) if stage == col + 1), None)
        if learned_row is not None and not np.isnan(values[learned_row]):
            ax.scatter(
                [col + 1], [values[learned_row]], marker="*", s=150, color=color, edgecolor="black", zorder=5
            )
    ax.set_xticks(stages)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Completed training stage")
    ax.set_ylabel("Success rate")
    ax.set_title(f"Retention and transfer trajectories ({num_trials} trials/cell)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="lower left")


def _plot_learning_curves(
    ax: plt.Axes, lc_rows: list[dict[str, int | float]], labels: list[str], num_trials_lc: int
) -> None:
    if not lc_rows:
        ax.text(0.5, 0.5, "No intermediate learning-curve data", ha="center", va="center")
        ax.set_axis_off()
        return
    for stage in sorted({int(row["stage"]) for row in lc_rows}):
        rows = sorted((row for row in lc_rows if row["stage"] == stage), key=lambda row: int(row["step"]))
        steps = np.array([int(row["step"]) for row in rows])
        rates = np.array([float(row["success_rate"]) for row in rows])
        label = labels[stage - 1] if stage <= len(labels) else f"Task {stage}"
        ax.errorbar(
            steps,
            rates,
            yerr=_binomial_error(rates, num_trials_lc),
            marker="o",
            linewidth=2,
            capsize=3,
            label=f"T{stage}: {label}",
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Finetuning step within stage")
    ax.set_ylabel("Current-task success rate")
    ax.set_title(f"Within-stage learning curves ({num_trials_lc} trials/point)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="lower right")


def _plot_immediate_vs_final(
    ax: plt.Axes, stages: list[int], labels: list[str], matrix: np.ndarray, num_trials: int
) -> tuple[np.ndarray, np.ndarray]:
    stage_to_row = {stage: idx for idx, stage in enumerate(stages)}
    immediate = np.array(
        [matrix[stage_to_row[task], task - 1] if task in stage_to_row else np.nan for task in range(1, len(labels) + 1)]
    )
    final = matrix[-1]
    x = np.arange(len(labels))
    width = 0.37
    ax.bar(
        x - width / 2,
        immediate,
        width,
        yerr=_binomial_error(immediate, num_trials),
        capsize=3,
        label="Immediately after learning",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        final,
        width,
        yerr=_binomial_error(final, num_trials),
        capsize=3,
        label="After final stage",
        color="#E45756",
    )
    ax.set_xticks(x, [f"T{i + 1}\n{label}" for i, label in enumerate(labels)], rotation=15, ha="right")
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Success rate")
    ax.set_title("Immediate plasticity versus final retention")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    return immediate, final


def _save_single_plot(path: pathlib.Path, plotter) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    plotter(ax)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or run_dir / "plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_results = json.loads((run_dir / "eval_results.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    stages, headers, matrix = _read_matrix(run_dir / "success_matrix.csv")
    lc_rows = _read_learning_curves(run_dir / "learning_curves.csv")
    task_strings = eval_results.get("task_strings", [])
    labels = [
        _short_task_name(task_strings[idx], f"Task {idx + 1}") if idx < len(task_strings) else headers[idx]
        for idx in range(len(headers))
    ]

    _save_single_plot(out_dir / "success_matrix.png", lambda ax: _plot_heatmap(ax, stages, labels, matrix))
    _save_single_plot(
        out_dir / "retention_trajectories.png",
        lambda ax: _plot_trajectories(ax, stages, labels, matrix, args.num_trials),
    )
    _save_single_plot(
        out_dir / "within_stage_learning_curves.png",
        lambda ax: _plot_learning_curves(ax, lc_rows, labels, args.num_trials_lc),
    )

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    immediate, final = _plot_immediate_vs_final(ax, stages, labels, matrix, args.num_trials)
    fig.savefig(out_dir / "immediate_vs_final.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    _plot_heatmap(axes[0, 0], stages, labels, matrix)
    _plot_trajectories(axes[0, 1], stages, labels, matrix, args.num_trials)
    _plot_learning_curves(axes[1, 0], lc_rows, labels, args.num_trials_lc)
    _plot_immediate_vs_final(axes[1, 1], stages, labels, matrix, args.num_trials)
    budget = eval_results.get("budget", metrics.get("budget", "?"))
    seed = eval_results.get("seed", metrics.get("seed", "?"))
    fig.suptitle(f"π₀.₅ LIBERO-Object continual finetuning — budget={budget}, seed={seed}", fontsize=17)
    fig.savefig(out_dir / "continual_dashboard.png", dpi=200)
    fig.savefig(out_dir / "continual_dashboard.pdf")
    plt.close(fig)

    forgetting = immediate - final
    forgetting[-1] = np.nan  # The final task has no later stage in which it can be forgotten.
    summary = {
        "budget": budget,
        "seed": seed,
        "num_trials_assumed": args.num_trials,
        "num_trials_lc_assumed": args.num_trials_lc,
        "mean_immediate_success": float(np.nanmean(immediate)),
        "final_average_success": float(np.nanmean(final)),
        "retained_fraction_of_immediate_performance": float(np.nanmean(final) / np.nanmean(immediate)),
        "average_forgetting_earlier_tasks": float(np.nanmean(forgetting)),
        "tasks": [
            {
                "stage": idx + 1,
                "name": labels[idx],
                "immediate_success": float(immediate[idx]),
                "final_success": float(final[idx]),
                "forgetting": None if math.isnan(forgetting[idx]) else float(forgetting[idx]),
            }
            for idx in range(len(labels))
        ],
        "reported_metrics": metrics,
    }
    (out_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote plots and summary to {out_dir}")


if __name__ == "__main__":
    main()
