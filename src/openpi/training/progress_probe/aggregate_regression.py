"""Aggregate V3 source-subset probe evaluations into a source-to-target R² matrix."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

SOURCE_SUBSETS = (
    "spatial_source",
    "object_source",
    "goal_source",
    "spatial_object_source",
    "spatial_goal_source",
    "object_goal_source",
    "all_source",
)
TARGET_SUITES = ("libero_spatial", "libero_object", "libero_goal")
SOURCE_LABELS = {
    "spatial_source": "Spatial source",
    "object_source": "Object source",
    "goal_source": "Goal source",
    "spatial_object_source": "Spatial + Object",
    "spatial_goal_source": "Spatial + Goal",
    "object_goal_source": "Object + Goal",
    "all_source": "All sources",
}
TARGET_LABELS = {"libero_spatial": "Spatial target", "libero_object": "Object target", "libero_goal": "Goal target"}
MATCHED_SOURCE = {
    "libero_spatial": "spatial_source",
    "libero_object": "object_source",
    "libero_goal": "goal_source",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=pathlib.Path("results/progress_probe/progress_regression_lite_v3"),
    )
    parser.add_argument("--source-subsets", nargs="+", default=list(SOURCE_SUBSETS))
    parser.add_argument("--target-suites", nargs="+", default=list(TARGET_SUITES))
    parser.add_argument("--gate1-source-subset", default="all_source")
    parser.add_argument("--output-name", default="compatibility_matrix")
    return parser.parse_args()


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open() as file:
        return list(csv.DictReader(file))


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_subsets = tuple(args.source_subsets)
    target_suites = tuple(args.target_suites)
    if not set(source_subsets) <= set(SOURCE_LABELS):
        raise ValueError(f"Unknown source subset labels: {sorted(set(source_subsets) - set(SOURCE_LABELS))}")
    if not set(target_suites) <= set(TARGET_LABELS):
        raise ValueError(f"Unknown target suites: {sorted(set(target_suites) - set(TARGET_LABELS))}")
    if args.gate1_source_subset not in source_subsets:
        raise ValueError("gate1-source-subset must be included in source-subsets.")
    output_dir = args.results_root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    long_rows = []
    for subset in source_subsets:
        path = args.results_root / subset / "evaluation" / "per_suite.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; train and evaluate every required source subset first.")
        long_rows.extend(
            {
                "probe_source_subset": subset,
                "target_suite": row["target_suite"],
                "num_target_demos": int(row["num_target_demos"]),
                "r2": float(row["r2"]),
                "r2_task_std": float(row["r2_std"]),
                "mae": float(row["mae"]),
                "spearman_rho": float(row["spearman_rho"]),
                "pairwise_accuracy": float(row["pairwise_accuracy"]),
                "wrong_language_r2": float(row["wrong_language_r2"]),
                "checkpoint": row["checkpoint"],
                "seed": int(row["seed"]),
            }
            for row in read_csv(path)
            if row["target_suite"] in target_suites
        )
    expected = {(source, target) for source in source_subsets for target in target_suites}
    observed = {(row["probe_source_subset"], row["target_suite"]) for row in long_rows}
    if observed != expected:
        raise ValueError(f"Incomplete compatibility matrix: missing={sorted(expected - observed)}")
    if len({row["seed"] for row in long_rows}) != 1:
        raise ValueError("Compatibility matrix cells must use the same seed.")
    for target in target_suites:
        counts = {row["num_target_demos"] for row in long_rows if row["target_suite"] == target}
        if len(counts) != 1:
            raise ValueError(f"Source probes evaluated different demonstrations for {target}: {sorted(counts)}")
    write_csv(output_dir / "source_to_target_compatibility_long.csv", long_rows)
    (output_dir / "source_to_target_compatibility.json").write_text(json.dumps(long_rows, indent=2) + "\n")

    matrix = np.asarray(
        [
            [
                next(
                    row["r2"]
                    for row in long_rows
                    if row["probe_source_subset"] == source and row["target_suite"] == target
                )
                for target in target_suites
            ]
            for source in source_subsets
        ]
    )
    wide_rows = [
        {
            "probe_source_subset": source,
            **{target: matrix[index, column] for column, target in enumerate(target_suites)},
        }
        for index, source in enumerate(source_subsets)
    ]
    write_csv(output_dir / "source_to_target_compatibility_matrix.csv", wide_rows)

    dependence = []
    all_source_index = source_subsets.index(args.gate1_source_subset)
    single_source_indices = [source_subsets.index(MATCHED_SOURCE[suite]) for suite in TARGET_SUITES]
    for column, target in enumerate(target_suites):
        values = matrix[:, column]
        single_source_values = matrix[single_source_indices, column]
        matched_index = TARGET_SUITES.index(target)
        unmatched_values = np.delete(single_source_values, matched_index)
        dependence.append(
            {
                "target_suite": target,
                "r2_min_all_compositions": float(values.min()),
                "r2_max_all_compositions": float(values.max()),
                "r2_range_all_compositions": float(np.ptp(values)),
                "r2_std_all_compositions": float(values.std()),
                "r2_range_single_source_suites": float(np.ptp(single_source_values)),
                "matched_single_source_r2": float(single_source_values[matched_index]),
                "mean_unmatched_single_source_r2": float(unmatched_values.mean()),
                "matched_minus_unmatched_r2": float(single_source_values[matched_index] - unmatched_values.mean()),
                "all_source_r2": float(values[all_source_index]),
                "best_source_subset": source_subsets[int(values.argmax())],
            }
        )
    write_csv(output_dir / "source_dependence_summary.csv", dependence)
    gate1_rows = read_csv(args.results_root / args.gate1_source_subset / "evaluation" / "per_suite.csv")
    gate1_expected = {*TARGET_SUITES, "libero_10"}
    gate1_observed = {row["target_suite"] for row in gate1_rows}
    if gate1_observed != gate1_expected:
        raise ValueError(f"Gate 1 requires all Split-A target suites; got {sorted(gate1_observed)}")
    write_csv(output_dir / "gate1_all_source_per_suite.csv", gate1_rows)

    lower = min(float(matrix.min()), 0.0)
    upper = max(float(matrix.max()), 0.0)
    if lower == upper:
        lower, upper = -1.0, 1.0
    normalization = colors.TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper) if lower < 0 < upper else None
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    image = axis.imshow(
        matrix,
        cmap="RdYlGn",
        norm=normalization,
        vmin=None if normalization else lower,
        vmax=None if normalization else upper,
    )
    axis.set_xticks(np.arange(len(target_suites)), [TARGET_LABELS[target] for target in target_suites])
    axis.set_yticks(np.arange(len(source_subsets)), [SOURCE_LABELS[source] for source in source_subsets])
    for row in range(len(source_subsets)):
        for column in range(len(target_suites)):
            axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", fontsize=11)
    axis.set_title("Progress-regression compatibility: source subset → target suite")
    figure.colorbar(image, ax=axis, label="Mean target-task R²", shrink=0.85)
    figure.tight_layout()
    figure.savefig(output_dir / "source_to_target_compatibility_matrix.png", dpi=200)
    figure.savefig(output_dir / "source_to_target_compatibility_matrix.pdf")
    plt.close(figure)
    print(f"Wrote {len(source_subsets)}x{len(target_suites)} compatibility matrix to {output_dir}")


if __name__ == "__main__":
    main()
