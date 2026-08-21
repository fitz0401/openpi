#!/usr/bin/env python3
"""Relate a frozen progress probe to formal LoRA outcomes and demo allocation.

This is a read-only analysis of existing probe/adaptation outputs. It never trains or evaluates a
model. C1 is joined to the exact seed-matched D1 trajectory recorded by each adaptation manifest.
"""

# ruff: noqa: PD010, PERF401

from __future__ import annotations

import argparse
import json
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CONTROLLED_SUITES = ("libero_spatial", "libero_object", "libero_goal")
PRIMARY_BUDGETS = ("1", "5", "10", "25")
SUITE_LABELS = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal"}
SUITE_COLORS = {"libero_spatial": "#4477AA", "libero_object": "#EE6677", "libero_goal": "#228833"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptation-root", required=True, type=pathlib.Path)
    parser.add_argument("--probe-per-demo", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--random-permutations", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260821)
    return parser.parse_args()


def load_adaptation_runs(root: pathlib.Path) -> pd.DataFrame:
    records = []
    for tidy_path in sorted((root / "runs" / "lora").rglob("tidy_results.csv")):
        rows = pd.read_csv(tidy_path)
        target = rows.loc[rows["evaluated_task_role"] == "target"]
        if len(target) != 1:
            raise ValueError(f"Expected exactly one target row in {tidy_path}; found {len(target)}")
        manifest_path = tidy_path.with_name("train_manifest.json")
        manifest = json.loads(manifest_path.read_text())
        row = target.iloc[0]
        records.append(
            {
                "target_suite": str(row["target_suite"]),
                "target_task_id": int(row["target_task_id"]),
                "subset_seed": int(row["subset_seed"]),
                "requested_data_budget": str(row["requested_data_budget"]),
                "actual_num_demos": int(row["actual_num_demos"]),
                "total_available_demos": int(row["total_available_demos"]),
                "c1_trajectory_id": int(manifest["c1_trajectory_id"]),
                "selected_trajectory_ids": json.dumps(manifest["selected_trajectory_ids"]),
                "zero_shot_success": float(row["zero_shot_target_success"]),
                "adapted_target_success": float(row["adapted_target_success"]),
                "target_gain": float(row["target_gain"]),
                "source_success_after": pd.to_numeric(row["source_success_after_macro"], errors="coerce"),
                "source_forgetting": pd.to_numeric(row["source_forgetting_macro"], errors="coerce"),
                "source_retention": pd.to_numeric(row["source_retention_macro"], errors="coerce"),
                "target_num_trials": int(row["target_num_trials"]),
                "manifest": str(manifest_path.resolve()),
            }
        )
    result = pd.DataFrame(records)
    result = result[result.target_suite.isin(CONTROLLED_SUITES)].copy()
    if result.empty:
        raise ValueError(f"No controlled-suite LoRA results found below {root}")
    duplicates = result.duplicated(
        ["target_suite", "target_task_id", "subset_seed", "requested_data_budget"], keep=False
    )
    if duplicates.any():
        raise ValueError(f"Duplicate adaptation cells:\n{result.loc[duplicates]}")
    return result


def attach_compatibility(runs: pd.DataFrame, probe_path: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    probe = pd.read_csv(probe_path)
    probe = probe[(probe["probe_source_subset"] == "all_source") & probe.target_suite.isin(CONTROLLED_SUITES)].copy()
    probe["episode_id"] = probe.episode_id.astype(int)
    per_task = probe.groupby(["target_suite", "target_task_id"], as_index=False).agg(
        C_all=("r2", "mean"), C_all_std=("r2", "std"), num_probe_demos=("r2", "size")
    )
    joined = runs.merge(per_task, on=["target_suite", "target_task_id"], how="left", validate="many_to_one")
    c1 = probe[["target_suite", "target_task_id", "episode_id", "r2"]].rename(
        columns={"episode_id": "c1_trajectory_id", "r2": "C1"}
    )
    joined = joined.merge(
        c1,
        on=["target_suite", "target_task_id", "c1_trajectory_id"],
        how="left",
        validate="many_to_one",
    )
    if joined[["C1", "C_all"]].isna().any().any():
        missing = joined.loc[joined.C1.isna() | joined.C_all.isna()]
        raise ValueError(f"Could not match all D1 trajectory IDs to probe episodes:\n{missing}")
    d1 = joined[joined.requested_data_budget == "1"]
    if not (d1.actual_num_demos == 1).all():
        raise ValueError("A requested 1-demo cell did not contain exactly one trajectory.")
    return joined, probe


def centered(frame: pd.DataFrame, columns: tuple[str, str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        output[column] = output[column] - output.groupby("target_suite")[column].transform("mean")
    return output


def correlation_record(name: str, x_name: str, y_name: str, frame: pd.DataFrame, mode: str) -> dict:
    selected = frame[["target_suite", x_name, y_name]].dropna()
    if mode == "suite_centered":
        selected = centered(selected, (x_name, y_name))
    if len(selected) < 3 or selected[x_name].nunique() < 2 or selected[y_name].nunique() < 2:
        pearson = spearman = math.nan
    else:
        pearson = float(stats.pearsonr(selected[x_name], selected[y_name]).statistic)
        spearman = float(stats.spearmanr(selected[x_name], selected[y_name]).statistic)
    return {
        "analysis": name,
        "x": x_name,
        "y": y_name,
        "mode": mode,
        "n": len(selected),
        "pearson": pearson,
        "spearman": spearman,
    }


def build_correlations(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    d1 = runs[runs.requested_data_budget == "1"].copy()
    c1_unique = d1.drop_duplicates(["target_suite", "target_task_id", "subset_seed"])
    for mode in ("raw", "suite_centered"):
        records.append(correlation_record("C1_vs_C_all", "C1", "C_all", c1_unique, mode))

    task_base = runs.groupby(["target_suite", "target_task_id"], as_index=False).agg(
        C_all=("C_all", "first"), zero_shot_success=("zero_shot_success", "first")
    )
    for mode in ("raw", "suite_centered"):
        records.append(correlation_record("C_all_vs_zero_shot", "C_all", "zero_shot_success", task_base, mode))

    for budget in (*PRIMARY_BUDGETS, "all_available"):
        selected = runs[runs.requested_data_budget == budget]
        for metric in ("adapted_target_success", "target_gain", "source_forgetting"):
            for x_name in ("C1", "C_all"):
                for mode in ("raw", "suite_centered"):
                    records.append(correlation_record(f"budget_{budget}", x_name, metric, selected, mode))

    wide = (
        runs[runs.requested_data_budget.isin(PRIMARY_BUDGETS)]
        .pivot(
            index=["target_suite", "target_task_id", "subset_seed", "C1", "C_all"],
            columns="requested_data_budget",
            values="adapted_target_success",
        )
        .reset_index()
    )
    for later in ("5", "10", "25"):
        wide[f"gain_{later}_minus_1"] = wide[later] - wide["1"]
        for x_name in ("C1", "C_all"):
            for mode in ("raw", "suite_centered"):
                records.append(
                    correlation_record(f"data_value_{later}_minus_1", x_name, f"gain_{later}_minus_1", wide, mode)
                )
    return pd.DataFrame(records), wide


def stage_trace(order: list[tuple[str, int]], table: pd.DataFrame) -> list[dict]:
    levels = dict.fromkeys(order, "1")
    lookup = table.set_index(["target_suite", "target_task_id", "requested_data_budget"])

    def summarize() -> dict:
        rows = [lookup.loc[(*task, level)] for task, level in levels.items()]
        return {
            "total_demos": sum(int(level) for level in levels.values()),
            "mean_target_success": float(np.mean([row.adapted_target_success for row in rows])),
            "mean_source_forgetting": float(np.nanmean([row.source_forgetting for row in rows]))
            if any(pd.notna(row.source_forgetting) for row in rows)
            else math.nan,
            "allocation": json.dumps({f"{suite}:{task_id}": level for (suite, task_id), level in levels.items()}),
        }

    output = [summarize()]
    for old, new in (("1", "5"), ("5", "10"), ("10", "25")):
        for task in order:
            if levels[task] != old:
                raise AssertionError("Stage-wise allocation state is inconsistent.")
            levels[task] = new
            output.append(summarize())
    return output


def oracle_trace(table: pd.DataFrame, budgets: list[int]) -> list[dict]:
    tasks = sorted(set(zip(table.target_suite, table.target_task_id, strict=False)))
    lookup = table.set_index(["target_suite", "target_task_id", "requested_data_budget"])
    states: dict[int, tuple[float, dict[tuple[str, int], str]]] = {0: (0.0, {})}
    for task in tasks:
        next_states = {}
        for cost, (value, allocation) in states.items():
            for level in PRIMARY_BUDGETS:
                new_cost = cost + int(level)
                new_value = value + float(lookup.loc[(*task, level)].adapted_target_success)
                previous = next_states.get(new_cost)
                if previous is None or new_value > previous[0]:
                    next_states[new_cost] = (new_value, {**allocation, task: level})
        states = next_states
    output = []
    for budget in budgets:
        if budget not in states:
            raise ValueError(f"No exact oracle allocation exists at total budget {budget}.")
        value, allocation = states[budget]
        rows = [lookup.loc[(*task, level)] for task, level in allocation.items()]
        output.append(
            {
                "total_demos": budget,
                "mean_target_success": value / len(tasks),
                "mean_source_forgetting": float(np.nanmean([row.source_forgetting for row in rows]))
                if any(pd.notna(row.source_forgetting) for row in rows)
                else math.nan,
                "allocation": json.dumps(
                    {f"{suite}:{task_id}": level for (suite, task_id), level in allocation.items()}
                ),
            }
        )
    return output


def allocation_analysis(
    runs: pd.DataFrame, *, permutations: int, random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = runs[runs.requested_data_budget.isin(PRIMARY_BUDGETS)].copy()
    seeds = sorted(primary.subset_seed.unique())
    policy_rows = []
    random_by_seed = {}
    for seed in seeds:
        table = primary[primary.subset_seed == seed]
        counts = table.groupby(["target_suite", "target_task_id"]).requested_data_budget.nunique()
        if len(counts) != 12 or not (counts == 4).all():
            raise ValueError(f"Seed {seed} does not have all 12 tasks x 4 primary budgets.")
        c1 = table.groupby(["target_suite", "target_task_id"]).C1.first()
        guided_order = list(c1.sort_values(kind="stable").index)
        reverse_order = list(reversed(guided_order))
        guided = stage_trace(guided_order, table)
        reverse = stage_trace(reverse_order, table)
        budgets = [row["total_demos"] for row in guided]
        oracle = oracle_trace(table, budgets)
        for policy, rows in (("C1_guided", guided), ("reverse_C1", reverse), ("oracle", oracle)):
            policy_rows.extend({**row, "policy": policy, "subset_seed": seed} for row in rows)

        rng = np.random.default_rng(random_seed + int(seed))
        tasks = list(c1.index)
        permutations_array = np.vstack([rng.permutation(len(tasks)) for _ in range(permutations)])
        lookup = table.set_index(["target_suite", "target_task_id", "requested_data_budget"])
        success = np.asarray(
            [[lookup.loc[(*task, level)].adapted_target_success for level in PRIMARY_BUDGETS] for task in tasks]
        )
        forgetting_values = np.asarray(
            [[lookup.loc[(*task, level)].source_forgetting for level in PRIMARY_BUDGETS] for task in tasks]
        )

        def random_stage_values(
            values: np.ndarray,
            *,
            budget_count: int = len(budgets),
            orderings: np.ndarray = permutations_array,
            task_count: int = len(tasks),
        ) -> np.ndarray:
            if np.isnan(values).all():
                return np.full((permutations, budget_count), np.nan)
            columns = [np.full(permutations, np.nanmean(values[:, 0]))]
            current = columns[0]
            for old_index, new_index in ((0, 1), (1, 2), (2, 3)):
                deltas = values[:, new_index] - values[:, old_index]
                ordered_deltas = deltas[orderings] / task_count
                stage = current[:, None] + np.cumsum(ordered_deltas, axis=1)
                columns.extend(stage[:, index] for index in range(stage.shape[1]))
                current = stage[:, -1]
            return np.column_stack(columns)

        random_scores = random_stage_values(success)
        random_forgetting = random_stage_values(forgetting_values)
        random_by_seed[seed] = (random_scores, random_forgetting)

    trace = pd.DataFrame(policy_rows)
    aggregate = trace.groupby(["policy", "total_demos"], as_index=False).agg(
        mean_target_success=("mean_target_success", "mean"),
        mean_source_forgetting=("mean_source_forgetting", "mean"),
    )
    aggregate["subset_seed"] = "mean"
    aggregate["allocation"] = ""
    trace = pd.concat([trace, aggregate[trace.columns]], ignore_index=True)

    scores = np.mean([value[0] for value in random_by_seed.values()], axis=0)
    forgetting_arrays = [value[1] for value in random_by_seed.values() if not np.isnan(value[1]).all()]
    forgetting = np.mean(forgetting_arrays, axis=0) if forgetting_arrays else np.full_like(scores, np.nan)
    budgets = sorted(trace[trace.policy == "C1_guided"].total_demos.unique())
    random_summary = pd.DataFrame(
        {
            "total_demos": budgets,
            "random_mean_target_success": scores.mean(axis=0),
            "random_std_target_success": scores.std(axis=0),
            "random_p05_target_success": np.quantile(scores, 0.05, axis=0),
            "random_p95_target_success": np.quantile(scores, 0.95, axis=0),
            "random_mean_source_forgetting": np.nanmean(forgetting, axis=0),
        }
    )
    guided = trace[(trace.policy == "C1_guided") & (trace.subset_seed == "mean")].sort_values("total_demos")
    reverse = trace[(trace.policy == "reverse_C1") & (trace.subset_seed == "mean")].sort_values("total_demos")
    oracle = trace[(trace.policy == "oracle") & (trace.subset_seed == "mean")].sort_values("total_demos")
    random_summary["c1_guided_target_success"] = guided.mean_target_success.to_numpy()
    random_summary["reverse_c1_target_success"] = reverse.mean_target_success.to_numpy()
    random_summary["oracle_target_success"] = oracle.mean_target_success.to_numpy()
    random_summary["guided_minus_random"] = (
        random_summary.c1_guided_target_success - random_summary.random_mean_target_success
    )
    random_summary["guided_minus_reverse"] = (
        random_summary.c1_guided_target_success - random_summary.reverse_c1_target_success
    )
    random_summary["gap_to_oracle"] = random_summary.oracle_target_success - random_summary.c1_guided_target_success
    random_summary["guided_random_percentile"] = [
        100 * float(np.mean(scores[:, index] <= value))
        for index, value in enumerate(random_summary.c1_guided_target_success)
    ]
    return trace, random_summary


def task_summary(runs: pd.DataFrame) -> pd.DataFrame:
    base = runs.groupby(["target_suite", "target_task_id"], as_index=False).agg(
        C_all=("C_all", "first"),
        C_all_std=("C_all_std", "first"),
        num_probe_demos=("num_probe_demos", "first"),
        zero_shot_success=("zero_shot_success", "first"),
        total_available_demos=("total_available_demos", "first"),
    )
    for budget in (*PRIMARY_BUDGETS, "all_available"):
        selected = runs[runs.requested_data_budget == budget]
        values = selected.groupby(["target_suite", "target_task_id"], as_index=False).agg(
            **{
                f"SR_{budget}": ("adapted_target_success", "mean"),
                f"SR_{budget}_std": ("adapted_target_success", "std"),
            }
        )
        base = base.merge(values, on=["target_suite", "target_task_id"], how="left")
    return base.sort_values(["target_suite", "target_task_id"])


def make_plots(
    output: pathlib.Path,
    runs: pd.DataFrame,
    marginal: pd.DataFrame,
    correlations: pd.DataFrame,
    random_summary: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    d1 = runs[runs.requested_data_budget == "1"].drop_duplicates(["target_suite", "target_task_id", "subset_seed"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for suite in CONTROLLED_SUITES:
        selected = d1[d1.target_suite == suite]
        axes[0].scatter(
            selected.C_all, selected.C1, s=45, alpha=0.8, label=SUITE_LABELS[suite], color=SUITE_COLORS[suite]
        )
        axes[1].scatter(selected.C1, selected.adapted_target_success, s=45, alpha=0.8, color=SUITE_COLORS[suite])
    axes[0].set(
        xlabel="C_all (mean per-demo R²)", ylabel="seed-matched C1 (R²)", title="One-demo compatibility stability"
    )
    axes[1].set(
        xlabel="seed-matched C1 (R²)", ylabel="LoRA-1 target success", title="Compatibility vs one-demo adaptation"
    )
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "A_compatibility_and_one_shot_relation.png", dpi=200)
    fig.savefig(output / "A_compatibility_and_one_shot_relation.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), sharex=True)
    for axis, later in zip(axes, ("5", "10", "25"), strict=True):
        field = f"gain_{later}_minus_1"
        for suite in CONTROLLED_SUITES:
            selected = marginal[marginal.target_suite == suite]
            axis.scatter(selected.C1, selected[field], s=40, alpha=0.8, color=SUITE_COLORS[suite])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=f"LoRA SR({later}) - SR(1)", xlabel="seed-matched C1")
    axes[0].set_ylabel("marginal target-success gain")
    fig.tight_layout()
    fig.savefig(output / "B_compatibility_vs_data_value.png", dpi=200)
    fig.savefig(output / "B_compatibility_vs_data_value.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    x = random_summary.total_demos.to_numpy()
    axis.fill_between(
        x,
        random_summary.random_p05_target_success,
        random_summary.random_p95_target_success,
        color="#BBBBBB",
        alpha=0.35,
        label="random priority 5-95%",
    )
    axis.plot(x, random_summary.random_mean_target_success, "--", color="#666666", label="random mean")
    axis.plot(x, random_summary.c1_guided_target_success, color="#0072B2", linewidth=2, label="low-C1 first")
    axis.plot(x, random_summary.reverse_c1_target_success, color="#D55E00", linewidth=1.8, label="high-C1 first")
    axis.plot(x, random_summary.oracle_target_success, color="#009E73", linewidth=1.8, label="oracle")
    for anchor in (12, 60, 120, 300):
        axis.axvline(anchor, color="#DDDDDD", linewidth=0.8)
    axis.set(
        xlabel="total demonstrations across 12 targets",
        ylabel="mean target success",
        title="Split-B fixed-budget LoRA allocation (mean over seeds)",
    )
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "C_fixed_budget_allocation.png", dpi=200)
    fig.savefig(output / "C_fixed_budget_allocation.pdf")
    plt.close(fig)

    selected = correlations[
        (correlations["x"] == "C1")
        & correlations.analysis.isin(
            ["budget_1", "data_value_5_minus_1", "data_value_10_minus_1", "data_value_25_minus_1"]
        )
        & correlations.y.isin(["adapted_target_success", "gain_5_minus_1", "gain_10_minus_1", "gain_25_minus_1"])
    ].copy()
    selected["label"] = (
        selected.analysis.str.replace("budget_1", "SR(1)")
        .str.replace("data_value_", "delta SR ")
        .str.replace("_minus_1", "-1")
    )
    pivot = selected.pivot(index="label", columns="mode", values="spearman")
    fig, axis = plt.subplots(figsize=(6.8, 3.5))
    image = axis.imshow(pivot.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns)
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            axis.text(column, row, f"{pivot.iloc[row, column]:.2f}", ha="center", va="center")
    axis.set_title("Spearman correlations with seed-matched C1")
    fig.colorbar(image, ax=axis, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(output / "D_relation_correlation_summary.png", dpi=200)
    fig.savefig(output / "D_relation_correlation_summary.pdf")
    plt.close(fig)


def write_report(
    output: pathlib.Path,
    correlations: pd.DataFrame,
    random_summary: pd.DataFrame,
    runs: pd.DataFrame,
) -> None:
    def rho(analysis: str, y: str, mode: str = "raw") -> float:
        rows = correlations[
            (correlations.analysis == analysis)
            & (correlations.x == "C1")
            & (correlations.y == y)
            & (correlations["mode"] == mode)
        ]
        return float(rows.iloc[0].spearman)

    low_range = random_summary[(random_summary.total_demos > 12) & (random_summary.total_demos < 120)]
    fraction_above_random = float((low_range.guided_minus_random > 0).mean())
    fraction_above_reverse = float((low_range.guided_minus_reverse > 0).mean())
    mean_advantage = float(low_range.guided_minus_random.mean())
    all_available = runs[runs.requested_data_budget == "all_available"]
    all_sr = float(all_available.adapted_target_success.mean())
    report = f"""# Split-B progress compatibility x LoRA validation

This analysis is independent of Split A. It uses the frozen no-proprio V3 probe trained only on
Split-B's 18 source tasks, the exact seed-matched D1 trajectory for every C1 value, and the formal
50-trial Spatial/Object/Goal LoRA results. LIBERO-10 is excluded.

## Main signals

- C1 vs C_all: Spearman **{rho("C1_vs_C_all", "C_all"):.3f}** raw and
  **{rho("C1_vs_C_all", "C_all", "suite_centered"):.3f}** after suite centering.
- C1 vs LoRA-1 target SR: Spearman **{rho("budget_1", "adapted_target_success"):.3f}** raw and
  **{rho("budget_1", "adapted_target_success", "suite_centered"):.3f}** suite-centered.
- C1 vs value of 5 rather than 1 demo: Spearman **{rho("data_value_5_minus_1", "gain_5_minus_1"):.3f}**
  raw and **{rho("data_value_5_minus_1", "gain_5_minus_1", "suite_centered"):.3f}** suite-centered.
- C1 vs value of 10 rather than 1 demo: Spearman **{rho("data_value_10_minus_1", "gain_10_minus_1"):.3f}**
  raw and **{rho("data_value_10_minus_1", "gain_10_minus_1", "suite_centered"):.3f}** suite-centered.
- Over the useful pre-10-demo range, low-C1-first beats the random mean at
  **{fraction_above_random:.0%}** of reachable budgets (mean ΔSR **{mean_advantage:+.3f}**) and
  beats reverse-C1 at **{fraction_above_reverse:.0%}** of budgets.
- Data-rich `all_available` reference mean target SR: **{all_sr:.3f}**. Its actual trajectory counts
  remain recorded per task and are never relabeled as 50 demos.

## Interpretation rule

The compatibility-guided allocation claim is supported only if the low-C1-first curve is broadly
above both random and reverse-C1 before all tasks reach 10 demos, not merely at one cherry-picked
budget. Correlations are descriptive: the 12 target tasks are not independent of suite and three
subset seeds do not constitute three independent task samples. See `correlation_summary.csv`,
`allocation_summary.csv`, and the figures for counterexamples and suite-centered results.
"""
    (output / "REPORT.md").write_text(report)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_adaptation_runs(args.adaptation_root)
    runs, probe = attach_compatibility(runs, args.probe_per_demo)
    expected = 12 * (3 * len(PRIMARY_BUDGETS) + 1)
    if len(runs) != expected:
        raise ValueError(f"Expected {expected} controlled-suite cells, found {len(runs)}")
    correlations, marginal = build_correlations(runs)
    allocation_trace, allocation_summary = allocation_analysis(
        runs, permutations=args.random_permutations, random_seed=args.random_seed
    )
    summary = task_summary(runs)

    runs.to_csv(args.output_dir / "merged_run_level.csv", index=False)
    summary.to_csv(args.output_dir / "merged_task_level.csv", index=False)
    correlations.to_csv(args.output_dir / "correlation_summary.csv", index=False)
    marginal.to_csv(args.output_dir / "marginal_data_gains.csv", index=False)
    allocation_trace.to_csv(args.output_dir / "allocation_trace.csv", index=False)
    allocation_summary.to_csv(args.output_dir / "allocation_summary.csv", index=False)
    make_plots(args.output_dir, runs, marginal, correlations, allocation_summary)
    write_report(args.output_dir, correlations, allocation_summary, runs)
    manifest = {
        "adaptation_root": str(args.adaptation_root.resolve()),
        "probe_per_demo": str(args.probe_per_demo.resolve()),
        "controlled_suites": list(CONTROLLED_SUITES),
        "num_adaptation_cells": len(runs),
        "num_target_tasks": len(summary),
        "num_probe_demos": len(probe),
        "random_permutations": args.random_permutations,
        "c1_binding": "exact c1_trajectory_id from each seed's D1 train_manifest",
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote Split compatibility analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
