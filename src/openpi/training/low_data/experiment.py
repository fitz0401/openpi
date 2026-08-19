"""Frozen experiment config and deterministic subset helpers for low-data adaptation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import random
from typing import Any

from openpi.training.continual.subsample import resolve_task_string

LIBERO_SUITE_TASKS: dict[str, tuple[str, ...]] = {
    "libero_spatial": (
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "pick up the black bowl next to the ramekin and place it on the plate",
        "pick up the black bowl from table center and place it on the plate",
        "pick up the black bowl on the cookie box and place it on the plate",
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "pick up the black bowl on the ramekin and place it on the plate",
        "pick up the black bowl next to the cookie box and place it on the plate",
        "pick up the black bowl on the stove and place it on the plate",
        "pick up the black bowl next to the plate and place it on the plate",
        "pick up the black bowl on the wooden cabinet and place it on the plate",
    ),
    "libero_goal": (
        "open the middle drawer of the cabinet",
        "put the bowl on the stove",
        "put the wine bottle on top of the cabinet",
        "open the top drawer and put the bowl inside",
        "put the bowl on top of the cabinet",
        "push the plate to the front of the stove",
        "put the cream cheese in the bowl",
        "turn on the stove",
        "put the bowl on the plate",
        "put the wine bottle on the rack",
    ),
    "libero_object": (
        "pick up the alphabet soup and place it in the basket",
        "pick up the cream cheese and place it in the basket",
        "pick up the salad dressing and place it in the basket",
        "pick up the bbq sauce and place it in the basket",
        "pick up the ketchup and place it in the basket",
        "pick up the tomato sauce and place it in the basket",
        "pick up the butter and place it in the basket",
        "pick up the milk and place it in the basket",
        "pick up the chocolate pudding and place it in the basket",
        "pick up the orange juice and place it in the basket",
    ),
    "libero_10": (
        "put both the alphabet soup and the tomato sauce in the basket",
        "put both the cream cheese box and the butter in the basket",
        "turn on the stove and put the moka pot on it",
        "put the black bowl in the bottom drawer of the cabinet and close it",
        "put the white mug on the left plate and put the yellow and white mug on the right plate",
        "pick up the book and place it in the back compartment of the caddy",
        "put the white mug on the plate and put the chocolate pudding to the right of the plate",
        "put both the alphabet soup and the cream cheese box in the basket",
        "put both moka pots on the stove",
        "put the yellow and white mug in the microwave and close it",
    ),
}

PRIMARY_DATA_BUDGETS = ("1", "5", "10", "25")
ALL_AVAILABLE_BUDGET = "all_available"
FINAL_DATA_BUDGETS = (*PRIMARY_DATA_BUDGETS, ALL_AVAILABLE_BUDGET)
OFFICIAL_ROLLOUT_HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


@dataclasses.dataclass(frozen=True)
class SourceRecipe:
    optimizer_steps: int
    batch_size: int = 32


@dataclasses.dataclass(frozen=True)
class AdaptationRecipe:
    data_budgets: tuple[str, ...]
    methods: tuple[str, ...]
    method_data_budgets: dict[str, tuple[str, ...]]
    seeds: tuple[int, ...]
    suite_seed_overrides: dict[str, tuple[int, ...]]
    data_budget_seed_overrides: dict[str, tuple[int, ...]]
    effective_epochs: float
    per_device_batch_size: int = 32
    world_size: int = 1
    gradient_accumulation_steps: int = 1
    hard_max_steps: int | None = None

    def validate(self) -> None:
        if self.data_budgets != FINAL_DATA_BUDGETS:
            raise ValueError(f"Final protocol requires data_budgets={FINAL_DATA_BUDGETS}.")
        if not self.methods or len(self.methods) != len(set(self.methods)) or not set(self.methods) <= {"full", "lora"}:
            raise ValueError("Adaptation methods must be a non-empty unique subset of {'lora', 'full'}.")
        supported_method_budgets = {
            "lora": FINAL_DATA_BUDGETS,
            "full": ("1", "10", ALL_AVAILABLE_BUDGET),
        }
        expected_method_budgets = {method: supported_method_budgets[method] for method in self.methods}
        if self.method_data_budgets != expected_method_budgets:
            raise ValueError(f"Selected methods require method_data_budgets={expected_method_budgets}.")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Adaptation seeds must be non-empty and unique.")
        for suite, seeds in self.suite_seed_overrides.items():
            if not seeds or len(seeds) != len(set(seeds)):
                raise ValueError(f"Seed override for {suite} must be non-empty and unique.")
            if not set(seeds) <= set(self.seeds):
                raise ValueError(f"Seed override for {suite} must be a subset of global seeds={self.seeds}.")
        for data_budget, seeds in self.data_budget_seed_overrides.items():
            if data_budget not in self.data_budgets:
                raise ValueError(f"Seed override references unknown data budget {data_budget!r}.")
            if not seeds or len(seeds) != len(set(seeds)):
                raise ValueError(f"Seed override for {data_budget} must be non-empty and unique.")
            if not set(seeds) <= set(self.seeds):
                raise ValueError(f"Seed override for {data_budget} must be a subset of global seeds={self.seeds}.")
        if self.effective_epochs <= 0:
            raise ValueError("adaptation.effective_epochs must be positive.")
        if self.per_device_batch_size <= 0 or self.world_size <= 0:
            raise ValueError("Batch size and world size must be positive.")
        if self.gradient_accumulation_steps != 1:
            raise ValueError("The existing trainer does not support gradient accumulation; use 1.")
        if self.hard_max_steps is not None and self.hard_max_steps <= 0:
            raise ValueError("adaptation.hard_max_steps must be null or positive.")

    @property
    def global_batch_size(self) -> int:
        return self.per_device_batch_size * self.world_size * self.gradient_accumulation_steps

    def data_budgets_for_method(self, method: str) -> tuple[str, ...]:
        return self.method_data_budgets[method]

    def seeds_for(self, suite: str, data_budget: str) -> tuple[int, ...]:
        suite_seeds = self.suite_seed_overrides.get(suite, self.seeds)
        budget_seeds = self.data_budget_seed_overrides.get(data_budget, self.seeds)
        selected = tuple(seed for seed in self.seeds if seed in suite_seeds and seed in budget_seeds)
        if not selected:
            raise ValueError(f"Seed overrides leave no valid seed for suite={suite}, data_budget={data_budget}.")
        return selected

    def calculate_optimizer_steps(self, *, num_training_examples: int) -> int:
        if num_training_examples <= 0:
            raise ValueError("num_training_examples must be positive.")
        calculated = math.ceil(self.effective_epochs * num_training_examples / self.global_batch_size)
        if self.hard_max_steps is not None and calculated > self.hard_max_steps:
            raise ValueError(
                "Protocol violation: calculated optimizer budget "
                f"{calculated} exceeds hard_max_steps={self.hard_max_steps}; refusing to truncate."
            )
        return calculated


@dataclasses.dataclass(frozen=True)
class EvalRecipe:
    target_num_trials_by_suite: dict[str, int] = dataclasses.field(
        default_factory=lambda: {
            "libero_spatial": 50,
            "libero_object": 50,
            "libero_goal": 50,
            "libero_10": 25,
        }
    )
    source_retention_num_trials: int = 25
    source_retention_subset_seeds: tuple[int, ...] = (0,)
    source_retention_disabled_target_suites: tuple[str, ...] = ("libero_10",)
    replan_steps: int = 5
    rollout_horizons: dict[str, int] = dataclasses.field(default_factory=lambda: dict(OFFICIAL_ROLLOUT_HORIZONS))
    allow_nonstandard_rollout_horizons: bool = False

    def validate(self) -> None:
        if self.source_retention_num_trials <= 0 or self.replan_steps <= 0:
            raise ValueError("Evaluation trial and replanning counts must be positive.")
        if set(self.target_num_trials_by_suite) != set(OFFICIAL_ROLLOUT_HORIZONS):
            raise ValueError("target_num_trials_by_suite must define every official LIBERO suite exactly once.")
        if any(num_trials <= 0 for num_trials in self.target_num_trials_by_suite.values()):
            raise ValueError("Target evaluation trial counts must be positive.")
        if not self.source_retention_subset_seeds or len(set(self.source_retention_subset_seeds)) != len(
            self.source_retention_subset_seeds
        ):
            raise ValueError("source_retention_subset_seeds must be non-empty and unique.")
        unknown_disabled = set(self.source_retention_disabled_target_suites) - set(OFFICIAL_ROLLOUT_HORIZONS)
        if unknown_disabled:
            raise ValueError(f"Unknown source-retention-disabled target suites: {sorted(unknown_disabled)}")
        if not self.allow_nonstandard_rollout_horizons and self.rollout_horizons != OFFICIAL_ROLLOUT_HORIZONS:
            raise ValueError(
                f"Final protocol requires official rollout_horizons={OFFICIAL_ROLLOUT_HORIZONS}; "
                "set allow_nonstandard_rollout_horizons=true only for an explicitly labelled debug run."
            )
        missing = set(OFFICIAL_ROLLOUT_HORIZONS) - set(self.rollout_horizons)
        if missing or any(horizon <= 0 for horizon in self.rollout_horizons.values()):
            raise ValueError(f"Evaluation rollout horizons are missing or invalid: {sorted(missing)}")

    def rollout_horizon(self, suite: str) -> int:
        return self.rollout_horizons[suite]

    def target_num_trials(self, suite: str) -> int:
        return self.target_num_trials_by_suite[suite]

    def should_evaluate_source_retention(self, *, target_suite: str, subset_seed: int) -> bool:
        return (
            target_suite not in self.source_retention_disabled_target_suites
            and subset_seed in self.source_retention_subset_seeds
        )

    def protocol_manifest(self) -> dict[str, object]:
        return {
            "target_num_trials_by_suite": dict(self.target_num_trials_by_suite),
            "source_retention_num_trials": self.source_retention_num_trials,
            "source_retention_subset_seeds": list(self.source_retention_subset_seeds),
            "source_retention_disabled_target_suites": list(self.source_retention_disabled_target_suites),
            "replan_steps": self.replan_steps,
            "rollout_horizons": dict(self.rollout_horizons),
        }

    @property
    def protocol_id(self) -> str:
        expected = {
            "target_num_trials_by_suite": {
                "libero_spatial": 50,
                "libero_object": 50,
                "libero_goal": 50,
                "libero_10": 25,
            },
            "source_retention_num_trials": 25,
            "source_retention_subset_seeds": [0],
            "source_retention_disabled_target_suites": ["libero_10"],
            "replan_steps": 5,
            "rollout_horizons": dict(OFFICIAL_ROLLOUT_HORIZONS),
        }
        if self.protocol_manifest() == expected:
            return "sog_target50_l10_target25_retention25_seed0_no_l10_retention"
        return "custom_evaluation_protocol"


@dataclasses.dataclass(frozen=True, order=True)
class TaskRef:
    suite: str
    task_id: int

    @property
    def key(self) -> str:
        return f"{self.suite}:{self.task_id}"


@dataclasses.dataclass(frozen=True)
class SuiteSplit:
    suite: str
    source_task_ids: tuple[int, ...]
    target_task_ids: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class PilotExperimentConfig:
    schema_version: int
    split_id: str
    task_order_index: int
    task_splits: tuple[SuiteSplit, ...]
    base_checkpoint: str
    checkpoint_root: str
    results_root: str
    source: SourceRecipe
    adaptation: AdaptationRecipe
    evaluation: EvalRecipe

    def validate(self) -> None:
        if self.schema_version != 3:
            raise ValueError(
                f"schema_version={self.schema_version} is historical and cannot launch future runs; "
                "the frozen protocol requires schema_version=3."
            )
        if self.task_order_index != 0:
            raise ValueError("Low-data task IDs currently use LIBERO task_order_index=0 only.")
        if not self.task_splits:
            raise ValueError("At least one task split is required.")
        suite_names = [split.suite for split in self.task_splits]
        if len(suite_names) != len(set(suite_names)):
            raise ValueError("Each suite may appear only once in task_splits.")
        for split in self.task_splits:
            if split.suite not in LIBERO_SUITE_TASKS:
                raise ValueError(f"Unsupported suite {split.suite!r}; choose from {sorted(LIBERO_SUITE_TASKS)}")
            source = set(split.source_task_ids)
            target = set(split.target_task_ids)
            if len(source) != len(split.source_task_ids) or len(target) != len(split.target_task_ids):
                raise ValueError(f"Duplicate task IDs in {split.suite}.")
            if source & target:
                raise ValueError(f"Source/target IDs overlap in {split.suite}: {sorted(source & target)}")
            valid_ids = set(range(len(LIBERO_SUITE_TASKS[split.suite])))
            if not source | target <= valid_ids:
                raise ValueError(f"{split.suite} task IDs must be in {sorted(valid_ids)}")
        source = set(self.source_task_refs())
        target = set(self.target_task_refs())
        if not source or not target:
            raise ValueError("Source and target task sets must both be non-empty.")
        if self.source.optimizer_steps <= 0:
            raise ValueError("Source optimizer-step budget must be positive.")
        unknown_seed_override_suites = set(self.adaptation.suite_seed_overrides) - set(suite_names)
        if unknown_seed_override_suites:
            raise ValueError(f"Seed overrides reference unknown suites: {sorted(unknown_seed_override_suites)}")
        unknown_retention_seeds = set(self.evaluation.source_retention_subset_seeds) - set(self.adaptation.seeds)
        if unknown_retention_seeds:
            raise ValueError(f"Source-retention evaluation references unknown seeds: {sorted(unknown_retention_seeds)}")
        self.adaptation.validate()
        self.evaluation.validate()

    @property
    def suite(self) -> str:
        """Backward-compatible single-suite accessor."""
        if len(self.task_splits) != 1:
            raise ValueError("This experiment contains multiple suites.")
        return self.task_splits[0].suite

    @property
    def source_task_ids(self) -> tuple[int, ...]:
        if len(self.task_splits) != 1:
            raise ValueError("Use source_task_refs() for a multi-suite experiment.")
        return self.task_splits[0].source_task_ids

    @property
    def target_task_ids(self) -> tuple[int, ...]:
        if len(self.task_splits) != 1:
            raise ValueError("Use target_task_refs() for a multi-suite experiment.")
        return self.task_splits[0].target_task_ids

    def task_string(self, suite: str, task_id: int) -> str:
        return LIBERO_SUITE_TASKS[suite][task_id]

    def source_task_refs(self) -> tuple[TaskRef, ...]:
        return tuple(TaskRef(split.suite, task_id) for split in self.task_splits for task_id in split.source_task_ids)

    def target_task_refs(self) -> tuple[TaskRef, ...]:
        return tuple(TaskRef(split.suite, task_id) for split in self.task_splits for task_id in split.target_task_ids)

    def source_task_ids_by_suite(self) -> dict[str, list[int]]:
        return {split.suite: list(split.source_task_ids) for split in self.task_splits if split.source_task_ids}

    def target_task_ids_by_suite(self) -> dict[str, list[int]]:
        return {split.suite: list(split.target_task_ids) for split in self.task_splits if split.target_task_ids}

    def task_strings(self, refs: tuple[TaskRef, ...]) -> tuple[str, ...]:
        return tuple(self.task_string(ref.suite, ref.task_id) for ref in refs)


def load_experiment_config(path: str | pathlib.Path) -> PilotExperimentConfig:
    path = pathlib.Path(path)
    raw = json.loads(path.read_text())
    schema_version = int(raw["schema_version"])
    if schema_version != 3:
        raise ValueError(
            f"schema_version={schema_version} in {path} is historical and cannot launch future runs; "
            "the frozen protocol requires schema_version=3."
        )
    if "task_splits" in raw:
        task_splits = tuple(
            SuiteSplit(
                suite=str(split["suite"]),
                source_task_ids=tuple(int(x) for x in split.get("source_task_ids", [])),
                target_task_ids=tuple(int(x) for x in split.get("target_task_ids", [])),
            )
            for split in raw["task_splits"]
        )
    else:
        task_splits = (
            SuiteSplit(
                suite=str(raw["suite"]),
                source_task_ids=tuple(int(x) for x in raw["source_task_ids"]),
                target_task_ids=tuple(int(x) for x in raw["target_task_ids"]),
            ),
        )
    config = PilotExperimentConfig(
        schema_version=schema_version,
        split_id=str(raw["split_id"]),
        task_order_index=int(raw.get("task_order_index", 0)),
        task_splits=task_splits,
        base_checkpoint=str(raw["base_checkpoint"]),
        checkpoint_root=str(raw.get("checkpoint_root", "./checkpoints/low_data_pilot")),
        results_root=str(raw.get("results_root", "./results/low_data_pilot")),
        source=SourceRecipe(**raw["source"]),
        adaptation=AdaptationRecipe(
            data_budgets=tuple(str(x) for x in raw["adaptation"]["data_budgets"]),
            methods=tuple(str(x) for x in raw["adaptation"]["methods"]),
            method_data_budgets={
                str(method): tuple(str(x) for x in budgets)
                for method, budgets in raw["adaptation"]["method_data_budgets"].items()
            },
            seeds=tuple(int(x) for x in raw["adaptation"]["seeds"]),
            suite_seed_overrides={
                str(suite): tuple(int(x) for x in seeds)
                for suite, seeds in raw["adaptation"].get("suite_seed_overrides", {}).items()
            },
            data_budget_seed_overrides={
                str(data_budget): tuple(int(x) for x in seeds)
                for data_budget, seeds in raw["adaptation"].get("data_budget_seed_overrides", {}).items()
            },
            effective_epochs=float(raw["adaptation"]["effective_epochs"]),
            per_device_batch_size=int(raw["adaptation"].get("per_device_batch_size", 32)),
            world_size=int(raw["adaptation"].get("world_size", 1)),
            gradient_accumulation_steps=int(raw["adaptation"].get("gradient_accumulation_steps", 1)),
            hard_max_steps=(
                int(raw["adaptation"]["hard_max_steps"])
                if raw["adaptation"].get("hard_max_steps") is not None
                else None
            ),
        ),
        evaluation=EvalRecipe(
            target_num_trials_by_suite={
                str(suite): int(num_trials)
                for suite, num_trials in raw["evaluation"]["target_num_trials_by_suite"].items()
            },
            source_retention_num_trials=int(raw["evaluation"]["source_retention_num_trials"]),
            source_retention_subset_seeds=tuple(
                int(seed) for seed in raw["evaluation"]["source_retention_subset_seeds"]
            ),
            source_retention_disabled_target_suites=tuple(
                str(suite) for suite in raw["evaluation"]["source_retention_disabled_target_suites"]
            ),
            replan_steps=int(raw.get("evaluation", {}).get("replan_steps", 5)),
            rollout_horizons={
                str(suite): int(horizon)
                for suite, horizon in raw.get("evaluation", {})
                .get("rollout_horizons", OFFICIAL_ROLLOUT_HORIZONS)
                .items()
            },
            allow_nonstandard_rollout_horizons=bool(
                raw.get("evaluation", {}).get("allow_nonstandard_rollout_horizons", False)
            ),
        ),
    )
    config.validate()
    return config


def target_grid(config: PilotExperimentConfig) -> list[tuple[str, int, str, str, int]]:
    """Return deterministic (suite, target, method, requested_data_budget, seed) cells."""
    return [
        (target.suite, target.task_id, method, data_budget, seed)
        for target in config.target_task_refs()
        for method in config.adaptation.methods
        for data_budget in config.adaptation.data_budgets_for_method(method)
        for seed in config.adaptation.seeds_for(target.suite, data_budget)
    ]


def evaluation_workload(config: PilotExperimentConfig) -> dict[str, object]:
    """Summarize the exact rollout workload implied by the Stage-B grid."""
    target_rollouts_by_suite = dict.fromkeys(OFFICIAL_ROLLOUT_HORIZONS, 0)
    target_cells_by_suite = dict.fromkeys(OFFICIAL_ROLLOUT_HORIZONS, 0)
    retention_cells = 0
    for suite, _task_id, _method, _data_budget, seed in target_grid(config):
        target_cells_by_suite[suite] += 1
        target_rollouts_by_suite[suite] += config.evaluation.target_num_trials(suite)
        if config.evaluation.should_evaluate_source_retention(target_suite=suite, subset_seed=seed):
            retention_cells += 1

    target_rollouts = sum(target_rollouts_by_suite.values())
    retention_rollouts = (
        retention_cells * len(config.source_task_refs()) * config.evaluation.source_retention_num_trials
    )
    target_max_env_steps = sum(
        target_rollouts_by_suite[suite] * config.evaluation.rollout_horizon(suite)
        for suite in OFFICIAL_ROLLOUT_HORIZONS
    )
    source_horizon_sum = sum(config.evaluation.rollout_horizon(ref.suite) for ref in config.source_task_refs())
    retention_max_env_steps = retention_cells * config.evaluation.source_retention_num_trials * source_horizon_sum
    return {
        "stage_b_cells": len(target_grid(config)),
        "target_cells_by_suite": target_cells_by_suite,
        "source_retention_cells": retention_cells,
        "target_rollouts_by_suite": target_rollouts_by_suite,
        "target_rollouts": target_rollouts,
        "source_retention_rollouts": retention_rollouts,
        "total_rollouts": target_rollouts + retention_rollouts,
        "target_max_env_steps": target_max_env_steps,
        "source_retention_max_env_steps": retention_max_env_steps,
        "total_max_env_steps": target_max_env_steps + retention_max_env_steps,
    }


def target_result_dir(
    config: PilotExperimentConfig,
    suite: str,
    task_id: int,
    method: str,
    requested_data_budget: str,
    seed: int,
) -> pathlib.Path:
    return (
        pathlib.Path(config.results_root)
        / config.split_id
        / "runs"
        / method
        / suite
        / f"task{task_id}"
        / f"budget{requested_data_budget}"
        / f"seed{seed}"
    )


def resolve_task_refs(meta: Any, config: PilotExperimentConfig, refs: tuple[TaskRef, ...]) -> tuple[str, ...]:
    return tuple(resolve_task_string(meta, config.task_string(ref.suite, ref.task_id)) for ref in refs)


def episodes_for_task(meta: Any, task: str) -> list[int]:
    canonical = resolve_task_string(meta, task)
    return sorted(ep_idx for ep_idx, episode in meta.episodes.items() if canonical in episode["tasks"])


def nested_episode_subsets(
    meta: Any,
    *,
    suite: str,
    task_id: int,
    task: str,
    seed: int,
    data_budgets: tuple[str, ...],
) -> tuple[list[int], dict[str, list[int]]]:
    """Shuffle once and use prefixes, including an explicitly labelled all-available set."""
    episodes = episodes_for_task(meta, task)
    if not episodes:
        raise ValueError(f"No trajectories found for {suite}:{task_id} ({task!r}).")
    seed_material = f"low-data-v1:{suite}:{task_id}:{seed}".encode()
    permutation_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    random.Random(permutation_seed).shuffle(episodes)
    subsets = {}
    for budget in data_budgets:
        if budget == ALL_AVAILABLE_BUDGET:
            subsets[budget] = list(episodes)
            continue
        count = int(budget)
        if count > len(episodes):
            raise ValueError(f"Requested D{count} for {suite}:{task_id}, but only {len(episodes)} trajectories exist.")
        subsets[budget] = episodes[:count]
    return episodes, subsets


def subset_statistics(
    meta: Any,
    episode_indices: list[int],
    *,
    optimizer_steps: int,
    global_batch_size: int,
) -> dict:
    num_transitions = sum(int(meta.episodes[index]["length"]) for index in episode_indices)
    num_training_examples = num_transitions  # one valid action-window start per selected LeRobot frame
    samples_seen = optimizer_steps * global_batch_size
    return {
        "num_selected_trajectories": len(episode_indices),
        "num_transitions": num_transitions,
        "num_training_examples": num_training_examples,
        "num_training_windows": num_training_examples,  # Legacy alias for historical analysis tools.
        "samples_seen": samples_seen,
        "optimizer_steps": optimizer_steps,
        "achieved_effective_epochs": samples_seen / max(num_training_examples, 1),
    }


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):  # noqa: UP038 -- imported by the Python 3.8 LIBERO env
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:  # noqa: UP038
        return value
    return repr(value)
