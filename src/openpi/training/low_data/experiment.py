"""Versioned experiment config and deterministic subset helpers for the low-data pilot."""

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
}


@dataclasses.dataclass(frozen=True)
class SourceRecipe:
    optimizer_steps: int
    batch_size: int = 32


@dataclasses.dataclass(frozen=True)
class TrainingBudget:
    name: str
    mode: str
    optimizer_steps: int | None = None
    max_steps: int | None = None
    max_effective_epochs: float | None = None
    min_steps: int = 1

    def validate(self) -> None:
        if not self.name or any(char.isspace() for char in self.name):
            raise ValueError("Training budget names must be non-empty and contain no whitespace.")
        if self.mode == "fixed_steps":
            if self.optimizer_steps is None or self.optimizer_steps <= 0:
                raise ValueError(f"Budget {self.name!r}: fixed_steps requires positive optimizer_steps.")
        elif self.mode == "capped_effective_epochs":
            if self.max_steps is None or self.max_steps <= 0:
                raise ValueError(f"Budget {self.name!r}: capped_effective_epochs requires positive max_steps.")
            if self.max_effective_epochs is None or self.max_effective_epochs <= 0:
                raise ValueError(
                    f"Budget {self.name!r}: capped_effective_epochs requires positive max_effective_epochs."
                )
            if self.min_steps <= 0 or self.min_steps > self.max_steps:
                raise ValueError(f"Budget {self.name!r}: require 0 < min_steps <= max_steps.")
        else:
            raise ValueError(
                f"Budget {self.name!r}: unsupported mode {self.mode!r}; "
                "choose 'fixed_steps' or 'capped_effective_epochs'."
            )

    def resolve_steps(self, *, num_training_windows: int, batch_size: int) -> int:
        if self.mode == "fixed_steps":
            assert self.optimizer_steps is not None
            return self.optimizer_steps
        assert self.max_steps is not None
        assert self.max_effective_epochs is not None
        epoch_steps = math.ceil(self.max_effective_epochs * num_training_windows / batch_size)
        return max(self.min_steps, min(self.max_steps, epoch_steps))


@dataclasses.dataclass(frozen=True)
class TargetRecipe:
    num_demos: tuple[int, ...]
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    budgets: tuple[TrainingBudget, ...]
    method_num_demos: dict[str, tuple[int, ...]]
    batch_size: int = 32

    def demos_for_method(self, method: str) -> tuple[int, ...]:
        return self.method_num_demos.get(method, self.num_demos)

    def budget(self, name: str | None) -> TrainingBudget:
        if name is None:
            if len(self.budgets) != 1:
                raise ValueError(f"Select one training budget from {[budget.name for budget in self.budgets]}")
            return self.budgets[0]
        matches = [budget for budget in self.budgets if budget.name == name]
        if not matches:
            raise ValueError(
                f"Unknown training budget {name!r}; choose from {[budget.name for budget in self.budgets]}"
            )
        return matches[0]


@dataclasses.dataclass(frozen=True)
class EvalRecipe:
    num_trials: int = 20
    max_steps: int = 280
    replan_steps: int = 5


@dataclasses.dataclass(frozen=True)
class PilotExperimentConfig:
    schema_version: int
    split_id: str
    suite: str
    task_order_index: int
    source_task_ids: tuple[int, ...]
    target_task_ids: tuple[int, ...]
    base_checkpoint: str
    checkpoint_root: str
    results_root: str
    source_results_dir: str | None
    source: SourceRecipe
    target: TargetRecipe
    evaluation: EvalRecipe

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported schema_version={self.schema_version}")
        if self.suite not in LIBERO_SUITE_TASKS:
            raise ValueError(f"Unsupported suite {self.suite!r}; choose from {sorted(LIBERO_SUITE_TASKS)}")
        if self.task_order_index != 0:
            raise ValueError("Pilot task IDs currently use LIBERO task_order_index=0 only.")
        source = set(self.source_task_ids)
        target = set(self.target_task_ids)
        if not source or not target:
            raise ValueError("Source and target task sets must both be non-empty.")
        if source & target:
            raise ValueError(f"Source/target IDs overlap: {sorted(source & target)}")
        valid_ids = set(range(len(LIBERO_SUITE_TASKS[self.suite])))
        if not source | target <= valid_ids:
            raise ValueError(f"Task IDs must be in {sorted(valid_ids)}")
        if any(n <= 0 for n in self.target.num_demos):
            raise ValueError("All num_demos values must be positive.")
        if tuple(sorted(set(self.target.num_demos))) != self.target.num_demos:
            raise ValueError("target.num_demos must be sorted and unique.")
        if not set(self.target.methods) <= {"full", "lora"}:
            raise ValueError("Pilot methods currently support only 'full' and 'lora'.")
        if set(self.target.method_num_demos) - set(self.target.methods):
            raise ValueError("method_num_demos contains a method not listed in target.methods.")
        for method in self.target.methods:
            demos = self.target.demos_for_method(method)
            if not demos or tuple(sorted(set(demos))) != demos:
                raise ValueError(f"Demo values for {method!r} must be non-empty, sorted, and unique.")
            if not set(demos) <= set(self.target.num_demos):
                raise ValueError(f"Demo values for {method!r} must be a subset of target.num_demos.")
        if self.source.optimizer_steps <= 0:
            raise ValueError("Source optimizer-step budget must be positive.")
        if not self.target.budgets:
            raise ValueError("At least one target training budget is required.")
        if len({budget.name for budget in self.target.budgets}) != len(self.target.budgets):
            raise ValueError("Target training budget names must be unique.")
        for budget in self.target.budgets:
            budget.validate()

    def task_string(self, task_id: int) -> str:
        return LIBERO_SUITE_TASKS[self.suite][task_id]

    def source_task_strings(self) -> tuple[str, ...]:
        return tuple(self.task_string(task_id) for task_id in self.source_task_ids)

    def target_task_strings(self) -> tuple[str, ...]:
        return tuple(self.task_string(task_id) for task_id in self.target_task_ids)

    def resolved_source_results_dir(self) -> pathlib.Path:
        if self.source_results_dir is not None:
            return pathlib.Path(self.source_results_dir)
        return pathlib.Path(self.results_root) / self.split_id / "source"


def _load_training_budgets(raw_target: dict[str, Any]) -> tuple[TrainingBudget, ...]:
    raw_budgets = raw_target.get("budgets")
    if raw_budgets is None:
        # Backward compatibility for the original pilot configs.
        return (
            TrainingBudget(
                name="fixed_steps",
                mode="fixed_steps",
                optimizer_steps=int(raw_target["optimizer_steps"]),
            ),
        )
    return tuple(
        TrainingBudget(
            name=str(raw["name"]),
            mode=str(raw["mode"]),
            optimizer_steps=int(raw["optimizer_steps"]) if raw.get("optimizer_steps") is not None else None,
            max_steps=int(raw["max_steps"]) if raw.get("max_steps") is not None else None,
            max_effective_epochs=(
                float(raw["max_effective_epochs"]) if raw.get("max_effective_epochs") is not None else None
            ),
            min_steps=int(raw.get("min_steps", 1)),
        )
        for raw in raw_budgets
    )


def load_experiment_config(path: str | pathlib.Path) -> PilotExperimentConfig:
    path = pathlib.Path(path)
    raw = json.loads(path.read_text())
    config = PilotExperimentConfig(
        schema_version=int(raw["schema_version"]),
        split_id=str(raw["split_id"]),
        suite=str(raw["suite"]),
        task_order_index=int(raw.get("task_order_index", 0)),
        source_task_ids=tuple(int(x) for x in raw["source_task_ids"]),
        target_task_ids=tuple(int(x) for x in raw["target_task_ids"]),
        base_checkpoint=str(raw["base_checkpoint"]),
        checkpoint_root=str(raw.get("checkpoint_root", "./checkpoints/low_data_pilot")),
        results_root=str(raw.get("results_root", "./results/low_data_pilot")),
        source_results_dir=str(raw["source_results_dir"]) if raw.get("source_results_dir") is not None else None,
        source=SourceRecipe(**raw["source"]),
        target=TargetRecipe(
            num_demos=tuple(int(x) for x in raw["target"]["num_demos"]),
            methods=tuple(str(x) for x in raw["target"]["methods"]),
            seeds=tuple(int(x) for x in raw["target"]["seeds"]),
            budgets=_load_training_budgets(raw["target"]),
            method_num_demos={
                str(method): tuple(int(x) for x in demos)
                for method, demos in raw["target"].get("method_num_demos", {}).items()
            },
            batch_size=int(raw["target"].get("batch_size", 32)),
        ),
        evaluation=EvalRecipe(**raw.get("evaluation", {})),
    )
    config.validate()
    return config


def target_grid(config: PilotExperimentConfig) -> list[tuple[int, str, int, int, str]]:
    """Return deterministic (target, method, demos, seed, budget-name) Stage-B cells."""
    return [
        (task_id, method, demos, seed, budget.name)
        for task_id in config.target_task_ids
        for method in config.target.methods
        for demos in config.target.demos_for_method(method)
        for seed in config.target.seeds
        for budget in config.target.budgets
    ]


def resolve_suite_tasks(meta: Any, config: PilotExperimentConfig, task_ids: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(resolve_task_string(meta, config.task_string(task_id)) for task_id in task_ids)


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
    budgets: tuple[int, ...],
) -> dict[int, list[int]]:
    """Shuffle once, then take prefixes so every requested demo subset is nested."""
    episodes = episodes_for_task(meta, task)
    seed_material = f"low-data-v1:{suite}:{task_id}:{seed}".encode()
    permutation_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    random.Random(permutation_seed).shuffle(episodes)
    return {budget: sorted(episodes[: min(budget, len(episodes))]) for budget in budgets}


def subset_statistics(meta: Any, episode_indices: list[int], *, optimizer_steps: int, batch_size: int) -> dict:
    num_transitions = sum(int(meta.episodes[index]["length"]) for index in episode_indices)
    num_training_windows = num_transitions  # one action-window start per selected LeRobot frame
    samples_seen = optimizer_steps * batch_size
    return {
        "num_selected_trajectories": len(episode_indices),
        "num_transitions": num_transitions,
        "num_training_windows": num_training_windows,
        "samples_seen": samples_seen,
        "optimizer_steps": optimizer_steps,
        "effective_epochs": samples_seen / max(num_training_windows, 1),
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
