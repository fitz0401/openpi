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

PILOT_LORA_DEMO_GRID = (1, 2, 5, 10, 20, 50)
MAIN_LORA_DEMO_GRID = (1, 5, 10, 25, 50)
FULL_FT_DEMO_ANCHORS = (1, 10, 50)


@dataclasses.dataclass(frozen=True)
class SourceRecipe:
    optimizer_steps: int
    batch_size: int = 32


@dataclasses.dataclass(frozen=True)
class TrainingBudget:
    name: str
    mode: str
    max_steps: int
    max_effective_epochs: float
    min_steps: int = 1

    def validate(self) -> None:
        if not self.name or any(char.isspace() for char in self.name):
            raise ValueError("Training budget names must be non-empty and contain no whitespace.")
        if self.mode != "capped_effective_epochs":
            raise ValueError("Fine-tune baselines require mode='capped_effective_epochs'.")
        if self.max_steps <= 0:
            raise ValueError(f"Budget {self.name!r}: max_steps must be positive.")
        if self.max_effective_epochs <= 0:
            raise ValueError(f"Budget {self.name!r}: max_effective_epochs must be positive.")
        if self.min_steps <= 0 or self.min_steps > self.max_steps:
            raise ValueError(f"Budget {self.name!r}: require 0 < min_steps <= max_steps.")

    def resolve_steps(self, *, num_training_windows: int, batch_size: int) -> int:
        epoch_steps = math.ceil(self.max_effective_epochs * num_training_windows / batch_size)
        return max(self.min_steps, min(self.max_steps, epoch_steps))


@dataclasses.dataclass(frozen=True)
class TargetRecipe:
    num_demos: tuple[int, ...]
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    budget: TrainingBudget
    method_num_demos: dict[str, tuple[int, ...]]
    batch_size: int = 32

    def demos_for_method(self, method: str) -> tuple[int, ...]:
        return self.method_num_demos.get(method, self.num_demos)


@dataclasses.dataclass(frozen=True)
class EvalRecipe:
    num_trials: int = 20
    max_steps: int = 280
    replan_steps: int = 5


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
    target: TargetRecipe
    evaluation: EvalRecipe

    def validate(self) -> None:
        if self.schema_version not in (1, 2):
            raise ValueError(f"Unsupported schema_version={self.schema_version}")
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
        if "full" in self.target.methods and self.target.demos_for_method("full") != FULL_FT_DEMO_ANCHORS:
            raise ValueError(f"Full FT protocol requires sparse demo anchors {FULL_FT_DEMO_ANCHORS}.")
        expected_lora_grid = MAIN_LORA_DEMO_GRID if self.schema_version == 2 else PILOT_LORA_DEMO_GRID
        if "lora" in self.target.methods and self.target.demos_for_method("lora") != expected_lora_grid:
            raise ValueError(f"LoRA protocol requires the complete demo grid {expected_lora_grid}.")
        if self.source.optimizer_steps <= 0:
            raise ValueError("Source optimizer-step budget must be positive.")
        self.target.budget.validate()

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
        schema_version=int(raw["schema_version"]),
        split_id=str(raw["split_id"]),
        task_order_index=int(raw.get("task_order_index", 0)),
        task_splits=task_splits,
        base_checkpoint=str(raw["base_checkpoint"]),
        checkpoint_root=str(raw.get("checkpoint_root", "./checkpoints/low_data_pilot")),
        results_root=str(raw.get("results_root", "./results/low_data_pilot")),
        source=SourceRecipe(**raw["source"]),
        target=TargetRecipe(
            num_demos=tuple(int(x) for x in raw["target"]["num_demos"]),
            methods=tuple(str(x) for x in raw["target"]["methods"]),
            seeds=tuple(int(x) for x in raw["target"]["seeds"]),
            budget=TrainingBudget(**raw["target"]["budget"]),
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


def target_grid(config: PilotExperimentConfig) -> list[tuple[str, int, str, int, int]]:
    """Return deterministic (suite, target, method, demos, seed) Stage-B cells."""
    return [
        (target.suite, target.task_id, method, demos, seed)
        for target in config.target_task_refs()
        for method in config.target.methods
        for demos in config.target.demos_for_method(method)
        for seed in config.target.seeds
    ]


def target_result_dir(
    config: PilotExperimentConfig,
    suite: str,
    task_id: int,
    method: str,
    num_demos: int,
    seed: int,
) -> pathlib.Path:
    return (
        pathlib.Path(config.results_root)
        / config.split_id
        / "runs"
        / method
        / suite
        / f"task{task_id}"
        / f"demos{num_demos}"
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
