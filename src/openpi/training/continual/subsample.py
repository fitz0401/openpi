"""Reproducible per-task demonstration subsampling for the continual benchmark.

Given a LeRobot dataset that may contain many tasks, we restrict training to a single task and to
exactly ``budget`` randomly-chosen demonstrations (episodes). Selection is fully determined by
``seed`` so runs are reproducible, and the chosen episode indices are stored to disk.

The subsampling rides on a native LeRobot feature: ``LeRobotDataset(..., episodes=[...])`` accepts
an explicit episode whitelist, so we only need to compute *which* episodes to keep.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import random
from typing import Any


@dataclasses.dataclass(frozen=True)
class SubsampleSpec:
    """Specifies a single-task demo-budget subsample.

    Attributes:
        task: The task's natural-language instruction (must match the dataset's task string).
        budget: The desired number of demonstrations. If the task has fewer demos than ``budget``,
            *all* available demos are used (capped, never padded).
        seed: Seed for the reproducible random selection of episodes.
    """

    task: str
    budget: int
    seed: int


@dataclasses.dataclass(frozen=True)
class EpisodeSubsetSpec:
    """Restrict a dataset to one or more tasks and, optionally, explicit episodes.

    This is used by the low-data pilot. ``episode_indices=None`` selects every trajectory for the
    resolved tasks (Stage A); explicit indices select a deterministic target subset (Stage B).
    """

    tasks: tuple[str, ...]
    episode_indices: tuple[int, ...] | None = None


def _normalize(text: str) -> str:
    """Loose normalization used as a fallback when exact task-string matching fails."""
    return " ".join(text.lower().split())


def resolve_task_string(meta: Any, task: str) -> str:
    """Resolve ``task`` against the dataset's known task strings.

    Returns the dataset's canonical task string (exact match preferred, whitespace/case-insensitive
    fallback). Raises if the task is absent or ambiguous so mismatches fail loudly rather than
    silently training on the wrong / empty task.
    """
    known = list(meta.task_to_task_index.keys())
    if task in known:
        return task
    matches = [k for k in known if _normalize(k) == _normalize(task)]
    if len(matches) == 1:
        logging.warning("Task %r matched dataset task %r via normalized fallback.", task, matches[0])
        return matches[0]
    raise ValueError(
        f"Task {task!r} not found in dataset tasks (got {len(matches)} normalized matches). Available tasks: {known}"
    )


def select_episode_indices(meta: Any, task: str, budget: int, seed: int) -> list[int]:
    """Return the sorted episode indices selected for ``(task, budget, seed)``.

    Args:
        meta: A ``LeRobotDatasetMetadata`` instance (provides ``episodes`` and ``task_to_task_index``).
        task: Task instruction string (resolved via :func:`resolve_task_string`).
        budget: Desired number of demos; clamped to the number available for the task.
        seed: Reproducibility seed.

    Selection is order-independent: we sample from the sorted list of the task's episodes with a
    dedicated ``random.Random(seed)`` and return the result sorted, so the same ``(task, budget,
    seed)`` always yields the same set regardless of dict iteration order.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")

    canonical = resolve_task_string(meta, task)
    task_episodes = sorted(ep_idx for ep_idx, ep in meta.episodes.items() if canonical in ep["tasks"])
    if not task_episodes:
        raise ValueError(f"No episodes found for task {canonical!r}.")

    n_available = len(task_episodes)
    n_select = min(budget, n_available)
    if budget > n_available:
        logging.warning(
            "Requested budget %d for task %r exceeds available demos (%d); using all %d.",
            budget,
            canonical,
            n_available,
            n_available,
        )

    rng = random.Random(seed)
    selected = sorted(rng.sample(task_episodes, n_select))
    logging.info(
        "Subsample: task=%r budget=%d seed=%d -> %d/%d episodes: %s",
        canonical,
        budget,
        seed,
        len(selected),
        n_available,
        selected,
    )
    return selected


def select_explicit_episode_subset(meta: Any, spec: EpisodeSubsetSpec) -> tuple[list[str], list[int]]:
    """Resolve tasks and validate/select explicit episode indices."""
    if not spec.tasks:
        raise ValueError("EpisodeSubsetSpec.tasks must not be empty.")
    canonical_tasks = [resolve_task_string(meta, task) for task in spec.tasks]
    allowed = sorted(
        ep_idx for ep_idx, episode in meta.episodes.items() if any(task in episode["tasks"] for task in canonical_tasks)
    )
    if not allowed:
        raise ValueError(f"No episodes found for tasks {canonical_tasks!r}.")
    if spec.episode_indices is None:
        return canonical_tasks, allowed

    selected = sorted(set(spec.episode_indices))
    if len(selected) != len(spec.episode_indices):
        raise ValueError("Explicit episode indices must be unique.")
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise ValueError(f"Episodes do not belong to the selected tasks: {invalid}")
    if not selected:
        raise ValueError("Explicit episode subset must not be empty.")
    return canonical_tasks, selected


def save_indices(
    path: str | pathlib.Path,
    spec: SubsampleSpec,
    episode_indices: list[int],
    *,
    n_available: int | None = None,
    frame_count: int | None = None,
) -> None:
    """Persist the chosen episode indices (and context) as JSON for reproducibility."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": spec.task,
        "budget": spec.budget,
        "seed": spec.seed,
        "n_selected": len(episode_indices),
        "n_available": n_available,
        "frame_count": frame_count,
        "episode_indices": episode_indices,
    }
    path.write_text(json.dumps(payload, indent=2))
    logging.info("Saved sampled indices to %s", path)
