"""Task-uniform LIBERO frame sampling for Progress Regression Lite v3."""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import torch

from openpi.training.low_data.experiment import PilotExperimentConfig
from openpi.training.low_data.experiment import TaskRef
from openpi.training.low_data.experiment import episodes_for_task
from openpi.training.low_data.experiment import resolve_task_refs


@dataclasses.dataclass(frozen=True)
class VisionLanguageRegressionBatch:
    images: torch.Tensor
    instructions: list[str]
    targets: torch.Tensor


class LiberoProgressData:
    """Thin image-only view over LeRobot; actions and proprioception are never returned."""

    def __init__(self, repo_id: str, *, root: str | None, image_key: str) -> None:
        self.meta = LeRobotDatasetMetadata(repo_id, root=root)
        self.dataset = LeRobotDataset(repo_id, root=root)
        self.image_key = image_key
        self.episode_data_index = self.dataset.episode_data_index

    def task_episodes(self, split: PilotExperimentConfig, refs: tuple[TaskRef, ...]) -> dict[TaskRef, list[int]]:
        task_strings = resolve_task_refs(self.meta, split, refs)
        return {ref: episodes_for_task(self.meta, task) for ref, task in zip(refs, task_strings, strict=True)}

    def task_instruction(self, split: PilotExperimentConfig, ref: TaskRef) -> str:
        return resolve_task_refs(self.meta, split, (ref,))[0]

    def episode_length(self, episode_id: int) -> int:
        return int(self.meta.episodes[episode_id]["length"])

    def _global_index(self, episode_id: int, local_index: int) -> int:
        start = int(self.episode_data_index["from"][episode_id])
        return start + local_index

    def image(self, episode_id: int, local_index: int) -> torch.Tensor:
        return self.dataset[self._global_index(episode_id, local_index)][self.image_key]

    def images(self, episode_id: int, local_indices: list[int]) -> torch.Tensor:
        return torch.stack([self.image(episode_id, index) for index in local_indices])


def split_train_validation_episodes(
    episodes_by_task: dict[TaskRef, list[int]],
    *,
    seed: int,
    validation_demos_per_task: int,
    max_train_demos_per_task: int | None,
) -> tuple[dict[TaskRef, list[int]], dict[TaskRef, list[int]]]:
    train, validation = {}, {}
    for ref, episodes in episodes_by_task.items():
        ordered = list(episodes)
        random.Random(f"progress-probe:{seed}:{ref.key}").shuffle(ordered)
        num_validation = min(validation_demos_per_task, max(0, len(ordered) - 1))
        validation[ref] = sorted(ordered[:num_validation])
        remaining = ordered[num_validation:]
        if max_train_demos_per_task is not None:
            remaining = remaining[:max_train_demos_per_task]
        train[ref] = sorted(remaining)
    return train, validation


def minimum_separation(length: int, fraction: float) -> int:
    return max(1, min(length - 1, math.ceil(length * fraction)))


class UniformVisionLanguageProgressSampler:
    """Choose task, trajectory, and frame; derive progress only as a training label."""

    def __init__(
        self,
        data: LiberoProgressData,
        split: PilotExperimentConfig,
        episodes_by_task: dict[TaskRef, list[int]],
        *,
        seed: int,
    ) -> None:
        self.data = data
        self.refs = tuple(ref for ref, episodes in episodes_by_task.items() if episodes)
        self.episodes_by_task = episodes_by_task
        self.instructions = {ref: data.task_instruction(split, ref) for ref in self.refs}
        self.rng = random.Random(seed)
        if not self.refs:
            raise ValueError("Vision-language progress sampler has no eligible tasks.")

    def sample_batch(self, batch_size: int) -> VisionLanguageRegressionBatch:
        images, instructions, targets = [], [], []
        for _ in range(batch_size):
            ref = self.rng.choice(self.refs)
            episode_id = self.rng.choice(self.episodes_by_task[ref])
            length = self.data.episode_length(episode_id)
            if length < 2:
                raise ValueError(f"Episode {episode_id} has length {length}; normalized progress is undefined.")
            local_index = self.rng.randrange(length)
            images.append(self.data.image(episode_id, local_index))
            instructions.append(self.instructions[ref])
            targets.append(local_index / (length - 1))
        return VisionLanguageRegressionBatch(
            images=torch.stack(images),
            instructions=instructions,
            targets=torch.tensor(targets, dtype=torch.float32),
        )


def task_manifest(
    split: PilotExperimentConfig,
    data: LiberoProgressData,
    episodes_by_task: dict[TaskRef, list[int]],
) -> list[dict[str, Any]]:
    return [
        {
            "suite": ref.suite,
            "task_id": ref.task_id,
            "instruction": data.task_instruction(split, ref),
            "episode_ids": episodes,
            "num_demos": len(episodes),
        }
        for ref, episodes in episodes_by_task.items()
    ]
