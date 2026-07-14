from types import SimpleNamespace

import pytest

from openpi.training.continual.subsample import EpisodeSubsetSpec
from openpi.training.continual.subsample import select_explicit_episode_subset


def _meta():
    return SimpleNamespace(
        task_to_task_index={"task a": 0, "task b": 1},
        episodes={
            3: {"tasks": ["task a"], "length": 5},
            4: {"tasks": ["task b"], "length": 7},
            5: {"tasks": ["task a"], "length": 9},
        },
    )


def test_multitask_subset_selects_all_matching_episodes():
    tasks, episodes = select_explicit_episode_subset(_meta(), EpisodeSubsetSpec(tasks=("task a", "task b")))
    assert tasks == ["task a", "task b"]
    assert episodes == [3, 4, 5]


def test_explicit_subset_rejects_episode_from_other_task():
    with pytest.raises(ValueError, match="do not belong"):
        select_explicit_episode_subset(_meta(), EpisodeSubsetSpec(tasks=("task a",), episode_indices=(4,)))
