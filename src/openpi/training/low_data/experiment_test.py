import pathlib
from types import SimpleNamespace

from openpi.training.low_data.experiment import TrainingBudget
from openpi.training.low_data.experiment import load_experiment_config
from openpi.training.low_data.experiment import nested_episode_subsets
from openpi.training.low_data.experiment import subset_statistics
from openpi.training.low_data.experiment import target_grid

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


def _meta():
    task = "target task"
    return SimpleNamespace(
        task_to_task_index={task: 0},
        episodes={idx: {"tasks": [task], "length": idx + 10} for idx in range(12)},
    )


def test_nested_subsets_are_deterministic_prefix_sets():
    kwargs = {
        "meta": _meta(),
        "suite": "libero_goal",
        "task_id": 8,
        "task": "target task",
        "seed": 3,
        "budgets": (1, 2, 5, 10, 20, 50),
    }
    first = nested_episode_subsets(**kwargs)
    second = nested_episode_subsets(**kwargs)
    assert first == second
    assert set(first[1]) < set(first[2]) < set(first[5]) < set(first[10])
    assert set(first[10]) < set(first[20])
    assert first[20] == first[50]  # capped at 12 available trajectories


def test_subset_statistics_records_samples_and_epochs():
    stats = subset_statistics(_meta(), [0, 1], optimizer_steps=5, batch_size=4)
    assert stats["num_selected_trajectories"] == 2
    assert stats["num_transitions"] == 21
    assert stats["num_training_windows"] == 21
    assert stats["samples_seen"] == 20
    assert stats["effective_epochs"] == 20 / 21


def test_capped_effective_epoch_budget_applies_cap_and_floor():
    budget = TrainingBudget(
        name="capped10epochs",
        mode="capped_effective_epochs",
        max_steps=500,
        max_effective_epochs=10,
        min_steps=25,
    )
    budget.validate()
    assert budget.resolve_steps(num_training_windows=10, batch_size=32) == 25
    assert budget.resolve_steps(num_training_windows=100, batch_size=32) == 32
    assert budget.resolve_steps(num_training_windows=10_000, batch_size=32) == 500


def test_formal_configs_lock_finetune_baseline_protocol():
    for suite in ("spatial", "goal"):
        config = load_experiment_config(_REPO_ROOT / f"examples/low_data/configs/libero_{suite}_8source_2target.json")
        assert config.target.budget.mode == "capped_effective_epochs"
        assert config.target.budget.max_effective_epochs == 10
        assert config.target.budget.max_steps == 500
        assert config.target.budget.min_steps == 25
        assert config.target.demos_for_method("full") == (1, 10, 50)
        assert config.target.demos_for_method("lora") == (1, 2, 5, 10, 20, 50)
        assert len(target_grid(config)) == 18
