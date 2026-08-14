import dataclasses
import pathlib
from types import SimpleNamespace

import pytest

from openpi.training.low_data.experiment import ALL_AVAILABLE_BUDGET
from openpi.training.low_data.experiment import FINAL_DATA_BUDGETS
from openpi.training.low_data.experiment import OFFICIAL_ROLLOUT_HORIZONS
from openpi.training.low_data.experiment import AdaptationRecipe
from openpi.training.low_data.experiment import load_experiment_config
from openpi.training.low_data.experiment import nested_episode_subsets
from openpi.training.low_data.experiment import subset_statistics
from openpi.training.low_data.experiment import target_grid

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


def _meta(num_episodes: int = 30):
    task = "target task"
    return SimpleNamespace(
        task_to_task_index={task: 0},
        episodes={idx: {"tasks": [task], "length": idx + 10} for idx in range(num_episodes)},
    )


def _adaptation(**overrides):
    values = {
        "data_budgets": FINAL_DATA_BUDGETS,
        "methods": ("lora", "full"),
        "method_data_budgets": {
            "lora": FINAL_DATA_BUDGETS,
            "full": ("1", "10", ALL_AVAILABLE_BUDGET),
        },
        "seeds": (0,),
        "suite_seed_overrides": {},
        "data_budget_seed_overrides": {},
        "effective_epochs": 10.0,
        "per_device_batch_size": 8,
        "world_size": 2,
        "gradient_accumulation_steps": 1,
        "hard_max_steps": None,
    }
    values.update(overrides)
    return AdaptationRecipe(**values)


def test_fixed_effective_epochs_has_no_floor_or_silent_cap():
    adaptation = _adaptation()
    assert adaptation.global_batch_size == 16
    assert adaptation.calculate_optimizer_steps(num_training_examples=1) == 1
    assert adaptation.calculate_optimizer_steps(num_training_examples=1000) == 625


def test_hard_max_steps_fails_loudly_instead_of_truncating():
    adaptation = _adaptation(hard_max_steps=500)
    with pytest.raises(ValueError, match="Protocol violation"):
        adaptation.calculate_optimizer_steps(num_training_examples=1000)


def test_nested_subsets_share_one_order_and_include_all_available():
    kwargs = {
        "meta": _meta(),
        "suite": "libero_goal",
        "task_id": 8,
        "task": "target task",
        "seed": 3,
        "data_budgets": FINAL_DATA_BUDGETS,
    }
    first_order, first = nested_episode_subsets(**kwargs)
    second_order, second = nested_episode_subsets(**kwargs)
    assert (first_order, first) == (second_order, second)
    assert first["1"] == first_order[:1]
    assert first["5"] == first_order[:5]
    assert first["10"] == first_order[:10]
    assert first["25"] == first_order[:25]
    assert first[ALL_AVAILABLE_BUDGET] == first_order
    assert set(first["1"]) < set(first["5"]) < set(first["10"]) < set(first["25"])


def test_subset_statistics_records_scientific_accounting():
    stats = subset_statistics(_meta(), [0, 1], optimizer_steps=6, global_batch_size=4)
    assert stats["num_selected_trajectories"] == 2
    assert stats["num_training_examples"] == 21
    assert stats["samples_seen"] == 24
    assert stats["achieved_effective_epochs"] == 24 / 21


def test_final_split_a_config_is_frozen_and_paired():
    config = load_experiment_config(_REPO_ROOT / "examples/low_data/configs/libero_main_18source_22target.json")
    assert len(config.source_task_refs()) == 18
    assert len(config.target_task_refs()) == 22
    assert config.adaptation.effective_epochs == 10.0
    assert config.adaptation.data_budgets == FINAL_DATA_BUDGETS
    assert config.adaptation.data_budgets_for_method("lora") == FINAL_DATA_BUDGETS
    assert config.adaptation.methods == ("lora",)
    assert config.adaptation.suite_seed_overrides == {"libero_10": (0,)}
    assert config.adaptation.data_budget_seed_overrides == {ALL_AVAILABLE_BUDGET: (0,)}
    assert config.adaptation.hard_max_steps is None
    assert config.evaluation.rollout_horizons == OFFICIAL_ROLLOUT_HORIZONS
    assert config.adaptation.seeds == (0, 1, 2)
    assert config.evaluation.num_trials == 25
    assert len(target_grid(config)) == 206
    assert target_grid(config)[0] == ("libero_spatial", 5, "lora", "1", 0)
    assert target_grid(config)[-1] == ("libero_10", 9, "lora", ALL_AVAILABLE_BUDGET, 0)


def test_suite_seed_override_only_changes_requested_suite():
    config = load_experiment_config(_REPO_ROOT / "examples/low_data/configs/libero_main_18source_22target.json")
    grid = target_grid(config)
    controlled_seeds = {
        seed for suite, _, _, budget, seed in grid if suite != "libero_10" and budget != ALL_AVAILABLE_BUDGET
    }
    libero_10_seeds = {seed for suite, _, _, _, seed in grid if suite == "libero_10"}
    all_available_seeds = {seed for _, _, _, budget, seed in grid if budget == ALL_AVAILABLE_BUDGET}
    assert controlled_seeds == {0, 1, 2}
    assert libero_10_seeds == {0}
    assert all_available_seeds == {0}


def test_nonstandard_rollout_horizon_is_rejected_without_debug_override():
    config = load_experiment_config(_REPO_ROOT / "examples/low_data/configs/libero_main_18source_22target.json")
    invalid = dataclasses.replace(
        config.evaluation,
        rollout_horizons={**OFFICIAL_ROLLOUT_HORIZONS, "libero_10": 280},
    )
    with pytest.raises(ValueError, match="official rollout_horizons"):
        invalid.validate()


def test_historical_configs_cannot_launch_future_runs():
    historical = _REPO_ROOT / "examples/low_data/configs/libero_goal_8source_2target.json"
    with pytest.raises(ValueError, match="historical"):
        load_experiment_config(historical)
