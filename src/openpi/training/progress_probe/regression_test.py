from __future__ import annotations

import csv
import pathlib

import numpy as np
import pytest
import torch
from torch import nn

from openpi.training.low_data.experiment import TaskRef
from openpi.training.progress_probe.aggregate_regression import main as aggregate_main
from openpi.training.progress_probe.dataset import UniformVisionLanguageProgressSampler
from openpi.training.progress_probe.evaluate_regression import aggregate
from openpi.training.progress_probe.metrics import regression_metrics
from openpi.training.progress_probe.model import ProgressRegressionVisionLanguageLite
from openpi.training.progress_probe.regression_config import load_regression_config


class _DummyFrozenClip(nn.Module):
    output_dim = 512

    def __init__(self) -> None:
        super().__init__()
        self.frozen_weight = nn.Parameter(torch.ones(1), requires_grad=False)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(1, 2, 3))[:, None].expand(-1, self.output_dim)

    def encode_text(self, instructions: list[str], device: torch.device) -> torch.Tensor:
        return torch.tensor([len(item) for item in instructions], dtype=torch.float32, device=device)[:, None].expand(
            -1, self.output_dim
        )


class _FakeData:
    def episode_length(self, episode_id: int) -> int:
        return {10: 5, 20: 3}[episode_id]

    def image(self, episode_id: int, local_index: int):
        return torch.full((3, 2, 2), float(local_index))

    def task_instruction(self, split, ref: TaskRef) -> str:
        return ref.key


def test_model_and_sampler_have_no_proprio_branch_or_input() -> None:
    model = ProgressRegressionVisionLanguageLite(_DummyFrozenClip())
    predictions = model(torch.rand(4, 3, 8, 8), ["a", "bb", "ccc", "dddd"])
    assert predictions.shape == (4,)
    assert torch.all((predictions >= 0) & (predictions <= 1))
    predictions.sum().backward()
    assert model.clip.frozen_weight.grad is None
    assert model.trainable_parameter_count() == 246_913
    assert all(not key.startswith("clip.") for key in model.trainable_state_dict())
    assert not hasattr(model, "state_encoder")
    ref = TaskRef("libero_spatial", 4)
    batch = UniformVisionLanguageProgressSampler(_FakeData(), None, {ref: [10]}, seed=3).sample_batch(12)
    assert set(batch.__dataclass_fields__) == {"images", "instructions", "targets"}


def test_regression_metrics_r2_reference_points() -> None:
    targets = np.linspace(0, 1, 6)
    perfect = regression_metrics(targets, targets, min_separation=1)
    constant = regression_metrics(targets, np.full_like(targets, targets.mean()), min_separation=1)
    assert perfect["r2"] == pytest.approx(1.0)
    assert perfect["mae"] == pytest.approx(0.0)
    assert perfect["spearman_rho"] == pytest.approx(1.0)
    assert perfect["pairwise_accuracy"] == pytest.approx(1.0)
    assert constant["r2"] == pytest.approx(0.0)


def test_task_and_suite_aggregation_are_macro_averages() -> None:
    base = {
        "probe_source_subset": "spatial_source",
        "target_suite": "libero_goal",
        "target_task_id": 1,
        "checkpoint": "/tmp/probe.pt",
        "seed": 0,
        "mae": 0.2,
        "spearman_rho": 0.8,
        "pairwise_accuracy": 0.9,
        "wrong_language_r2": 0.1,
    }
    task = aggregate([{**base, "r2": 1.0}, {**base, "r2": -1.0}], ("target_suite", "target_task_id"), "task")
    assert task[0]["r2"] == pytest.approx(0.0)
    assert task[0]["num_target_demos"] == 2


def test_matrix_aggregation_smoke(tmp_path: pathlib.Path, monkeypatch) -> None:
    singles = ("spatial_source", "object_source", "goal_source")
    leave_one_out = ("spatial_object_source", "spatial_goal_source", "object_goal_source")
    subsets = (*singles, *leave_one_out, "all_source")
    suites = ("libero_spatial", "libero_object", "libero_goal")
    for source_index, subset in enumerate(subsets):
        directory = tmp_path / subset / "evaluation"
        directory.mkdir(parents=True)
        rows = []
        output_suites = (*suites, "libero_10") if subset == "all_source" else suites
        for target_index, suite in enumerate(output_suites):
            rows.append(
                {
                    "target_suite": suite,
                    "num_target_demos": 10,
                    "r2": 0.1 * (source_index + target_index),
                    "r2_std": 0.01,
                    "mae": 0.2,
                    "spearman_rho": 0.7,
                    "pairwise_accuracy": 0.8,
                    "wrong_language_r2": 0.0,
                    "checkpoint": f"/{subset}.pt",
                    "seed": 0,
                }
            )
        with (directory / "per_suite.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    arguments = ["aggregate_regression", "--results-root", str(tmp_path)]
    arguments.extend(("--source-subsets", *subsets))
    monkeypatch.setattr("sys.argv", arguments)
    aggregate_main()
    matrix = list(csv.DictReader((tmp_path / "compatibility_matrix/source_to_target_compatibility_matrix.csv").open()))
    assert len(matrix) == len(subsets)
    assert (tmp_path / "compatibility_matrix/source_to_target_compatibility_matrix.png").exists()


@pytest.mark.parametrize(
    ("name", "source_count", "target_count"),
    [
        ("spatial_source", 6, 12),
        ("object_source", 6, 12),
        ("goal_source", 6, 12),
        ("spatial_object_source", 12, 12),
        ("spatial_goal_source", 12, 12),
        ("object_goal_source", 12, 12),
        ("all_source", 18, 22),
    ],
)
def test_paper_v3_configs(name, source_count, target_count) -> None:
    repo_root = pathlib.Path(__file__).parents[4]
    config = load_regression_config(repo_root / f"examples/progress_probe/configs/progress_regression_v3_{name}.json")
    assert config.schema_version == 3
    assert len(config.source_refs()) == source_count
    assert len(config.target_refs()) == target_count


def test_dataset_root_can_be_supplied_by_environment(tmp_path: pathlib.Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENPI_LIBERO_DATA_ROOT", str(tmp_path / "../dataset"))
    repo_root = pathlib.Path(__file__).parents[4]
    config = load_regression_config(
        repo_root / "examples/progress_probe/configs/progress_regression_v3_smoke.json"
    )
    assert config.dataset.root == str((tmp_path / "../dataset").resolve())
