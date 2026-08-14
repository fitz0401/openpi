"""Configuration for the paper's Progress Regression Lite v3 probes."""

from __future__ import annotations

import dataclasses
import json
import pathlib

from openpi.training.low_data.experiment import load_experiment_config
from openpi.training.progress_probe.config import DatasetConfig
from openpi.training.progress_probe.config import ModelConfig


@dataclasses.dataclass(frozen=True)
class RegressionTrainingConfig:
    seed: int
    optimizer_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    validation_demos_per_task: int = 1
    validation_frames_per_task: int = 128
    validate_every: int = 100
    log_every: int = 10
    max_grad_norm: float = 1.0
    max_source_tasks: int | None = None
    max_train_demos_per_task: int | None = None


@dataclasses.dataclass(frozen=True)
class RegressionEvaluationConfig:
    batch_size: int = 128
    min_separation_fraction: float = 0.1
    max_eval_demos: int | None = None
    max_target_tasks: int | None = None
    wrong_language: bool = True


@dataclasses.dataclass(frozen=True)
class RegressionProbeConfig:
    schema_version: int
    experiment_name: str
    probe_source_subset: str
    source_suites: tuple[str, ...]
    target_suites: tuple[str, ...]
    split_config: pathlib.Path
    output_dir: pathlib.Path
    dataset: DatasetConfig
    model: ModelConfig
    training: RegressionTrainingConfig
    evaluation: RegressionEvaluationConfig

    def split(self):
        return load_experiment_config(self.split_config)

    def source_refs(self):
        allowed = set(self.source_suites)
        return tuple(ref for ref in self.split().source_task_refs() if ref.suite in allowed)

    def target_refs(self):
        allowed = set(self.target_suites)
        return tuple(ref for ref in self.split().target_task_refs() if ref.suite in allowed)

    def validate(self) -> None:
        if self.schema_version != 3:
            raise ValueError(f"The paper probe requires schema_version 3, got {self.schema_version}.")
        if not self.probe_source_subset:
            raise ValueError("probe_source_subset must be non-empty.")
        split = self.split()
        available_source_suites = {ref.suite for ref in split.source_task_refs()}
        available_target_suites = {ref.suite for ref in split.target_task_refs()}
        if not self.source_suites or not set(self.source_suites) <= available_source_suites:
            raise ValueError(f"source_suites must be a non-empty subset of {sorted(available_source_suites)}.")
        if not self.target_suites or not set(self.target_suites) <= available_target_suites:
            raise ValueError(f"target_suites must be a non-empty subset of {sorted(available_target_suites)}.")
        if not self.source_refs() or not self.target_refs():
            raise ValueError("Selected source and target suites must contain tasks in Split A.")
        if not self.dataset.successful_expert_demos_only:
            raise ValueError("Progress regression requires successful expert demonstrations only.")
        if self.training.optimizer_steps <= 0 or self.training.batch_size <= 0:
            raise ValueError("Training steps and batch size must be positive.")
        if self.training.validation_frames_per_task <= 0:
            raise ValueError("validation_frames_per_task must be positive.")
        if self.training.validate_every <= 0 or self.training.log_every <= 0:
            raise ValueError("Validation and logging intervals must be positive.")
        if self.training.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive.")
        if not 0 < self.evaluation.min_separation_fraction < 1:
            raise ValueError("evaluation.min_separation_fraction must be in (0, 1).")


def load_regression_config(path: str | pathlib.Path) -> RegressionProbeConfig:
    path = pathlib.Path(path).resolve()
    raw = json.loads(path.read_text())

    def resolve(value: str) -> pathlib.Path:
        resolved = pathlib.Path(value)
        return resolved if resolved.is_absolute() else (path.parent / resolved).resolve()

    config = RegressionProbeConfig(
        schema_version=int(raw["schema_version"]),
        experiment_name=str(raw["experiment_name"]),
        probe_source_subset=str(raw["probe_source_subset"]),
        source_suites=tuple(str(item) for item in raw["source_suites"]),
        target_suites=tuple(str(item) for item in raw["target_suites"]),
        split_config=resolve(raw["split_config"]),
        output_dir=resolve(raw["output_dir"]),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=RegressionTrainingConfig(**raw["training"]),
        evaluation=RegressionEvaluationConfig(**raw.get("evaluation", {})),
    )
    config.validate()
    return config


def resolved_regression_config(config: RegressionProbeConfig) -> dict:
    def convert(value):
        if dataclasses.is_dataclass(value):
            return {field.name: convert(getattr(value, field.name)) for field in dataclasses.fields(value)}
        if isinstance(value, pathlib.Path):
            return str(value)
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        return value

    return convert(config)
