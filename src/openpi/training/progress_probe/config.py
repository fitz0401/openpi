"""Shared configuration records for Progress Regression Lite v3."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    repo_id: str = "physical-intelligence/libero"
    root: str | None = None
    image_key: str = "image"
    successful_expert_demos_only: bool = True


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    clip_model: str = "openai/clip-vit-base-patch32"
