"""Continual finetuning benchmark utilities for pi0.5 (stage 1: characterization only).

This package contains the small, reusable pieces that the continual benchmark adds on top of
the standard openpi training pipeline:

- ``subsample``: reproducible per-task demonstration subsampling (demo-budget sweep).
- ``metrics``: success-matrix metrics (average accuracy, forgetting, backward/forward transfer).

The sequential finetuning itself reuses ``scripts/train.py`` unchanged -- see
``scripts/continual_finetune.py`` for the orchestrator.
"""

from openpi.training.continual.subsample import SubsampleSpec
from openpi.training.continual.subsample import select_episode_indices

__all__ = ["SubsampleSpec", "select_episode_indices"]
