"""Continual-learning metrics computed from a success matrix.

Success matrix convention (``R``):
    ``R[i][j]`` = success rate on task ``j`` after finishing training **stage** ``i``.
    Rows ``i`` range over ``0..N``: row ``0`` is the *pretrained* zero-shot baseline (before any
    finetuning); rows ``1..N`` are after each of the ``N`` sequential tasks.
    Columns ``j`` range over ``1..N`` (the tasks, in training order).

Because we evaluate the *full* row (all ``N`` tasks) after every stage, both backward transfer /
forgetting (looking at earlier tasks) and forward transfer (looking at not-yet-trained tasks) are
available.

This module is intentionally dependency-light (numpy only) and pure so it can be unit-tested in
isolation.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class ContinualMetrics:
    average_accuracy: float
    average_forgetting: float
    backward_transfer: float
    forward_transfer: float | None
    newest_task_success: float
    n_tasks: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def matrix_to_array(matrix: Mapping[int, Mapping[int, float]], n_tasks: int) -> np.ndarray:
    """Convert a nested ``{stage: {task: sr}}`` mapping into a dense ``(N+1, N)`` array.

    Stages are ``0..N`` (row 0 = pretrained baseline), tasks are ``1..N`` mapped to columns ``0..N-1``.
    Missing cells are filled with NaN so partially-filled matrices surface clearly.
    """
    arr = np.full((n_tasks + 1, n_tasks), np.nan, dtype=np.float64)
    for stage, row in matrix.items():
        for task, sr in row.items():
            arr[stage, task - 1] = sr
    return arr


def compute_metrics(matrix: Mapping[int, Mapping[int, float]], n_tasks: int) -> ContinualMetrics:
    """Compute continual-learning metrics from a (possibly partial) success matrix.

    Forward transfer requires the pretrained baseline row (stage 0); if it is absent, FWT is None.
    """
    arr = matrix_to_array(matrix, n_tasks)
    final = arr[n_tasks]  # success on every task after the last training stage.

    # Average accuracy: mean final success over all tasks.
    average_accuracy = float(np.nanmean(final))

    # Average forgetting: for each task j < N, max success seen at/after it was trained
    # (stages j..N-1) minus its final success.
    forgetting_terms: list[float] = []
    for j in range(1, n_tasks):  # tasks 1..N-1 (1-indexed)
        col = j - 1
        learned_curve = arr[j:n_tasks, col]  # stages j..N-1
        if np.all(np.isnan(learned_curve)) or np.isnan(final[col]):
            continue
        forgetting_terms.append(float(np.nanmax(learned_curve) - final[col]))
    average_forgetting = float(np.mean(forgetting_terms)) if forgetting_terms else 0.0

    # Backward transfer: mean over tasks j < N of (final - success right after task j was trained).
    bwt_terms: list[float] = []
    for j in range(1, n_tasks):
        col = j - 1
        diag = arr[j, col]  # success on task j immediately after training stage j
        if np.isnan(diag) or np.isnan(final[col]):
            continue
        bwt_terms.append(float(final[col] - diag))
    backward_transfer = float(np.mean(bwt_terms)) if bwt_terms else 0.0

    # Forward transfer: mean over tasks j >= 2 of (zero-shot success on j before training it
    # [stage j-1] minus pretrained baseline [stage 0]).
    forward_transfer: float | None = None
    if not np.all(np.isnan(arr[0])):
        fwt_terms: list[float] = []
        for j in range(2, n_tasks + 1):
            col = j - 1
            pre = arr[j - 1, col]
            base = arr[0, col]
            if np.isnan(pre) or np.isnan(base):
                continue
            fwt_terms.append(float(pre - base))
        forward_transfer = float(np.mean(fwt_terms)) if fwt_terms else 0.0

    newest_task_success = float(final[n_tasks - 1])

    return ContinualMetrics(
        average_accuracy=average_accuracy,
        average_forgetting=average_forgetting,
        backward_transfer=backward_transfer,
        forward_transfer=forward_transfer,
        newest_task_success=newest_task_success,
        n_tasks=n_tasks,
    )


def aggregate(metrics_list: Sequence[ContinualMetrics]) -> dict:
    """Average metrics across seeds, returning mean/std per field."""
    if not metrics_list:
        return {}
    fields = ["average_accuracy", "average_forgetting", "backward_transfer", "newest_task_success"]
    out: dict = {"n_seeds": len(metrics_list), "n_tasks": metrics_list[0].n_tasks}
    for f in fields:
        vals = np.array([getattr(m, f) for m in metrics_list], dtype=np.float64)
        out[f"{f}_mean"] = float(np.mean(vals))
        out[f"{f}_std"] = float(np.std(vals))
    fwt = [m.forward_transfer for m in metrics_list if m.forward_transfer is not None]
    if fwt:
        out["forward_transfer_mean"] = float(np.mean(fwt))
        out["forward_transfer_std"] = float(np.std(fwt))
    return out
