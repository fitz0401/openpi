"""Temporal-ranking metrics for scalar progress scores."""

from __future__ import annotations

import math

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def temporal_metrics(scores: np.ndarray, *, min_separation: int) -> dict[str, float | int]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 2:
        raise ValueError("Temporal metrics require at least two scalar scores.")
    later_minus_earlier = []
    valid_differences = []
    for earlier in range(len(scores) - 1):
        all_differences = scores[earlier + 1 :] - scores[earlier]
        later_minus_earlier.append(all_differences)
        if earlier + min_separation < len(scores):
            valid_differences.append(scores[earlier + min_separation :] - scores[earlier])
    all_differences = np.concatenate(later_minus_earlier)
    valid_differences_array = np.concatenate(valid_differences)
    num_correct_pairs = int(np.sum(valid_differences_array > 0))
    pairwise_accuracy = num_correct_pairs / len(valid_differences_array)

    concordance_sum = float(np.sign(all_differences).sum())
    num_pairs = len(all_differences)
    num_ties = int(np.sum(all_differences == 0))
    denominator = math.sqrt(num_pairs * (num_pairs - num_ties))
    kendall_tau = concordance_sum / denominator if denominator else 0.0

    score_ranks = _average_ranks(scores)
    frame_order = np.arange(len(scores), dtype=np.float64)
    centered_scores = score_ranks - score_ranks.mean()
    centered_order = frame_order - frame_order.mean()
    spearman_denominator = np.linalg.norm(centered_scores) * np.linalg.norm(centered_order)
    spearman_rho = (
        float(np.dot(centered_scores, centered_order) / spearman_denominator) if spearman_denominator else 0.0
    )
    return {
        "pairwise_accuracy": pairwise_accuracy,
        "kendall_tau": kendall_tau,
        "spearman_rho": spearman_rho,
        "num_valid_pairs": len(valid_differences_array),
        "num_correct_pairs": num_correct_pairs,
    }


def regression_metrics(targets: np.ndarray, predictions: np.ndarray, *, min_separation: int) -> dict[str, float | int]:
    """Metrics for one trajectory; R² uses its mean-progress constant baseline."""
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape or targets.ndim != 1 or len(targets) < 2:
        raise ValueError("Regression metrics require equally sized 1-D target/prediction arrays.")
    residual_sum_squares = float(np.square(targets - predictions).sum())
    total_sum_squares = float(np.square(targets - targets.mean()).sum())
    r2 = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else 0.0
    temporal = temporal_metrics(predictions, min_separation=min_separation)
    return {
        "r2": r2,
        "mae": float(np.abs(targets - predictions).mean()),
        "spearman_rho": temporal["spearman_rho"],
        "pairwise_accuracy": temporal["pairwise_accuracy"],
        "num_valid_pairs": temporal["num_valid_pairs"],
        "num_correct_pairs": temporal["num_correct_pairs"],
    }
