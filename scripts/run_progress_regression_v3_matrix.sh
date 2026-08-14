#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-all}"
CONFIG_DIR="examples/progress_probe/configs"
CONFIGS=(
  "$CONFIG_DIR/progress_regression_v3_spatial_source.json"
  "$CONFIG_DIR/progress_regression_v3_object_source.json"
  "$CONFIG_DIR/progress_regression_v3_goal_source.json"
  "$CONFIG_DIR/progress_regression_v3_spatial_object_source.json"
  "$CONFIG_DIR/progress_regression_v3_spatial_goal_source.json"
  "$CONFIG_DIR/progress_regression_v3_object_goal_source.json"
  "$CONFIG_DIR/progress_regression_v3_all_source.json"
)
SOURCE_SUBSETS=(
  spatial_source
  object_source
  goal_source
  spatial_object_source
  spatial_goal_source
  object_goal_source
  all_source
)

cd "$REPO_ROOT"
case "$PHASE" in
  train)
    for config in "${CONFIGS[@]}"; do scripts/train_progress_regression_probe.sh "$config"; done
    ;;
  eval)
    for config in "${CONFIGS[@]}"; do scripts/eval_progress_regression_probe.sh "$config"; done
    ;;
  aggregate)
    uv run python -m openpi.training.progress_probe.aggregate_regression \
      --results-root results/progress_probe/progress_regression_lite_v3 \
      --output-name compatibility_matrix \
      --source-subsets "${SOURCE_SUBSETS[@]}"
    ;;
  all)
    for config in "${CONFIGS[@]}"; do scripts/train_progress_regression_probe.sh "$config"; done
    for config in "${CONFIGS[@]}"; do scripts/eval_progress_regression_probe.sh "$config"; done
    uv run python -m openpi.training.progress_probe.aggregate_regression \
      --results-root results/progress_probe/progress_regression_lite_v3 \
      --output-name compatibility_matrix \
      --source-subsets "${SOURCE_SUBSETS[@]}"
    ;;
  *)
    echo "usage: $0 {train|eval|aggregate|all}" >&2
    exit 2
    ;;
esac
