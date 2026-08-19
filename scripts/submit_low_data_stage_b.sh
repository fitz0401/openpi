#!/bin/bash
# Reuse an immutable Stage-A checkpoint and submit baseline refresh -> Stage B -> finalizer.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXPERIMENT_CONFIG=${1:-examples/low_data/configs/libero_main_18source_22target.json}
EXPERIMENT_CONFIG=$(realpath "${EXPERIMENT_CONFIG}")
MAX_CONCURRENT=${MAX_CONCURRENT:-4}

if ! [[ "${MAX_CONCURRENT}" =~ ^[1-4]$ ]]; then
  echo "MAX_CONCURRENT must be between 1 and 4 (got ${MAX_CONCURRENT})" >&2
  exit 2
fi

cd "${REPO_ROOT}"
read -r GRID_SIZE NUM_SOURCE NUM_TARGET NUM_SEEDS NUM_LIBERO10_SEEDS NUM_ALL_AVAILABLE_SEEDS METHODS EVAL_PROTOCOL_ID RESULTS_DIR SOURCE_CHECKPOINT BASELINE_READY <<< "$(uv run python - "${EXPERIMENT_CONFIG}" <<'PY'
import json
import pathlib
import sys

from openpi.training.low_data.experiment import load_experiment_config, target_grid

config = load_experiment_config(sys.argv[1])
result_dir = pathlib.Path(config.results_root) / config.split_id
source_dir = result_dir / "source"
manifest_path = source_dir / "train_manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"Stage-B-only submission requires an existing Stage-A manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
checkpoint = pathlib.Path(manifest["checkpoint_dir"])
if not checkpoint.is_absolute():
    checkpoint = pathlib.Path.cwd() / checkpoint
if not (checkpoint / "params").is_dir():
    raise SystemExit(f"Stage-B-only submission requires existing Stage-A params: {checkpoint / 'params'}")

baseline_paths = (source_dir / "source_eval.json", source_dir / "target_zero_shot_eval.json")
try:
    expected = config.evaluation.protocol_manifest()
    baseline_ready = all(
        json.loads(path.read_text()).get("evaluation_protocol") == expected for path in baseline_paths
    )
except (FileNotFoundError, json.JSONDecodeError):
    baseline_ready = False

print(
    len(target_grid(config)),
    len(config.source_task_refs()),
    len(config.target_task_refs()),
    len(config.adaptation.seeds),
    len(config.adaptation.seeds_for("libero_10", "1")),
    len(config.adaptation.seeds_for("libero_spatial", "all_available")),
    ",".join(config.adaptation.methods),
    config.evaluation.protocol_id,
    result_dir,
    checkpoint,
    int(baseline_ready),
)
PY
)"

if [ "${NUM_SOURCE}" -ne 18 ] || [ "${NUM_TARGET}" -ne 22 ]; then
  echo "Refusing unexpected Split-A task manifest: source=${NUM_SOURCE}, target=${NUM_TARGET}" >&2
  exit 2
fi
if [ "${METHODS}" != lora ] || [ "${NUM_SEEDS}" -ne 3 ] || [ "${NUM_LIBERO10_SEEDS}" -ne 1 ] || [ "${NUM_ALL_AVAILABLE_SEEDS}" -ne 1 ] || [ "${GRID_SIZE}" -ne 206 ] || [ "${EVAL_PROTOCOL_ID}" != sog_target50_l10_target25_retention25_seed0_no_l10_retention ]; then
  echo "Refusing unexpected paper grid: methods=${METHODS}, global_seeds=${NUM_SEEDS}, libero10_seeds=${NUM_LIBERO10_SEEDS}, all_available_seeds=${NUM_ALL_AVAILABLE_SEEDS}, cells=${GRID_SIZE}, evaluation_protocol=${EVAL_PROTOCOL_ID}" >&2
  exit 2
fi

echo "Reusing immutable Stage-A checkpoint: ${SOURCE_CHECKPOINT}"
echo "Submitting ${GRID_SIZE} LoRA-only Stage-B cells with evaluation=${EVAL_PROTOCOL_ID} and maximum ${MAX_CONCURRENT} concurrent."
my_dodrio_quota -p starting_2026_047 || true

BASELINE_JOB_ID=""
DEPENDENCY_ARGS=()
if [ "${BASELINE_READY}" = 1 ]; then
  echo "Source and zero-shot baselines already match ${EVAL_PROTOCOL_ID}; no refresh job needed."
else
  BASELINE_JOB_ID=$(qsub \
    -N ld_base_eval \
    -l walltime=24:00:00 \
    -v "EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG},SOURCE_EVAL_ONLY=1" \
    scripts/job_low_data_stage_a.sh | tail -n 1)
  DEPENDENCY_ARGS=(-W "depend=afterok:${BASELINE_JOB_ID}")
  echo "Baseline refresh job: ${BASELINE_JOB_ID} (evaluation only; Stage A is not retrained)"
fi

ARRAY_RANGE="0-$((GRID_SIZE - 1))%${MAX_CONCURRENT}"
STAGE_B_JOB_ID=$(qsub \
  -N ld_main_B \
  "${DEPENDENCY_ARGS[@]}" \
  -t "${ARRAY_RANGE}" \
  -v "EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG},DELETE_TARGET_CHECKPOINT=1,DELETE_TARGET_CHECKPOINT_ON_FAILURE=1" \
  scripts/job_low_data_stage_b.sh | tail -n 1)

FINAL_JOB_ID=$(qsub \
  -N ld_finalize \
  -W "depend=afterany:${STAGE_B_JOB_ID}" \
  -v "EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG}" \
  scripts/job_low_data_finalize.sh | tail -n 1)

mkdir -p "${RESULTS_DIR}"
uv run python - "${RESULTS_DIR}/submission_manifest.json" "${EXPERIMENT_CONFIG}" \
  "${BASELINE_JOB_ID}" "${STAGE_B_JOB_ID}" "${FINAL_JOB_ID}" "${ARRAY_RANGE}" \
  "${GRID_SIZE}" "${NUM_SEEDS}" "${NUM_LIBERO10_SEEDS}" "${NUM_ALL_AVAILABLE_SEEDS}" "${METHODS}" \
  "${MAX_CONCURRENT}" "${SOURCE_CHECKPOINT}" <<'PY'
import datetime
import json
import pathlib
import sys

from openpi.training.low_data.experiment import evaluation_workload, load_experiment_config

path = pathlib.Path(sys.argv[1])
config = load_experiment_config(sys.argv[2])
timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
previous = json.loads(path.read_text()) if path.exists() else None
record = {
    "submitted_at": timestamp,
    "experiment_config": sys.argv[2],
    "stage_b_only_resubmission": True,
    "baseline_eval_job_id": sys.argv[3] or None,
    "stage_b_array_job_id": sys.argv[4],
    "finalize_job_id": sys.argv[5],
    "stage_b_array_range": sys.argv[6],
    "stage_b_grid_size": int(sys.argv[7]),
    "num_adaptation_seeds": int(sys.argv[8]),
    "num_libero_10_seeds": int(sys.argv[9]),
    "num_all_available_seeds": int(sys.argv[10]),
    "methods": sys.argv[11].split(","),
    "evaluation_protocol": config.evaluation.protocol_manifest(),
    "evaluation_workload": evaluation_workload(config),
    "max_concurrent": int(sys.argv[12]),
    "source_checkpoint": sys.argv[13],
    "baseline_eval_walltime": "24:00:00",
    "stage_b_walltime": "24:00:00",
}
if previous is not None:
    record["original_stage_a_job_id"] = previous.get("stage_a_job_id") or previous.get("original_stage_a_job_id")
history_path = path.with_name("submission_history.jsonl")
with history_path.open("a") as history:
    if previous is not None:
        history.write(json.dumps({"superseded_at": timestamp, "manifest": previous}) + "\n")
    history.write(json.dumps(record) + "\n")
path.write_text(json.dumps(record, indent=2) + "\n")
PY

echo "Stage B array:  ${STAGE_B_JOB_ID} (${ARRAY_RANGE})"
echo "Finalize job:   ${FINAL_JOB_ID} (after Stage B terminates)"
echo "Results:        ${RESULTS_DIR}"
