#!/bin/bash
# Config-driven native-Slurm entry point for the complete Stage-A/Stage-B workflow.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CLUSTER=
if [ "${1:-}" = --cluster ]; then
  CLUSTER=${2:?--cluster requires sofia or leonardo}
  shift 2
fi
if [ -n "${CLUSTER}" ]; then
  # shellcheck source=scripts/low_data_cluster_profiles.sh
  source "${SCRIPT_DIR}/low_data_cluster_profiles.sh"
  load_low_data_cluster_profile "${CLUSTER}"
fi
REPO_ROOT=${OPENPI_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
exec "${REPO_ROOT}/scripts/submit_low_data_main_slurm.sh" "$@"
