#!/bin/bash
# Config-driven native-Slurm entry point for the complete Stage-A/Stage-B workflow.

set -euo pipefail

REPO_ROOT=${OPENPI_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
exec "${REPO_ROOT}/scripts/submit_low_data_main_slurm.sh" "$@"
