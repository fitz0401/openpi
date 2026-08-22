#!/bin/bash
# Backward-compatible Sofia wrapper for the generic cluster bootstrap.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/bootstrap_low_data_cluster.sh" --cluster sofia "$@"
