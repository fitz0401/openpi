#!/bin/bash
# Leonardo wrapper for environment creation plus Pi0.5/LIBERO prefetch.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/bootstrap_low_data_cluster.sh" --cluster leonardo "$@"
