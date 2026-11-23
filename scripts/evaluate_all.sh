#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
exec "$FAST_JEPA_PYTHON" "$FAST_JEPA_ROOT/tools/evaluate_matrix.py" "$@"
