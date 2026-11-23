#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
export PYTHONPATH="$FAST_JEPA_ROOT/inference:$FAST_JEPA_ROOT"
exec "$FAST_JEPA_PYTHON" -m dynagoal.matrix "$@"
