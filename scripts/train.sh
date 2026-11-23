#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
task=${1:?Usage: train.sh TASK [SEED] [Hydra overrides...]}
seed=${2:-0}
if (( $# >= 2 )); then shift 2; else shift; fi
case "$task" in pusht|cube|reacher|tworoom) ;; *) echo "Unknown task: $task" >&2; exit 2;; esac
export PYTHONPATH="$FAST_JEPA_ROOT/training"
exec "$FAST_JEPA_PYTHON" "$FAST_JEPA_ROOT/training/train.py"   --config-path "$FAST_JEPA_ROOT/training/config/fast" --config-name "$task"   "seed=$seed" "output_model_name=fast_${task}_s${seed}_e1" "$@"
