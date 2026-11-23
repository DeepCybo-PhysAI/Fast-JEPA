#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
task=${1:?Usage: eval.sh TASK CHECKPOINT [EVAL_SEED] [Hydra overrides...]}
checkpoint=${2:?Checkpoint .pt with adjacent config.json is required}
seed=${3:-42}
if (( $# >= 3 )); then shift 3; else shift 2; fi
case "$task" in pusht|cube|reacher|tworoom) ;; *) echo "Unknown task: $task" >&2; exit 2;; esac
"$FAST_JEPA_PYTHON" -c 'import pygame; assert pygame.version.ver == "2.6.1" and not getattr(pygame, "IS_CE", False), "Wrong renderer: rerun scripts/install.sh to select pygame 2.6.1"'
checkpoint=$(realpath "$checkpoint")
[[ -f "$checkpoint" && -f "$(dirname "$checkpoint")/config.json" ]] || { echo "Missing checkpoint/config.json" >&2; exit 2; }
weight_name=$(basename "$(dirname "$checkpoint")")
run_root=${FAST_JEPA_RUN_ROOT:-$FAST_JEPA_ROOT/outputs/${task}/${weight_name}/eval_${seed}}
mkdir -p "$run_root"
export PAPER_EVAL_VIDEO_DIR=${PAPER_EVAL_VIDEO_DIR:-$run_root/videos}
export PYTHONPATH="$FAST_JEPA_ROOT/inference"
exec "$FAST_JEPA_PYTHON" "$FAST_JEPA_ROOT/inference/eval_isolated.py" --config-name "$task"   "policy=$checkpoint" "seed=$seed" eval.num_eval=100 eval.inference_mode=guarded_a solver=fast_jepa   "cache_dir=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}" "output.filename=$run_root/evaluation.txt"   "hydra.run.dir=$run_root/hydra" "$@"
