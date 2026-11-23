#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
checkpoint_root=${1:?Usage: reproduce_tables.sh CHECKPOINT_ROOT OUTPUT_ROOT}
output_root=${2:?Output root is required}
bash "$FAST_JEPA_ROOT/scripts/evaluate_all.sh" --checkpoint-root "$checkpoint_root" --output "$output_root/table1"
bash "$FAST_JEPA_ROOT/scripts/evaluate_dynagoal.sh" --subset shift --checkpoint-root "$checkpoint_root" --output "$output_root/table2"
bash "$FAST_JEPA_ROOT/scripts/evaluate_dynagoal.sh" --subset deadline --checkpoint-root "$checkpoint_root" --output "$output_root/table3"
