#!/usr/bin/env bash
set -euo pipefail
FAST_JEPA_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FAST_JEPA_PYTHON=${FAST_JEPA_PYTHON:-$FAST_JEPA_ROOT/.venv/bin/python}
if [[ ! -x "$FAST_JEPA_PYTHON" ]]; then
  echo "Set FAST_JEPA_PYTHON to an installed Python, or create .venv first." >&2
  exit 2
fi
export STABLEWM_HOME=${STABLEWM_HOME:-$FAST_JEPA_ROOT/cache}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy}
export PYTHONUNBUFFERED=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
