#!/usr/bin/env bash

set -euo pipefail
FAST_JEPA_ROOT=$(cd "$(dirname "$0")/.." && pwd)
FAST_JEPA_VENV=${FAST_JEPA_VENV:-$FAST_JEPA_ROOT/.venv}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
lock=$(mktemp)
trap 'rm -f "$lock"' EXIT
"$PYTHON_BIN" - "$FAST_JEPA_ROOT/requirements-blackwell.lock" "$lock" <<'PY'
import sys
from pathlib import Path
lines=Path(sys.argv[1]).read_text().splitlines()
Path(sys.argv[2]).write_text('\n'.join(x for x in lines if not x.startswith(('torch==','torchvision==')))+'\n')
PY
wheel_dir=${FAST_JEPA_WHEELHOUSE:-$FAST_JEPA_ROOT/.wheelhouse}
"$PYTHON_BIN" "$FAST_JEPA_ROOT/tools/prefetch_cuda.py" --output "$wheel_dir"
if command -v uv >/dev/null 2>&1; then
  if [[ ! -x "$FAST_JEPA_VENV/bin/python" ]]; then uv venv --python "$PYTHON_BIN" "$FAST_JEPA_VENV"; fi
  uv pip install --python "$FAST_JEPA_VENV/bin/python" --no-deps --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128
  uv pip install --python "$FAST_JEPA_VENV/bin/python" --no-deps "$wheel_dir"/*.whl
  uv pip install --python "$FAST_JEPA_VENV/bin/python" --no-deps --index-url https://pypi.org/simple -r "$lock"

  uv pip install --python "$FAST_JEPA_VENV/bin/python" --no-deps --reinstall --index-url https://pypi.org/simple \
    pygame==2.6.1 opencv-python==4.13.0.92
  uv pip check --python "$FAST_JEPA_VENV/bin/python"
else
  if [[ ! -x "$FAST_JEPA_VENV/bin/python" ]]; then "$PYTHON_BIN" -m venv "$FAST_JEPA_VENV"; fi
  "$FAST_JEPA_VENV/bin/python" -m pip install --upgrade pip
  "$FAST_JEPA_VENV/bin/python" -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128
  "$FAST_JEPA_VENV/bin/python" -m pip install --no-deps "$wheel_dir"/*.whl
  "$FAST_JEPA_VENV/bin/python" -m pip install --no-deps --index-url https://pypi.org/simple -r "$lock"
  "$FAST_JEPA_VENV/bin/python" -m pip install --no-deps --force-reinstall --index-url https://pypi.org/simple \
    pygame==2.6.1 opencv-python==4.13.0.92
  "$FAST_JEPA_VENV/bin/python" -m pip check
fi
