#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PATH="${RFX_VENV_PATH:-$ROOT/.venv}"
PYTHON_BIN="$VENV_PATH/bin/python"

echo "[bootstrap-teleop] running source setup"
bash scripts/setup-from-source.sh

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[bootstrap-teleop] missing python at $PYTHON_BIN" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  echo "[bootstrap-teleop] installing teleop extras (+ LeRobot exporter)"
  if ! uv pip install --python "$PYTHON_BIN" -e "$ROOT[teleop,teleop-lerobot]"; then
    echo "[bootstrap-teleop] warning: failed to install teleop-lerobot extras; continuing with core teleop extras"
    uv pip install --python "$PYTHON_BIN" -e "$ROOT[teleop]"
  fi
else
  echo "[bootstrap-teleop] installing teleop extras (+ LeRobot exporter) with pip"
  if ! "$PYTHON_BIN" -m pip install -e "$ROOT[teleop,teleop-lerobot]"; then
    echo "[bootstrap-teleop] warning: failed to install teleop-lerobot extras; continuing with core teleop extras"
    "$PYTHON_BIN" -m pip install -e "$ROOT[teleop]"
  fi
fi

echo "[bootstrap-teleop] running diagnostics"
bash scripts/doctor-teleop.sh

echo "[bootstrap-teleop] complete"
