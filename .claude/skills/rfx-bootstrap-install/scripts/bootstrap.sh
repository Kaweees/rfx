#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

echo "[bootstrap] repo root: $ROOT"

if command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] installing Python dev dependencies with uv"
  uv pip install -e '.[dev]'
else
  if command -v python3 >/dev/null 2>&1; then
    echo "[bootstrap] installing Python dev dependencies with python3 -m pip"
    python3 -m pip install -e '.[dev]'
  elif command -v python >/dev/null 2>&1; then
    echo "[bootstrap] installing Python dev dependencies with python -m pip"
    python -m pip install -e '.[dev]'
  else
    echo "No Python interpreter found. Install python3 (or uv) and retry." >&2
    exit 1
  fi
fi

echo "[bootstrap] enabling git hooks"
./scripts/setup-git-hooks.sh

echo "[bootstrap] complete"
