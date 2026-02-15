#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

resolve_python() {
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    echo "$ROOT/.venv/bin/python"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi

  echo "No Python interpreter found. Create .venv or install python3." >&2
  exit 1
}

baseline_path_for_backend() {
  local backend="$1"
  echo "$RFX_PERF_BASELINE_DIR/rfxjit_microkernels_${backend}.json"
}

run_backend_check() {
  local backend="$1"
  local baseline_path="$2"
  local output_path

  if [[ ! -f "$baseline_path" ]]; then
    echo "[perf-gate] bootstrapping local baseline for backend=$backend"
    bash "$ROOT/scripts/perf-baseline.sh" \
      --backend "$backend" \
      --output-dir "$RFX_PERF_BASELINE_DIR" \
      --size "$RFX_PERF_SIZE" \
      --iterations "$RFX_PERF_ITERATIONS" \
      --warmup "$RFX_PERF_WARMUP" \
      --seed "$RFX_PERF_SEED"
  fi

  output_path="$(mktemp "${TMPDIR:-/tmp}/rfxjit-${backend}.XXXXXX")"
  echo "[perf-gate] checking backend=$backend"
  bash "$ROOT/scripts/perf-check.sh" \
    --baseline "$baseline_path" \
    --output "$output_path" \
    --backend "$backend" \
    --size "$RFX_PERF_SIZE" \
    --iterations "$RFX_PERF_ITERATIONS" \
    --warmup "$RFX_PERF_WARMUP" \
    --seed "$RFX_PERF_SEED" \
    --threshold-pct "$RFX_PERF_THRESHOLD_PCT" \
    --fail-on-regression
}

PYTHON_BIN="$(resolve_python)"
RFX_PERF_SIZE="${RFX_PERF_SIZE:-65536}"
RFX_PERF_ITERATIONS="${RFX_PERF_ITERATIONS:-200}"
RFX_PERF_WARMUP="${RFX_PERF_WARMUP:-10}"
RFX_PERF_SEED="${RFX_PERF_SEED:-42}"
RFX_PERF_THRESHOLD_PCT="${RFX_PERF_THRESHOLD_PCT:-10}"
RFX_PERF_BASELINE_DIR="${RFX_PERF_BASELINE_DIR:-$ROOT/.rfx/perf-baselines}"
mkdir -p "$RFX_PERF_BASELINE_DIR"

BACKEND_JSON="$(
  PYTHONPATH="$ROOT:$ROOT/rfx/python:${PYTHONPATH:-}" "$PYTHON_BIN" - <<'PY'
import json
from rfxJIT.runtime import available_backends

print(json.dumps(available_backends()))
PY
)"

is_backend_available() {
  local backend="$1"
  "$PYTHON_BIN" - "$BACKEND_JSON" "$backend" <<'PY'
import json
import sys

avail = json.loads(sys.argv[1])
name = sys.argv[2]
print("1" if avail.get(name, False) else "0")
PY
}

echo "[perf-gate] local regression gate (threshold=${RFX_PERF_THRESHOLD_PCT}%)"
echo "[perf-gate] params: size=${RFX_PERF_SIZE} iterations=${RFX_PERF_ITERATIONS} warmup=${RFX_PERF_WARMUP}"
echo "[perf-gate] available backends: $BACKEND_JSON"
echo "[perf-gate] local baseline dir: $RFX_PERF_BASELINE_DIR"

run_backend_check "cpu" "$(baseline_path_for_backend cpu)"

for backend in cuda metal; do
  if [[ "$(is_backend_available "$backend")" == "1" ]]; then
    run_backend_check "$backend" "$(baseline_path_for_backend "$backend")"
  else
    echo "[perf-gate] backend=$backend unavailable; skipping"
  fi
done

echo "[perf-gate] all backend checks passed"
