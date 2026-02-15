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

usage() {
  cat <<'USAGE'
Usage: scripts/perf-baseline.sh [options]

Options:
  --backend <name>         cpu|cuda|metal|all (default: cpu)
  --output-dir <path>      Baseline output directory (default: docs/perf/baselines)
  --size <int>             Benchmark tensor size (default: 65536)
  --iterations <int>       Timing iterations (default: 200)
  --warmup <int>           Warmup iterations (default: 10)
  --seed <int>             RNG seed (default: 42)
  -h, --help               Show this help
USAGE
}

BACKEND="cpu"
OUTPUT_DIR="$ROOT/docs/perf/baselines"
SIZE=65536
ITERATIONS=200
WARMUP=10
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="${2:-}"
      shift 2
      ;;
    --size)
      SIZE="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --iterations)
      ITERATIONS="${2:-}"
      shift 2
      ;;
    --warmup)
      WARMUP="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

PYTHON_BIN="$(resolve_python)"
mkdir -p "$OUTPUT_DIR"

available_json="$(
  PYTHONPATH="$ROOT:$ROOT/rfx/python:${PYTHONPATH:-}" "$PYTHON_BIN" - <<'PY'
import json
from rfxJIT.runtime import available_backends

print(json.dumps(available_backends()))
PY
)"

is_backend_available() {
  local backend="$1"
  "$PYTHON_BIN" - "$available_json" "$backend" <<'PY'
import json
import sys

avail = json.loads(sys.argv[1])
name = sys.argv[2]
print("1" if avail.get(name, False) else "0")
PY
}

emit_baseline() {
  local backend="$1"
  local output_path="$OUTPUT_DIR/rfxjit_microkernels_${backend}.json"
  echo "[perf-baseline] generating backend=$backend -> $output_path"
  PYTHONPATH="$ROOT:$ROOT/rfx/python:${PYTHONPATH:-}" "$PYTHON_BIN" -m rfxJIT.runtime.benchmark \
    --size "$SIZE" \
    --iterations "$ITERATIONS" \
    --warmup "$WARMUP" \
    --seed "$SEED" \
    --backend "$backend" \
    --json-out "$output_path"
}

if [[ "$BACKEND" == "all" ]]; then
  for backend in cpu cuda metal; do
    if [[ "$backend" == "cpu" || "$(is_backend_available "$backend")" == "1" ]]; then
      emit_baseline "$backend"
    else
      echo "[perf-baseline] backend=$backend unavailable; skipping"
    fi
  done
else
  case "$BACKEND" in
    cpu|cuda|metal) ;;
    *)
      echo "Unsupported backend: $BACKEND (expected cpu|cuda|metal|all)" >&2
      exit 1
      ;;
  esac

  if [[ "$BACKEND" != "cpu" && "$(is_backend_available "$BACKEND")" != "1" ]]; then
    echo "[perf-baseline] backend=$BACKEND unavailable on this machine" >&2
    exit 1
  fi
  emit_baseline "$BACKEND"
fi

echo "[perf-baseline] done"
