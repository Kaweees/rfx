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
Usage: scripts/teleop-jitter-check.sh [options]

Options:
  --rate-hz <float>      Control loop rate (default: 350)
  --duration-s <float>   Measurement duration in seconds (default: 1.0)
  --warmup-s <float>     Warmup duration in seconds (default: 0.5)
  --p99-ms <float>       p99 jitter budget in milliseconds (default: 0.5)
  --output <path>        Optional JSON output path
  -h, --help             Show this help
USAGE
}

RATE_HZ="350"
DURATION_S="1.0"
WARMUP_S="0.5"
P99_MS="0.5"
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rate-hz)
      RATE_HZ="${2:-}"
      shift 2
      ;;
    --duration-s)
      DURATION_S="${2:-}"
      shift 2
      ;;
    --warmup-s)
      WARMUP_S="${2:-}"
      shift 2
      ;;
    --p99-ms)
      P99_MS="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="${2:-}"
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

if [[ -n "${RFX_SKIP_TELEOP_JITTER_GATE:-}" ]]; then
  echo "[teleop-jitter] skipped (RFX_SKIP_TELEOP_JITTER_GATE is set)"
  exit 0
fi

if [[ -z "$OUTPUT_PATH" ]]; then
  mkdir -p "$ROOT/.rfx"
  OUTPUT_PATH="$ROOT/.rfx/teleop-jitter.json"
fi

echo "[teleop-jitter] running benchmark rate=${RATE_HZ}Hz duration=${DURATION_S}s warmup=${WARMUP_S}s p99<=${P99_MS}ms"

PYTHONPATH="$ROOT:$ROOT/rfx/python:${PYTHONPATH:-}" "$PYTHON_BIN" - "$RATE_HZ" "$DURATION_S" "$WARMUP_S" "$P99_MS" "$OUTPUT_PATH" <<'PY'
import json
import sys
from pathlib import Path

from rfx.teleop.benchmark import assert_jitter_budget, run_jitter_benchmark

rate_hz = float(sys.argv[1])
duration_s = float(sys.argv[2])
warmup_s = float(sys.argv[3])
p99_ms = float(sys.argv[4])
output_path = Path(sys.argv[5])

result = run_jitter_benchmark(rate_hz=rate_hz, duration_s=duration_s, warmup_s=warmup_s)
payload = result.to_dict()
payload["p99_budget_s"] = p99_ms / 1000.0

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)

print(json.dumps(payload, indent=2, sort_keys=True))

assert_jitter_budget(result, p99_budget_s=p99_ms / 1000.0)
PY

echo "[teleop-jitter] passed"
echo "[teleop-jitter] report=$OUTPUT_PATH"
