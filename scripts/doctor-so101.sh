#!/usr/bin/env bash
set -euo pipefail

echo "[so101] checking core tools..."
for tool in uv python3; do
  if command -v "$tool" >/dev/null 2>&1; then
    echo "[ok] $tool"
  else
    echo "[missing] $tool"
  fi
done

echo "[so101] checking serial ports..."
ls /dev/ttyACM* /dev/ttyUSB* /dev/tty.usbmodem* /dev/cu.usbmodem* 2>/dev/null || true

echo "[so101] checking python runtime deps..."
uv run --python 3.13 python - <<'PY'
mods = ["torch", "yaml", "rfx"]
for m in mods:
    try:
        __import__(m)
        print(f"[ok] {m}")
    except Exception as e:
        print(f"[missing] {m}: {e}")
try:
    import rfx._rfx as _r
    print("[ok] rfx._rfx extension")
except Exception as e:
    print("[missing] rfx._rfx extension:", e)
PY

echo "[so101] done"
