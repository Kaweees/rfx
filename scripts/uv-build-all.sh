#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

read_version() {
  local file="$1"
  sed -nE 's/^version = "([^"]+)"/\1/p' "$file" | head -n1
}

if ! command -v uv >/dev/null 2>&1; then
  echo "[build] missing uv"
  exit 1
fi

CORE_VERSION="$(read_version pyproject.toml)"
if [[ -z "$CORE_VERSION" ]]; then
  echo "[build] failed to read version from pyproject.toml"
  exit 1
fi

for pkg in packages/rfx-sim packages/rfx-go2 packages/rfx-lerobot; do
  PKG_VERSION="$(read_version "${pkg}/pyproject.toml")"
  if [[ -z "$PKG_VERSION" ]]; then
    echo "[build] failed to read version from ${pkg}/pyproject.toml"
    exit 1
  fi
  if [[ "$PKG_VERSION" != "$CORE_VERSION" ]]; then
    echo "[build] version mismatch:"
    echo "  core (rfx):        ${CORE_VERSION}"
    echo "  ${pkg##*/}: ${PKG_VERSION}"
    echo "[build] set matching versions before building release artifacts."
    exit 1
  fi
done

DIST_DIR="${ROOT}/dist"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

echo "[build] version: ${CORE_VERSION}"
echo "[build] building core package (rfx)..."
uv build --out-dir "$DIST_DIR"

for pkg in packages/rfx-sim packages/rfx-go2 packages/rfx-lerobot; do
  if [[ -f "${pkg}/pyproject.toml" ]]; then
    echo "[build] building extension package: ${pkg}"
    uv build "$pkg" --out-dir "$DIST_DIR"
  fi
done

echo "[build] artifacts:"
ls -1 "$DIST_DIR"
