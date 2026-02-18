#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "[publish] missing uv"
  exit 1
fi

if [[ ! -d dist ]]; then
  echo "[publish] dist/ missing. Run scripts/uv-build-all.sh first."
  exit 1
fi

INDEX="${1:-testpypi}" # pypi | testpypi
DRY_RUN="${2:-}"

case "$INDEX" in
  pypi)
    REPO_URL="https://upload.pypi.org/legacy/"
    CHECK_URL="https://pypi.org/simple/"
    ;;
  testpypi)
    REPO_URL="https://test.pypi.org/legacy/"
    CHECK_URL="https://test.pypi.org/simple/"
    ;;
  *)
    echo "[publish] unknown index '$INDEX' (expected pypi|testpypi)"
    exit 1
    ;;
esac

DRY_RUN_FLAG=""
DRY_RUN_LABEL=""
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  DRY_RUN_FLAG="--dry-run"
  DRY_RUN_LABEL=" --dry-run"
elif [[ -n "$DRY_RUN" ]]; then
  echo "[publish] unknown option '$DRY_RUN' (expected empty or --dry-run)"
  exit 1
fi

if [[ -n "${UV_PUBLISH_TOKEN:-}" ]]; then
  echo "[publish] publishing to ${INDEX} with token auth${DRY_RUN_LABEL}"
  if [[ -n "$DRY_RUN_FLAG" ]]; then
    uv publish --publish-url "$REPO_URL" --check-url "$CHECK_URL" --dry-run dist/*
  else
    uv publish --publish-url "$REPO_URL" --check-url "$CHECK_URL" dist/*
  fi
elif [[ -n "${UV_PUBLISH_USERNAME:-}" && -n "${UV_PUBLISH_PASSWORD:-}" ]]; then
  echo "[publish] publishing to ${INDEX} with username/password auth${DRY_RUN_LABEL}"
  if [[ -n "$DRY_RUN_FLAG" ]]; then
    uv publish --publish-url "$REPO_URL" --check-url "$CHECK_URL" \
      --username "$UV_PUBLISH_USERNAME" \
      --password "$UV_PUBLISH_PASSWORD" \
      --dry-run \
      dist/*
  else
    uv publish --publish-url "$REPO_URL" --check-url "$CHECK_URL" \
      --username "$UV_PUBLISH_USERNAME" \
      --password "$UV_PUBLISH_PASSWORD" \
      dist/*
  fi
else
  echo "[publish] missing credentials."
  echo "Set one of:"
  echo "  UV_PUBLISH_TOKEN=<pypi-token>"
  echo "  UV_PUBLISH_USERNAME=<user> UV_PUBLISH_PASSWORD=<pass>"
  exit 1
fi
