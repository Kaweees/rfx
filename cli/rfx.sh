#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'USAGE'
Usage: cli/rfx.sh <doctor|doctor-teleop|doctor-so101|so101-demo|so101-bimanual|so101-setup|bootstrap|bootstrap-teleop|setup-source|check|pkg-create|pkg-list|run|launch|graph|topic-list|collect|validate|train|eval|shadow|deploy|runs|lineage|reproduce>
USAGE
}

doctor() {
  for tool in cargo git bash; do
    if command -v "$tool" >/dev/null 2>&1; then
      echo "[ok] $tool"
    else
      echo "[missing] $tool"
    fi
  done

  for tool in python3 python uv moon; do
    if command -v "$tool" >/dev/null 2>&1; then
      echo "[ok] $tool"
    fi
  done
}

bootstrap() {
  bash .claude/skills/rfx-bootstrap-install/scripts/bootstrap.sh
}

bootstrap_teleop() {
  bash scripts/bootstrap-teleop.sh
}

setup_source() {
  bash scripts/setup-from-source.sh
}

doctor_teleop() {
  bash scripts/doctor-teleop.sh
}

doctor_so101() {
  bash scripts/doctor-so101.sh
}

check() {
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets --all-features -- -D warnings
  scripts/python-checks.sh ci
}

runtime_cli() {
  uv run --python 3.13 python -m rfx.runtime.cli "$@"
}

so101_demo() {
  uv run --python 3.13 rfx/examples/so101_quickstart.py "$@"
}

so101_bimanual() {
  uv run --python 3.13 rfx/examples/teleop_record.py --config rfx/configs/so101_bimanual.yaml "$@"
}

so101_setup() {
  uv run --python 3.13 python scripts/so101-setup.py "$@"
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  case "$1" in
    doctor)
      doctor
      ;;
    doctor-teleop)
      doctor_teleop
      ;;
    doctor-so101)
      shift
      doctor_so101 "$@"
      ;;
    so101-demo)
      shift
      so101_demo "$@"
      ;;
    so101-bimanual)
      shift
      so101_bimanual "$@"
      ;;
    so101-setup)
      shift
      so101_setup "$@"
      ;;
    bootstrap)
      bootstrap
      ;;
    bootstrap-teleop)
      bootstrap_teleop
      ;;
    setup-source)
      setup_source
      ;;
    check)
      check
      ;;
    pkg-create)
      shift
      runtime_cli pkg-create "$@"
      ;;
    pkg-list)
      shift
      runtime_cli pkg-list "$@"
      ;;
    run)
      shift
      runtime_cli run "$@"
      ;;
    launch)
      shift
      runtime_cli launch "$@"
      ;;
    graph)
      shift
      runtime_cli graph "$@"
      ;;
    topic-list)
      shift
      runtime_cli topic-list "$@"
      ;;
    collect)
      shift
      runtime_cli collect "$@"
      ;;
    validate)
      shift
      runtime_cli validate "$@"
      ;;
    train)
      shift
      runtime_cli train "$@"
      ;;
    eval)
      shift
      runtime_cli eval "$@"
      ;;
    shadow)
      shift
      runtime_cli shadow "$@"
      ;;
    deploy)
      shift
      runtime_cli deploy "$@"
      ;;
    runs)
      shift
      runtime_cli runs "$@"
      ;;
    lineage)
      shift
      runtime_cli lineage "$@"
      ;;
    reproduce)
      shift
      runtime_cli reproduce "$@"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
