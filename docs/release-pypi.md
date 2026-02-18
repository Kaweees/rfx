# PyPI Release (uv)

This repo can publish four Python packages together:

- `rfx-sdk`
- `rfx-sdk-sim`
- `rfx-sdk-go2`
- `rfx-sdk-lerobot`

## Prerequisites

- `uv` installed
- CPython 3.13 available (`uv python install 3.13`)
- PyPI token in `UV_PUBLISH_TOKEN`

For TestPyPI, create a TestPyPI token and use the same env var.

## 1) Build all artifacts

From repo root:

```bash
bash scripts/uv-build-all.sh
```

This script:

- validates all package versions match
- clears and recreates `dist/`
- builds sdist + wheel for each package

## CI Multi-Platform Build (recommended)

To publish across major platforms (Linux, macOS Intel/Apple Silicon, Windows):

1. Set repository secrets:
   - `TEST_PYPI_TOKEN`
   - `PYPI_TOKEN`
2. Run workflow: `.github/workflows/publish-python.yml`
3. Choose:
   - `index`: `testpypi` or `pypi`
   - `dry_run`: `true` first, then `false`

The workflow builds:

- `rfx-sdk` wheels on `ubuntu-latest`, `windows-latest`, `macos-13`, `macos-14`
- source distributions and pure wheels for `rfx-sdk`, `rfx-sdk-sim`, `rfx-sdk-go2`, `rfx-sdk-lerobot`
- current Python target: CPython 3.13

## 2) Dry-run publish (recommended)

```bash
UV_PUBLISH_TOKEN=*** bash scripts/uv-publish-all.sh testpypi --dry-run
```

## 3) Publish to TestPyPI

```bash
UV_PUBLISH_TOKEN=*** bash scripts/uv-publish-all.sh testpypi
```

## 4) Publish to PyPI

```bash
UV_PUBLISH_TOKEN=*** bash scripts/uv-publish-all.sh pypi
```

## Install after release

Core SDK:

```bash
uv pip install rfx-sdk
```

Add simulation + robot integrations:

```bash
uv pip install rfx-sdk rfx-sdk-sim rfx-sdk-go2 rfx-sdk-lerobot
```

Or with extras from core package:

```bash
uv pip install "rfx-sdk[teleop]"
```
