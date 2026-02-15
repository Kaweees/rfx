# Contributing to rfx

Thanks for contributing.

## Development Setup

Recommended (full contributor setup):

```bash
git clone https://github.com/quantbagel/rfx.git
cd rfx
bash scripts/setup-from-source.sh
```

Manual equivalent:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
uv pip install --python .venv/bin/python -e .
cargo fetch
./.venv/bin/pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push
```

## Before Opening a PR

Run local quality checks:

```bash
./.venv/bin/pre-commit run --all-files --hook-stage pre-push
```

This runs Rust and Python formatting/lint/test checks and local performance gates.

## Performance Regressions

Local pre-push checks enforce CPU regression thresholds and attempt GPU checks when available.

Refresh baselines when needed:

```bash
bash scripts/perf-baseline.sh --backend all
```

See `docs/perf/README.md` for baseline policy and workflows.

## Pull Request Guidelines

- Keep changes focused and include tests for behavior changes.
- Update docs and examples when APIs or workflows change.
- Use clear commit messages that describe intent.
- Ensure CI passes in `.github/workflows/ci.yml`.

## Branch and Push Notes

- Optional local protection for direct pushes to `main`:

```bash
export RFX_BLOCK_MAIN_PUSH=1
```

## Getting Help

- Bug reports and feature requests: https://github.com/quantbagel/rfx/issues
- Questions and discussion: https://github.com/quantbagel/rfx/discussions
- Community expectations: `CODE_OF_CONDUCT.md`
