<div align="center">

[<img alt="rfx logo" src="docs/assets/logo.svg" width="220" />](https://github.com/quantbagel/rfx)

rfx: A ground-up replacement for ROS, built for the foundation model era.

<h3>

[Homepage](https://github.com/quantbagel/rfx) | [Documentation](https://deepwiki.com/quantbagel/rfx) | [Discord](https://discord.gg/xV8bAGM8WT)

</h3>

[![CI](https://github.com/quantbagel/rfx/actions/workflows/ci.yml/badge.svg)](https://github.com/quantbagel/rfx/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/xV8bAGM8WT)

</div>

---

rfx is robotics infrastructure for the data and embodiment layer:

- Rust core for real-time performance and safety
- Python SDK for fast research iteration
- ROS interop bindings for incremental migration
- Simulation, teleoperation, and hardware pipelines designed for scalable data collection
- `rfxJIT` IR/compiler/runtime that lowers and executes kernels across `cpu`/`cuda`/`metal`

ROS became the default robotics middleware over the last 15+ years, but it was designed for component message passing, not model-first robotics and large-scale data pipelines. rfx is designed from first principles for that new workflow.

It is inspired by PyTorch (ergonomics), JAX (functional transforms and IR-based AD), and TVM (scheduling/codegen), while explicitly targeting ROS replacement over time.

---

## Repository layout

```
rfx/            Rust core + Python package + tests + configs + examples
rfxJIT/         IR compiler and runtime (cpu/cuda/metal backends)
cli/            Command-line tools
docs/           Internal docs, perf baselines, contributor workflows
scripts/        Setup and CI helper scripts
.github/        GitHub Actions workflows
```

## Core interface

All robots in rfx implement the same three-method protocol:

```python
observation = robot.observe()
robot.act(action)
robot.reset()
```

This interface is consistent across simulation, real hardware, and teleoperation.

## Installation

The recommended install for contributors is from source.

### From source

```bash
git clone https://github.com/quantbagel/rfx.git
cd rfx
bash scripts/setup-from-source.sh
```

### Direct (GitHub)

```bash
uv pip install git+https://github.com/quantbagel/rfx.git
```

### PyPI (after release)

```bash
uv pip install rfx-sdk
uv pip install rfx-sdk-sim rfx-sdk-go2 rfx-sdk-lerobot
```

### TestPyPI (current test channel)

```bash
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple \
  rfx-sdk rfx-sdk-sim rfx-sdk-go2 rfx-sdk-lerobot torch
```

### Direct (local path)

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e /absolute/path/to/rfx
```

## Runtime switches (`rfxJIT`)

```bash
export RFX_JIT=1
export RFX_JIT_BACKEND=auto  # auto|cpu|cuda|metal
export RFX_JIT_STRICT=0      # 1 to raise if requested backend fails
```

With `RFX_JIT=1`, `@rfx.policy(jit=True)` can route NumPy policy calls through `rfxJIT` while preserving fallback behavior.

## Quality and performance checks

Run local pre-push checks:

```bash
./.venv/bin/pre-commit run --all-files --hook-stage pre-push
```

Run the CPU perf gate used in CI:

```bash
bash scripts/perf-check.sh \
  --baseline docs/perf/baselines/rfxjit_microkernels_cpu.json \
  --backend cpu \
  --threshold-pct 10
```

## Documentation

- Full documentation: [deepwiki.com/quantbagel/rfx](https://deepwiki.com/quantbagel/rfx)
- Docs entrypoint: `docs/README.md`
- SO101 quickstart: `docs/so101.md`
- Contributor workflow: `docs/workflow.md`
- Performance workflow: `docs/perf/README.md`
- Contributing guide: `CONTRIBUTING.md`

## Community and support

- Issues: https://github.com/quantbagel/rfx/issues
- Discussions: https://github.com/quantbagel/rfx/discussions
- Pull requests: https://github.com/quantbagel/rfx/pulls
- Community expectations: `CODE_OF_CONDUCT.md`

## License

MIT. See `LICENSE`.
