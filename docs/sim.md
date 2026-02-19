# Simulation Guide

This document explains how to run rfx in simulation without robot hardware.

Current status:

- Genesis integration is available now.
- Isaac Lab and MuJoCo integration are planned and will be added in this same guide.

## Quickstart (Genesis)

Run this from the repo root:

```bash
uv pip install --python 3.13 torch
uv run --python 3.13 rfx/examples/genesis_viewer.py --auto-install
```

What this does:

- starts the Genesis backend
- opens the viewer
- auto-installs `genesis-world` if missing

## Go2 URDF Integration

Go2 assets are not bundled in this repository.
Place your own URDF + meshes under `rfx/assets/robots/go2/` before running Go2 examples.

Create this layout:

```text
rfx/assets/robots/go2/
├── urdf/
│   ├── go2.urdf
│   └── meshes/...
└── mjcf/
    └── go2.xml
```

Then run:

```bash
uv run --python 3.13 rfx/examples/universal_go2.py --backend genesis --auto-install
```

`rfx/configs/go2.yaml` now defaults `urdf_path` to:

- `rfx/assets/robots/go2/urdf/go2.urdf`

For a Go2 command demo:

```bash
uv run --python 3.13 rfx/examples/universal_go2.py --backend genesis --auto-install
```

## Minimal Python API

```python
from rfx.sim import SimRobot

robot = SimRobot.from_config(
    "rfx/configs/so101.yaml",
    backend="genesis",
    viewer=True,
    auto_install=True,
)

obs = robot.reset()
```

## Common Run Modes

Viewer mode:

```bash
uv run --python 3.13 rfx/examples/genesis_viewer.py
```

Headless mode (no viewer):

```python
from rfx.sim import SimRobot
robot = SimRobot.from_config("rfx/configs/so101.yaml", backend="genesis", viewer=False)
```

Control runtime knobs:

```bash
uv run --python 3.13 rfx/examples/genesis_viewer.py \
  --num-envs 1 \
  --steps 2000 \
  --substeps 4
```

## Auto-install Behavior

If Genesis is not installed:

- rfx first tries: `uv pip install --python <active-interpreter> genesis-world`
- if `uv` is unavailable, it falls back to: `<active-interpreter> -m pip install genesis-world`

Current limitation:

- Genesis dependencies do not currently publish wheels for CPython 3.14.
- Use a CPython 3.13 environment for Genesis (`uv run --python 3.13 ...`).

You can also opt in globally:

```bash
export RFX_AUTO_INSTALL_GENESIS=1
```

Then any Genesis backend start will attempt install automatically.

## Troubleshooting

If Genesis install fails, run:

```bash
uv pip install genesis-world
```

If viewer does not appear:

- check that your machine supports graphics for the active environment
- retry with `--device cpu`
- validate by running headless first

## Roadmap

This guide will gain sections for:

- Isaac Lab integration
- MuJoCo integration
- backend switching patterns and shared config conventions
