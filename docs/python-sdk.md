# Python SDK

This document describes the Python SDK surface and package structure.

## Design Goals

- One simple Python API for users.
- Keep robot/provider specifics in extension packages.
- Allow simulation and hardware stacks to evolve independently.

## SDK Entry Point

```python
import rfx

rfx.use("sim", "go2")  # optional provider activation from rfx namespace

bot = rfx.connect_robot(
    "go2",
    backend="genesis",  # mock | genesis | mjx | real
)

bot.reset()
bot.command(vx=0.6, vy=0.0, yaw=0.1)
obs = bot.step()
bot.close()
```

Everything starts from `import rfx`. You can still import other Python libraries (for example `torch`) in the same script as needed.

When `config` is omitted, `rfx-sdk` now uses built-in defaults (`GO2_CONFIG`/`SO101_CONFIG`) so examples work from wheel installs without repo-local YAML files.

## Package Layout

- `rfx`: base Python SDK and common interfaces
- `rfx-sdk-sim`: simulation adapters/backends/controllers
- `rfx-sdk-go2`: Unitree Go2-specific real/sim adapters and utilities
- `rfx-sdk-lerobot`: LeRobot dataset/export integration

These packages are scaffolded under `packages/` in this repository.

## Boundary Rules

- `rfx-sdk` (distribution) should not depend on `rfx-sdk-go2` or `rfx-sdk-lerobot`.
- `rfx-sdk-sim` can depend on `rfx-sdk`, and optional simulators.
- `rfx-sdk-go2` can depend on `rfx-sdk` and optionally `rfx-sdk-sim`.
- `rfx-sdk-lerobot` can depend on `rfx-sdk` and `lerobot`.

## Migration Plan

1. Keep current imports working from `rfx`.
2. Move provider-specific code into `packages/rfx-go2`.
3. Move simulator-specific extras into `packages/rfx-sim`.
4. Keep `rfx.connect_robot(...)` stable and route through installed providers.

## Run Example

```bash
uv run --python 3.13 rfx/examples/universal_go2.py --backend genesis --auto-install
```
