# rfx

Primary source tree for rfx, the PyTorch for robots. This directory contains the Rust core, Python API, tests, configs, and examples.

## Directory Structure

- **`crates/`** -- Rust workspace with two crates:
  - `rfx-core` -- core library: math primitives, control loops, hardware drivers, communication, and neural space definitions
  - `rfx-python` -- PyO3 bindings that expose Rust types to Python as the `_rfx` native module
- **`python/`** -- Python package with submodules for the robot protocol, config, simulation backends, real hardware backends, teleoperation, agents, skills, decorators, observation processing, and JIT integration
- **`tests/`** -- pytest suite covering unit tests and integration tests for Rust bindings
- **`configs/`** -- YAML robot configuration files (e.g. `so101.yaml`, `go2.yaml`)
- **`examples/`** -- runnable scripts demonstrating agent control, teleoperation recording, policy deployment, and training

## Core Interface

All robots implement the three-method protocol:

```python
observation = robot.observe()
robot.act(action)
robot.reset()
```

This interface is consistent across simulation, real hardware, and teleoperation.
