# rfx-python

PyO3 bindings for rfx-core. Builds as the `_rfx` native extension module (cdylib).

## Exposed Types

- **Quaternion** -- Hamilton quaternion with arithmetic and SLERP
- **Transform** -- SE3 rigid body transform
- **LowPassFilter** -- Configurable low-pass filter
- **Pid / PidConfig** -- PID controller with tuning parameters
- **Go2 / Go2Config / Go2State** -- Unitree Go2 quadruped interface
- **Transport primitives** -- In-process pub/sub for inter-component messaging

## Building

```bash
# Debug build
maturin develop

# Release build
maturin develop --release
```

The Python side imports these types via `from rfx._rfx import ...` in `rfx/__init__.py`.
