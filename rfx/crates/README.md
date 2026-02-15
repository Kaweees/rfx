# rfx Rust Crates

Rust workspace containing the core library and Python bindings for rfx.

## Crates

### rfx-core

Core library providing math primitives (quaternion, SE3 transform, low-pass filter, PID), control loops, communication infrastructure (channels, topics, transport), hardware drivers (Unitree Go2 via DDS, SO-101 via serial), simulation backend traits, and neural observation/action space definitions.

Key dependencies: nalgebra, tokio, crossbeam, serde, dust_dds (optional), serialport (optional).

### rfx-python

PyO3 cdylib crate that exposes rfx-core types to Python as the `_rfx` native extension module. Binds Quaternion, Transform, LowPassFilter, Pid, Go2, and transport primitives for use from the Python API.

## Building

```bash
# Build native Python extension (development)
maturin develop

# Build Rust crates only
cargo build
```
