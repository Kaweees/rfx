# rfx-core

Core Rust library for rfx. Provides the foundational types and drivers used by the Python API via PyO3 bindings.

## Modules

- **`math/`** -- Quaternion (Hamilton convention), Transform (SE3), LowPassFilter, PID controller with integral windup protection
- **`control/`** -- ControlLoop, state machines for robot lifecycle management
- **`comm/`** -- Channels, topics, streams, and in-process keyed transport for inter-component communication
- **`hardware/`** -- Hardware drivers:
  - Go2: Unitree Go2 quadruped via DDS protocol (12 motors, IMU)
  - SO-101: 6-DOF arm via USB serial (Rust serial driver)
- **`neural/`** -- ObservationSpace and ActionSpace definitions
- **`sim/`** -- SimBackend trait, MockSimBackend for testing, SimConfig

## Cargo Features

- `hardware-go2` (default) -- Enables Go2 DDS driver
- `hardware-so101` (default) -- Enables SO-101 serial driver
- `dds-cyclone` (optional) -- Use CycloneDDS backend instead of dust_dds
