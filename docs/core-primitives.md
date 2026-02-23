# Core Primitives

rfx organizes its API into focused package surfaces:

- `rfx.robot` — Robot protocol, config, URDF, factory functions
- `rfx.teleop` — Teleoperation sessions, transport, recording
- `rfx.collection` — Dataset recording and hub operations
- `rfx.runtime` — CLI, lifecycle, health, otel
- `rfx.sim` — Simulation backends

## Intent

- Keep teleop/session concerns isolated.
- Keep robot/hardware and discovery concerns isolated.
- Keep runtime/operations concerns isolated.
- Keep simulation as a separate package surface.

## Usage

```python
from rfx.robot import lerobot
from rfx.teleop import run

arm = lerobot.so101()
run(arm, logging=True)
```

Or access modules directly:

```python
import rfx

rfx.robot      # Robot protocol, config, URDF
rfx.teleop     # Teleop sessions, transport
rfx.collection # Data collection + dataset APIs
rfx.runtime    # CLI, lifecycle
rfx.sim        # Simulation backends
```
