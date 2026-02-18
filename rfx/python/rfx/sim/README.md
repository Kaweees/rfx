# rfx.sim

Simulation backends for rfx. All backends implement the Robot protocol (`observe`/`act`/`reset`).

## Modules

- **`base.py`** -- SimRobot: wraps any simulation backend and adds `get_reward`/`get_done` on top of the standard Robot interface
- **`mock.py`** -- MockRobot / MockBackend: zero-dependency spring-damper physics for unit testing and CI
- **`genesis.py`** -- GenesisBackend: GPU-accelerated simulation via genesis-world with URDF loading, viewer mode, and configurable physics settings
- **`mjx.py`** -- MjxBackend: JAX-accelerated MuJoCo with `jax.vmap` batching for parallel environments

## Genesis Quickstart

```python
from rfx.sim import SimRobot

robot = SimRobot.from_config(
    "rfx/configs/so101.yaml",
    backend="genesis",
    viewer=True,
    auto_install=True,  # installs genesis-world if missing
)
```
