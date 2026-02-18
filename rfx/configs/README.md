# rfx Configs

Robot configuration YAML files used by rfx to define robot parameters.

## Files

- **`so101.yaml`** -- SO-101 6-DOF arm (state_dim=12, action_dim=6, 50 Hz control rate)
- **`so101_bimanual.yaml`** -- SO-101 bimanual teleop runtime config (left/right leader-follower pair ports + camera streams)
- **`go2.yaml`** -- Unitree Go2 quadruped (state_dim=34, action_dim=12, 200 Hz control rate), default URDF path: `rfx/assets/robots/go2/urdf/go2.urdf` (user-provided asset)

## Usage

```python
from rfx.config import RobotConfig
from rfx.sim import SimRobot

config = RobotConfig.from_yaml("so101.yaml")
robot = SimRobot.from_config("go2.yaml")
```

## Simulation Assets

Place robot assets under:

- `rfx/assets/robots/<robot>/urdf/`
- `rfx/assets/robots/<robot>/mjcf/`

Go2 expected layout:

- `rfx/assets/robots/go2/urdf/go2.urdf`
- `rfx/assets/robots/go2/mjcf/go2.xml` (for upcoming MuJoCo integration)
- Go2 files are not bundled in this repo/package.

## Config Search Paths

1. Current working directory
2. `rfx/configs/` package directory
3. `$RFX_CONFIG_DIR` environment variable
