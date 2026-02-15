# rfx Configs

Robot configuration YAML files used by rfx to define robot parameters.

## Files

- **`so101.yaml`** -- SO-101 6-DOF arm (state_dim=12, action_dim=6, 50 Hz control rate)
- **`go2.yaml`** -- Unitree Go2 quadruped (state_dim=36, action_dim=12, 200 Hz control rate)

## Usage

```python
from rfx.config import RobotConfig
from rfx.sim import SimRobot

config = RobotConfig.from_yaml("so101.yaml")
robot = SimRobot.from_config("go2.yaml")
```

## Config Search Paths

1. Current working directory
2. `rfx/configs/` package directory
3. `$RFX_CONFIG_DIR` environment variable
