# Python SDK

This document describes the Python SDK surface and package structure.

## Design Goals

- One simple Python API for users.
- Keep robot/provider specifics in extension packages.
- Allow simulation and hardware stacks to evolve independently.
- Every saved model is self-describing: load and deploy with zero context.

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

## Model Management

Policies in rfx are saved as self-describing directories. Every saved model bundles its weights, architecture, robot config, and normalizer state so it can be loaded with zero knowledge of how it was trained.

### Save

```python
from rfx.nn import MLP
from rfx.utils.transforms import ObservationNormalizer

policy = MLP(obs_dim=48, act_dim=12, hidden=[256, 256])
normalizer = ObservationNormalizer(state_dim=48)
# ... training loop ...

policy.save("runs/go2-walk-v1",
    robot_config=config,
    normalizer=normalizer,
    training_info={"total_steps": 50000, "best_reward": 245.3})
```

This creates:

```
runs/go2-walk-v1/
├── rfx_config.json        # Architecture + robot + training metadata
├── model.safetensors      # Weight tensors (tinygrad safe_save format)
└── normalizer.json        # ObservationNormalizer state (optional)
```

### Load

```python
loaded = rfx.load_policy("runs/go2-walk-v1")
loaded = rfx.load_policy("hf://rfx-community/go2-walk-v1")  # from HuggingFace Hub

loaded.policy           # The reconstructed tinygrad policy (MLP, ActorCritic, etc.)
loaded.robot_config     # RobotConfig or None
loaded.normalizer       # ObservationNormalizer or None
loaded.policy_type      # "MLP", "ActorCritic", etc.
loaded.training_info    # {"total_steps": 50000, ...}
```

`LoadedPolicy` is callable and handles torch/tinygrad conversion automatically, so it plugs directly into `rfx.run()`:

```python
robot = rfx.RealRobot("so101.yaml")
rfx.run(robot, loaded, rate_hz=50)
```

When called with a `dict[str, torch.Tensor]` (as returned by `robot.observe()`), it normalizes, converts to tinygrad, runs the policy, and converts the action back to torch.

### Inspect

Quick metadata check without loading weights:

```python
config = rfx.inspect_policy("runs/go2-walk-v1")
print(config["policy_type"])     # "MLP"
print(config["policy_config"])   # {"obs_dim": 48, "act_dim": 12, "hidden": [256, 256]}
```

### Share via HuggingFace Hub

```python
rfx.push_policy("runs/go2-walk-v1", "rfx-community/go2-walk-v1")
```

### Custom Policy Types

Register custom architectures so they can be auto-detected on load:

```python
from rfx.nn import Policy, register_policy

@register_policy
class MyTransformerPolicy(Policy):
    def __init__(self, obs_dim, act_dim, num_heads=4):
        ...

    def config_dict(self):
        return {"obs_dim": self.obs_dim, "act_dim": self.act_dim, "num_heads": self.num_heads}

    def forward(self, obs):
        ...
```

### End-to-End Example

```python
import rfx
from rfx.nn import MLP
from rfx.utils.transforms import ObservationNormalizer

# 1. Train
config = rfx.RobotConfig(name="Go2", state_dim=48, action_dim=12, control_freq_hz=200)
policy = MLP(obs_dim=48, act_dim=12, hidden=[256, 256])
normalizer = ObservationNormalizer(state_dim=48)
# ... training loop updates policy weights and normalizer stats ...

# 2. Save
policy.save("runs/go2-walk-v1",
    robot_config=config,
    normalizer=normalizer,
    training_info={"total_steps": 50000, "best_reward": 245.3})

# 3. Share
rfx.push_policy("runs/go2-walk-v1", "rfx-community/go2-walk-v1")

# 4. Load (on any machine)
loaded = rfx.load_policy("hf://rfx-community/go2-walk-v1")

# 5. Deploy
robot = rfx.RealRobot(loaded.robot_config)
rfx.run(robot, loaded, rate_hz=loaded.robot_config.control_freq_hz)
```

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
