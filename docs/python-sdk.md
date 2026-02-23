# Python SDK

This document describes the Python SDK surface and package structure.

## Design Goals

- One simple Python API for users.
- Keep robot/provider specifics in extension packages.
- Allow simulation and hardware stacks to evolve independently.
- Every saved model is self-describing: load and deploy with zero context.

## SDK Entry Point

```python
from rfx.robot import lerobot
from rfx.teleop import run

arm = lerobot.so101()
run(arm, logging=True)
```

Primary surfaces:

- `rfx.robot` — Robot protocol, config, URDF, factory functions (`rfx.robot.lerobot`)
- `rfx.teleop` — Teleoperation sessions, transport, recording
- `rfx.collection` — Dataset recording, hub push/pull, collection helpers
- `rfx.runtime` — CLI, lifecycle, health, otel
- `rfx.sim` — Simulation backends
- `rfx.nn` / `rfx.rl` — Neural policies and RL training

When `config` is omitted, rfx uses built-in defaults (`GO2_CONFIG`/`SO101_CONFIG`) so examples work from wheel installs without repo-local YAML files.

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

```
rfx/python/rfx/
├── robot/          # Robot protocol, config, URDF, lerobot factories
├── teleop/         # Teleoperation sessions, transport, recording
├── collection/     # Dataset recording, hub integration, collect CLI helpers
├── real/           # Real hardware backends (SO-101, Go2, G1)
├── sim/            # Simulation backends (MuJoCo, Genesis, mock)
├── nn/             # Neural network policies (MLP, ActorCritic)
├── rl/             # RL training loops
├── envs/           # Gym-style environments
├── runtime/        # CLI, lifecycle, health, otel, dora bridge
├── drivers/        # Hardware driver registry
├── tf/             # Transform broadcaster/listener
├── workflow/       # Training workflow stages
├── agent.py        # LLM agent integration
├── skills.py       # Skill registry
├── hub.py          # Model save/load/push
├── session.py      # Session runner
├── decorators.py   # @control_loop, @policy, MotorCommands
├── jit.py          # JIT compilation runtime
├── node.py         # Zenoh transport factory & discovery
└── observation.py  # Observation spec & padding
```

## Boundary Rules

- `rfx-sdk` (distribution) should not depend on robot-specific hardware packages.
- Simulation backends are optional (installed via `rfx-sdk[teleop]`).
- LeRobot integration is optional (installed via `rfx-sdk[teleop-lerobot]`).

## Run Example

```bash
uv run --python 3.13 rfx/examples/universal_go2.py --backend genesis --auto-install
```
