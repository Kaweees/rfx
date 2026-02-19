<div align="center">

[<img alt="rfx logo" src="docs/assets/logo.svg" width="220" />](https://github.com/quantbagel/rfx)

**The robotics framework for the foundation model era.**

[Documentation](https://deepwiki.com/quantbagel/rfx) | [Discord](https://discord.gg/xV8bAGM8WT)

[![CI](https://github.com/quantbagel/rfx/actions/workflows/ci.yml/badge.svg)](https://github.com/quantbagel/rfx/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/xV8bAGM8WT)

</div>

---

```bash
uv pip install rfx-sdk
```

```python
import rfx

# Every robot, same interface
robot = rfx.RealRobot("so101.yaml")
obs = robot.observe()
robot.act(action)

# Every model, self-describing
loaded = rfx.load_policy("hf://rfx-community/go2-walk-v1")
rfx.run(robot, loaded, rate_hz=50)
```

---

## Why rfx

ROS was built for message passing between components. We're in a different era -- the workflow is **collect demos, train a policy, deploy, iterate**. rfx is built from scratch for that loop, with all the infrastructure you'd expect from a robotics framework underneath.

- **Rust core** for real-time control, Python SDK for fast research
- **Three-method robot interface** -- `observe()`, `act()`, `reset()` -- same API for sim and real
- **Self-describing models** -- save once, load anywhere, deploy with zero config
- **HuggingFace Hub native** -- push and pull policies like you push datasets
- **Zenoh transport** -- pub/sub topics, nodes, launch graphs -- all the ROS primitives, none of the pain
- **ROS 2 interop** -- coexist with existing ROS 2 stacks via zenoh-plugin-ros2dds bridge
- **Batteries included** -- simulation (Genesis, MJX), teleoperation, LeRobot export, hardware drivers

## Install

```bash
uv pip install rfx-sdk
```

With simulation and robot extras:

```bash
uv pip install rfx-sdk rfx-sdk-sim rfx-sdk-go2 rfx-sdk-lerobot
```

From source:

```bash
git clone https://github.com/quantbagel/rfx.git && cd rfx
bash scripts/setup-from-source.sh
```

## The interface

Every robot in rfx -- simulated or real -- implements the same three methods:

```python
robot = rfx.SimRobot.from_config("go2.yaml", backend="genesis")
# robot = rfx.RealRobot("so101.yaml", port="/dev/ttyACM0")

obs = robot.observe()    # {"state": Tensor(1, 64), "images": ...}
robot.act(action)        # Tensor(1, 64)
robot.reset()
```

Run a policy against any robot with one call:

```python
rfx.run(robot, policy, rate_hz=200, duration=30.0)
```

Rate-controlled loop with jitter tracking, error handling, and clean shutdown built in.

## Train

```python
from rfx.nn import MLP
from rfx.utils.transforms import ObservationNormalizer

policy = MLP(obs_dim=48, act_dim=12, hidden=[256, 256])
normalizer = ObservationNormalizer(state_dim=48)

# ... your training loop ...
```

tinygrad-native policies. MLP, ActorCritic, or subclass `Policy` for your own architecture.

## Save

Every saved model is a self-describing directory. Weights, architecture, robot config, normalizer -- everything needed to reconstruct and deploy.

```python
policy.save("runs/go2-walk-v1",
    robot_config=config,
    normalizer=normalizer,
    training_info={"total_steps": 50000, "best_reward": 245.3})
```

```
runs/go2-walk-v1/
  rfx_config.json       # architecture + robot + training metadata
  model.safetensors     # weights
  normalizer.json       # observation normalizer state
```

Push to HuggingFace Hub:

```python
rfx.push_policy("runs/go2-walk-v1", "rfx-community/go2-walk-v1")
```

## Load and deploy

Load from disk or Hub. No need to know the architecture, hyperparameters, or training setup.

```python
loaded = rfx.load_policy("runs/go2-walk-v1")
loaded = rfx.load_policy("hf://rfx-community/go2-walk-v1")

loaded.policy_type       # "MLP"
loaded.robot_config      # RobotConfig(name="Go2", ...)
loaded.training_info     # {"total_steps": 50000, "best_reward": 245.3}

# Deploy -- torch/tinygrad conversion handled automatically
robot = rfx.RealRobot(loaded.robot_config)
rfx.run(robot, loaded, rate_hz=loaded.robot_config.control_freq_hz)
```

Inspect metadata without loading weights:

```python
rfx.inspect_policy("runs/go2-walk-v1")
```

## Supported hardware

| Robot | Type | Interface | Status |
|-------|------|-----------|--------|
| **SO-101** | 6-DOF arm | USB serial (Rust driver) | Ready |
| **Unitree Go2** | Quadruped | Ethernet DDS (Zenoh/dust_dds) | Ready |

Custom robots: implement `observe()` / `act()` / `reset()` or write a YAML config with URDF.

## Simulation

```python
# Genesis (GPU-accelerated)
robot = rfx.SimRobot.from_config("so101.yaml", backend="genesis", viewer=True)

# MJX (JAX-accelerated MuJoCo)
robot = rfx.SimRobot.from_config("go2.yaml", backend="mjx", num_envs=4096)

# Mock (zero dependencies, for testing)
robot = rfx.MockRobot(state_dim=12, action_dim=6)
```

## Teleoperation

Bimanual SO-101 recording at 350 Hz with LeRobot export:

```python
from rfx.teleop import BimanualSo101Session, TeleopSessionConfig

config = TeleopSessionConfig.from_yaml("so101_bimanual.yaml")
session = BimanualSo101Session(config)
session.run()
```

## Communication and runtime

Under the hood, rfx has a full robotics communication stack built on [Zenoh](https://zenoh.io) -- topics, nodes, launch files, graph introspection. It's there when you need it, invisible when you don't.

```python
from rfx.teleop import create_transport

# Pub/sub messaging -- Rust-backed when available, Python fallback otherwise
transport = create_transport(backend="zenoh")
transport.publish("robot/state", state_bytes)
transport.subscribe("robot/cmd", callback)
```

Nodes follow a simple lifecycle contract:

```python
from rfx.runtime.node import Node

class ControlNode(Node):
    publish_topics = ("robot/cmd",)
    subscribe_topics = ("robot/state",)

    def setup(self):    ...   # init
    def tick(self):     ...   # one loop iteration
    def shutdown(self): ...   # cleanup
```

Launch multiple nodes from a YAML graph:

```yaml
name: go2-deploy
nodes:
  - package: go2_ctrl
    node: policy_node
    rate_hz: 200
  - package: go2_ctrl
    node: logger_node
    rate_hz: 10
```

```bash
rfx launch go2_deploy.yaml
rfx graph        # inspect the active node graph
rfx topic-list   # see all live topics
```

**ROS 2 coexistence**: rfx topics are visible to ROS 2 tools via the [zenoh-plugin-ros2dds](https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds) bridge. Migrate incrementally -- run rfx nodes alongside existing ROS 2 nodes on the same DDS domain, no code changes required on either side.

**Enterprise scaling**: The same code that runs on one robot runs on a fleet. No architecture changes, no extra infra. [Get in touch](https://discord.gg/xV8bAGM8WT) -- we'll handle scaling so you don't have to.

## rfxJIT

Built-in kernel compiler that lowers and executes across `cpu`, `cuda`, and `metal`. IR-based autodiff, optimization passes (constant folding, dead-op elimination, fusion), and a tinyJIT-style cache+replay runtime.

```python
from rfx.jit import value_and_grad, available_backends

# JAX-style functional transforms
loss_and_grads = value_and_grad(loss_fn)

# Check what's available on this machine
available_backends()  # {"cpu": True, "cuda": True, "metal": False}
```

Enable globally via environment:

```bash
export RFX_JIT=1                   # enable rfxJIT execution paths
export RFX_JIT_BACKEND=auto        # auto | cpu | cuda | metal
```

When enabled, policy inference automatically routes through rfxJIT when possible, with transparent fallback to tinygrad's TinyJit.

## Custom policies

Register your own architectures for automatic save/load:

```python
from rfx.nn import Policy, register_policy

@register_policy
class MyPolicy(Policy):
    def __init__(self, obs_dim, act_dim):
        self.obs_dim, self.act_dim = obs_dim, act_dim
        # ... build layers ...

    def config_dict(self):
        return {"obs_dim": self.obs_dim, "act_dim": self.act_dim}

    def forward(self, obs):
        # ... your forward pass ...
```

## Docs

- [Full documentation](https://deepwiki.com/quantbagel/rfx)
- [SO-101 quickstart](docs/so101.md)
- [Simulation guide](docs/sim.md)
- [Python SDK reference](docs/python-sdk.md)
- [Contributor workflow](docs/workflow.md)

## Community

- [Issues](https://github.com/quantbagel/rfx/issues)
- [Discussions](https://github.com/quantbagel/rfx/discussions)
- [Discord](https://discord.gg/xV8bAGM8WT)
- [Contributing](CONTRIBUTING.md)

## License

MIT
