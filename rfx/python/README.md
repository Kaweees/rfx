# rfx Python Package

Python package root. The `rfx/` subdirectory is the importable package (`import rfx`).

## Key Modules

- **`robot.py`** -- Robot protocol (`observe`/`act`/`reset`) and RobotBase ABC
- **`config.py`** -- RobotConfig, JointConfig, CameraConfig dataclasses
- **`nn.py`** -- Policy base class, MLP, ActorCritic, policy registry (`@register_policy`), self-describing save/load (directory-based with `rfx_config.json` + `model.safetensors`)
- **`hub.py`** -- `load_policy()`, `push_policy()`, `inspect_policy()`, LoadedPolicy (callable wrapper with automatic torch/tinygrad bridging)
- **`session.py`** -- Session inference loop with rate control, jitter tracking, and `rfx.run()` convenience function
- **`observation.py`** -- ObservationSpec, make_observation, ObservationBuffer
- **`skills.py`** -- `@skill` decorator, Skill dataclass, SkillRegistry
- **`agent.py`** -- LLM Agent (Anthropic/OpenAI backends), MockAgent for testing
- **`decorators.py`** -- `@control_loop`, `@policy`, MotorCommands
- **`jit.py`** -- PolicyJitRuntime, rfxJIT integration

## Subpackages

- **`sim/`** -- Simulation backends: SimRobot, MockRobot, Genesis (GPU), MJX (JAX)
- **`real/`** -- Real hardware backends: So101Backend, Go2Backend, Camera
- **`teleop/`** -- Bimanual SO-101 teleoperation, transport layer, LeRobot recording
- **`utils/`** -- Padding, normalization (with serialization), action chunking utilities
