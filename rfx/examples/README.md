# rfx Examples

Runnable scripts demonstrating rfx capabilities.

## Scripts

- **`agent_control.py`** -- LLM agent controlling a robot with `@skill` functions
- **`teleop_record.py`** -- Bimanual teleoperation recording with LeRobot export
- **`deploy_policy.py`** -- Sim-to-real policy deployment
- **`deploy_real.py`** -- Real hardware policy deployment
- **`train_walking.py`** -- Go2 locomotion training
- **`balance.py`** -- Go2 balance controller
- **`walk.py`** -- Go2 walking gait
- **`train_vla.py`** -- Vision-language-action model training
- **`neural_policy.py`** -- Neural network policy example
- **`quick_experiment.py`** -- Rapid prototyping template
- **`genesis_viewer.py`** -- Live Genesis simulation viewer (headful)
- **`go2_walk_genesis.py`** -- Open-loop Go2 trot/walk demo in Genesis
- **`universal_go2.py`** -- Same command-level API across mock/genesis/real backends
- **`python_sdk_one_import.py`** -- Single `import rfx` workflow with provider activation
- **`so101_quickstart.py`** -- Minimal real-hardware SO101 quickstart

## Running

```bash
uv run rfx/examples/<script>.py
```

Notes:

- Go2 Genesis examples require user-provided Go2 URDF/mesh assets at `rfx/assets/robots/go2/urdf/`.
