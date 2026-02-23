# rfx.teleop

Teleoperation stack for bimanual SO-101 recording at 350 Hz.

## Modules

- **`config.py`** -- TeleopSessionConfig and sub-configs: ArmPairConfig, CameraStreamConfig, TransportConfig, HybridConfig, JitPolicyConfig
- **`session.py`** -- BimanualSo101Session: main control loop with async camera capture and recording integration
- **`transport.py`** -- InprocTransport (Python), RustTransport (native Rust bindings), ZenohTransport, HybridTransport (local hot path + Zenoh control-plane), TransportEnvelope pub/sub messaging
- **`recorder.py`** -- LeRobotRecorder: JSONL + NumPy episode recording, plus LeRobot/MCAP export helpers
- **`lerobot_writer.py`** -- LeRobotPackageWriter: direct export to LeRobot dataset format
- **`mcap_writer.py`** -- McapEpisodeWriter: MCAP timeline export
- **`benchmark.py`** -- `run_jitter_benchmark` and `assert_jitter_budget` for CI timing gates

## OpenTelemetry Debugging

Enable runtime tracing for teleop and SO-101 backend:

```bash
export RFX_OTEL=1
export RFX_OTEL_EXPORTER=console        # or: otlp
export RFX_OTEL_SAMPLE_EVERY=100        # trace one control-loop tick every N iterations
# export RFX_OTEL_OTLP_ENDPOINT=http://localhost:4318/v1/traces
```

Install dependencies if needed:

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

## Beginner Unified Pattern

Use one robot spec call and one session call:

```python
from rfx.teleop import run, so101

arm = so101(leader_port="/dev/cu.usbmodemA", follower_port="/dev/cu.usbmodemB")
run(arm)
```

Session profiles are composable:

```python
from rfx.teleop import run, so101

arm = so101()
run(
    arm,
    logging=True,
    rate_hz=200,
    duration_s=20.0,
    cameras=[],
    data_output="demos",
    lineage="startup/so101-v1",
    scale="single-arm",
    format="native",
    otel=True,
)
```
