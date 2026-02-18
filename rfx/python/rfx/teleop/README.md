# rfx.teleop

Teleoperation stack for bimanual SO-101 recording at 350 Hz.

## Modules

- **`config.py`** -- TeleopSessionConfig and sub-configs: ArmPairConfig, CameraStreamConfig, TransportConfig, JitPolicyConfig
- **`session.py`** -- BimanualSo101Session: main control loop with async camera capture and recording integration
- **`transport.py`** -- InprocTransport (Python), RustTransport (native Rust bindings), TransportEnvelope pub/sub messaging
- **`recorder.py`** -- LeRobotRecorder: JSONL + NumPy episode recording, plus LeRobot/MCAP export helpers
- **`lerobot_writer.py`** -- LeRobotPackageWriter: direct export to LeRobot dataset format
- **`mcap_writer.py`** -- McapEpisodeWriter: MCAP timeline export
- **`benchmark.py`** -- `run_jitter_benchmark` and `assert_jitter_budget` for CI timing gates
