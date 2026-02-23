"""
rfx.teleop.session - High-rate SO-101 teleoperation runtime.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..runtime.otel import flush_otel, get_tracer, init_otel
from .config import ArmPairConfig, CameraStreamConfig, TeleopSessionConfig
from .recorder import LeRobotRecorder, RecordedEpisode
from .transport import TransportLike, create_transport


@dataclass(frozen=True)
class LoopTimingStats:
    """Loop timing summary for jitter and overrun analysis."""

    iterations: int
    overruns: int
    target_period_s: float
    avg_period_s: float
    p50_jitter_s: float
    p95_jitter_s: float
    p99_jitter_s: float
    max_jitter_s: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "iterations": int(self.iterations),
            "overruns": int(self.overruns),
            "target_period_s": float(self.target_period_s),
            "avg_period_s": float(self.avg_period_s),
            "p50_jitter_s": float(self.p50_jitter_s),
            "p95_jitter_s": float(self.p95_jitter_s),
            "p99_jitter_s": float(self.p99_jitter_s),
            "max_jitter_s": float(self.max_jitter_s),
        }


class ArmPair(Protocol):
    """Protocol for leader/follower arm pairs used by the session loop."""

    name: str

    def step(self) -> Sequence[float]: ...

    def go_home(self) -> None: ...

    def disconnect(self) -> None: ...


class _So101ArmPair:
    """SO-101 leader/follower arm pair using the universal Zenoh pipeline.

    Both arms are RobotNodes on a shared Rust transport.  The leader
    publishes state at 50 Hz; ``step()`` reads the latest state and sends
    it as a command to the follower — all through the transport, never
    touching serial from Python.
    """

    def __init__(self, pair: ArmPairConfig, **kwargs: Any) -> None:
        from ..real.so101 import So101Backend

        self.name = pair.name

        # Create a shared Rust transport for both RobotNodes.
        # Default to in-process transport for local teleop reliability.
        # Set RFX_TELEOP_NODE_TRANSPORT=zenoh to force Zenoh bus.
        transport_mode = os.getenv("RFX_TELEOP_NODE_TRANSPORT", "inproc").strip().lower()
        if transport_mode == "zenoh":
            from .. import node as _node_mod

            rust_transport = _node_mod.auto_transport()
        else:
            from rfx._rfx import Transport as _RustTransport

            rust_transport = _RustTransport()

        from ..robot.config import SO101_CONFIG

        self._leader = So101Backend(
            config=SO101_CONFIG,
            port=pair.leader_port,
            is_leader=True,
            transport=rust_transport,
            name=f"{pair.name}-leader",
        )
        self._follower = So101Backend(
            config=SO101_CONFIG,
            port=pair.follower_port,
            is_leader=False,
            transport=rust_transport,
            name=f"{pair.name}-follower",
        )

    def step(self) -> Sequence[float]:
        positions = self._leader.read_positions()
        # Send positions directly through the follower's node (via transport)
        self._follower.node.send_command(positions.tolist())
        return [float(v) for v in positions.tolist()]

    def go_home(self) -> None:
        self._follower.go_home()

    def disconnect(self) -> None:
        self._leader.disconnect()
        self._follower.disconnect()


class _CameraWorker:
    """Asynchronous camera reader that captures frames independently of control loop."""

    def __init__(
        self,
        config: CameraStreamConfig,
        frame_callback: Callable[[str, int, int, np.ndarray], None],
    ) -> None:
        self.config = config
        self._frame_callback = frame_callback

        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_count = 0
        self._latest_frame_index = -1
        self._latest_timestamp_ns = 0
        self._error: Exception | None = None
        self._state_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-{self.config.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def latest_frame_index(self) -> int:
        with self._state_lock:
            return self._latest_frame_index

    @property
    def latest_timestamp_ns(self) -> int:
        with self._state_lock:
            return self._latest_timestamp_ns

    @property
    def frame_count(self) -> int:
        with self._state_lock:
            return self._frame_count

    @property
    def error(self) -> Exception | None:
        return self._error

    def _run(self) -> None:
        try:
            from ..real.camera import Camera
        except Exception as exc:  # pragma: no cover - environment dependent
            self._error = exc
            return

        period_s = 1.0 / max(self.config.fps, 1)
        camera = Camera(
            device_id=self.config.device_id,
            resolution=(self.config.width, self.config.height),
            fps=self.config.fps,
        )

        frame_index = -1
        try:
            while self._running.is_set():
                tick_start = time.perf_counter()
                frame = camera.capture()
                frame_index += 1
                ts_ns = time.time_ns()
                frame_array = self._to_numpy(frame)

                with self._state_lock:
                    self._frame_count = frame_index + 1
                    self._latest_frame_index = frame_index
                    self._latest_timestamp_ns = ts_ns

                self._frame_callback(self.config.name, frame_index, ts_ns, frame_array)

                elapsed = time.perf_counter() - tick_start
                sleep_s = period_s - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except Exception as exc:  # pragma: no cover - environment dependent
            self._error = exc
        finally:
            camera.release()

    @staticmethod
    def _to_numpy(frame: Any) -> np.ndarray:
        value = frame
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)


class BimanualSo101Session:
    """Python-first high-rate teleop session with async cameras and recorder integration."""

    def __init__(
        self,
        config: TeleopSessionConfig,
        *,
        recorder: LeRobotRecorder | None = None,
        collection_recorder: Any | None = None,
        pair_factory: Callable[[ArmPairConfig], ArmPair] | None = None,
        transport: TransportLike | None = None,
    ) -> None:
        self.config = config
        self.recorder = recorder or LeRobotRecorder(config.output_dir)
        self._collection_recorder = collection_recorder
        self._pair_factory = pair_factory or _So101ArmPair
        self.transport = transport or create_transport(config.transport)
        init_otel(service_name="rfx-teleop")
        self._tracer = get_tracer("rfx.teleop.session")
        self._otel_sample_every = max(1, int(os.getenv("RFX_OTEL_SAMPLE_EVERY", "100")))
        self._trace_every = max(0, int(os.getenv("RFX_TELEOP_TRACE_EVERY", "0")))

        self._pairs: list[ArmPair] = []
        self._camera_workers: list[_CameraWorker] = []
        self._control_thread: threading.Thread | None = None
        self._running = threading.Event()
        self._state_lock = threading.Lock()
        self._record_lock = threading.Lock()
        self._active_episode_id: str | None = None
        self._latest_positions: dict[str, tuple[float, ...]] = {}
        self._latest_camera_indices: dict[str, int] = {
            camera.name: -1 for camera in self.config.cameras
        }
        self._last_timestamp_ns: int = 0
        self._loop_error: Exception | None = None

        self._period_samples = deque(maxlen=self.config.max_timing_samples)
        self._jitter_samples = deque(maxlen=self.config.max_timing_samples)
        self._iterations = 0
        self._overruns = 0

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def is_recording(self) -> bool:
        with self._record_lock:
            return self._active_episode_id is not None

    @classmethod
    def from_ports(
        cls,
        *,
        left_leader_port: str = "/dev/ttyACM0",
        left_follower_port: str = "/dev/ttyACM1",
        right_leader_port: str = "/dev/ttyACM2",
        right_follower_port: str = "/dev/ttyACM3",
        **kwargs: Any,
    ) -> BimanualSo101Session:
        config = TeleopSessionConfig.bimanual(
            left_leader_port=left_leader_port,
            left_follower_port=left_follower_port,
            right_leader_port=right_leader_port,
            right_follower_port=right_follower_port,
            **kwargs,
        )
        return cls(config=config)

    @classmethod
    def from_single_pair(
        cls,
        *,
        leader_port: str = "/dev/ttyACM0",
        follower_port: str = "/dev/ttyACM1",
        **kwargs: Any,
    ) -> BimanualSo101Session:
        config = TeleopSessionConfig.single_arm_pair(
            name="main",
            leader_port=leader_port,
            follower_port=follower_port,
            **kwargs,
        )
        return cls(config=config)

    def start(self) -> None:
        if self.is_running:
            return
        with self._tracer.start_as_current_span("teleop.start") as span:
            span.set_attribute("rate_hz", float(self.config.rate_hz))
            span.set_attribute("arm_pairs", len(self.config.arm_pairs))

            self._pairs = [self._pair_factory(pair_config) for pair_config in self.config.arm_pairs]
            self._loop_error = None

            self._camera_workers = [
                _CameraWorker(camera_config, frame_callback=self._on_camera_frame)
                for camera_config in self.config.cameras
            ]
            for worker in self._camera_workers:
                worker.start()

            self._running.set()
            self._control_thread = threading.Thread(
                target=self._control_loop,
                name="teleop-control-loop",
                daemon=True,
            )
            self._control_thread.start()

    def stop(self) -> None:
        if not self.is_running and self._control_thread is None:
            return
        with self._tracer.start_as_current_span("teleop.stop") as span:
            span.set_attribute("recording_active", bool(self.is_recording))

            if self.is_recording:
                self.stop_recording()

            self._running.clear()
            if self._control_thread is not None:
                self._control_thread.join(timeout=2.0)
                self._control_thread = None

            for worker in self._camera_workers:
                worker.stop()
            self._camera_workers = []

            for pair in self._pairs:
                pair.disconnect()
            self._pairs = []
            flush_otel()

    def go_home(self) -> None:
        for pair in self._pairs:
            pair.go_home()

    def latest_positions(self) -> dict[str, tuple[float, ...]]:
        with self._state_lock:
            return dict(self._latest_positions)

    def latest_camera_frame_indices(self) -> dict[str, int]:
        with self._state_lock:
            return dict(self._latest_camera_indices)

    def latest_timestamp_ns(self) -> int:
        with self._state_lock:
            return int(self._last_timestamp_ns)

    def reset_timing_stats(self) -> None:
        """Reset accumulated loop timing counters and histograms."""
        with self._state_lock:
            self._period_samples.clear()
            self._jitter_samples.clear()
            self._iterations = 0
            self._overruns = 0

    def start_recording(
        self,
        *,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        with self._record_lock:
            if self._active_episode_id is not None:
                raise RuntimeError("Recording is already active")
            episode_id = self.recorder.start_episode(
                session_config=self.config,
                label=label,
                metadata=metadata,
            )
            self._active_episode_id = episode_id
            return episode_id

    def stop_recording(self) -> RecordedEpisode:
        with self._record_lock:
            if self._active_episode_id is None:
                raise RuntimeError("Recording is not active")
            self._active_episode_id = None
        return self.recorder.finalize_episode(loop_stats=self.timing_stats().to_dict())

    def record_episode(
        self,
        *,
        duration_s: float,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RecordedEpisode:
        if duration_s <= 0:
            raise ValueError("duration_s must be > 0")

        if not self.is_running:
            self.start()

        self.start_recording(label=label, metadata=metadata)
        deadline = time.perf_counter() + duration_s
        while self.is_running and time.perf_counter() < deadline:
            time.sleep(0.01)
        return self.stop_recording()

    def timing_stats(self) -> LoopTimingStats:
        with self._state_lock:
            period_samples = np.asarray(tuple(self._period_samples), dtype=np.float64)
            jitter_samples = np.asarray(tuple(self._jitter_samples), dtype=np.float64)
            iterations = self._iterations
            overruns = self._overruns

        if period_samples.size == 0:
            return LoopTimingStats(
                iterations=iterations,
                overruns=overruns,
                target_period_s=self.config.period_s,
                avg_period_s=0.0,
                p50_jitter_s=0.0,
                p95_jitter_s=0.0,
                p99_jitter_s=0.0,
                max_jitter_s=0.0,
            )

        return LoopTimingStats(
            iterations=iterations,
            overruns=overruns,
            target_period_s=self.config.period_s,
            avg_period_s=float(np.mean(period_samples)),
            p50_jitter_s=float(np.percentile(jitter_samples, 50)),
            p95_jitter_s=float(np.percentile(jitter_samples, 95)),
            p99_jitter_s=float(np.percentile(jitter_samples, 99)),
            max_jitter_s=float(np.max(jitter_samples)),
        )

    def check_health(self) -> None:
        if self._loop_error is not None:
            raise RuntimeError("Control loop failed") from self._loop_error
        for worker in self._camera_workers:
            if worker.error is not None:
                raise RuntimeError(f"Camera worker failed: {worker.config.name}") from worker.error

    def _control_loop(self) -> None:
        target_period = self.config.period_s
        next_deadline = time.perf_counter()
        last_loop_start = next_deadline

        while self._running.is_set():
            loop_start = time.perf_counter()
            dt = loop_start - last_loop_start
            last_loop_start = loop_start
            should_trace_tick = (self._iterations % self._otel_sample_every) == 0
            span_ctx = (
                self._tracer.start_as_current_span("teleop.tick") if should_trace_tick else None
            )

            try:
                if span_ctx is not None:
                    with span_ctx as span:
                        span.set_attribute("dt_s", float(dt))
                        span.set_attribute("overruns", int(self._overruns))
                        pair_positions = self._step_pairs()
                else:
                    pair_positions = self._step_pairs()
            except Exception as exc:
                self._loop_error = exc
                self._running.clear()
                with self._tracer.start_as_current_span("teleop.loop_error") as span:
                    span.set_attribute("error_type", type(exc).__name__)
                    span.add_event("control_loop_exception", {"message": str(exc)})
                break

            timestamp_ns = time.time_ns()
            camera_indices = self.latest_camera_frame_indices()

            with self._state_lock:
                self._latest_positions = pair_positions
                self._last_timestamp_ns = timestamp_ns
                self._iterations += 1
                if dt > target_period:
                    self._overruns += 1
                jitter = abs(dt - target_period)
                self._period_samples.append(dt)
                self._jitter_samples.append(jitter)

            with self._record_lock:
                recording = self._active_episode_id is not None
            if recording:
                self.recorder.append_control_step(
                    timestamp_ns=timestamp_ns,
                    dt_s=dt,
                    pair_positions=pair_positions,
                    camera_frame_indices=camera_indices,
                )
                if self._collection_recorder is not None:
                    state = np.concatenate(
                        [np.asarray(v, dtype=np.float32) for v in pair_positions.values()]
                    )
                    self._collection_recorder.add_frame(state=state)
            self._publish_pair_positions(
                pair_positions=pair_positions,
                camera_indices=camera_indices,
                timestamp_ns=timestamp_ns,
            )
            self._maybe_log_live_trace(pair_positions=pair_positions, dt_s=dt)

            next_deadline += target_period
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s > 0:
                # Hybrid wait strategy: coarse sleep followed by a short busy spin.
                # Avoiding `sleep(0)` near deadline reduces OS re-schedule jitter spikes.
                spin_window_s = 0.0012
                if sleep_s > spin_window_s:
                    time.sleep(sleep_s - spin_window_s)
                while time.perf_counter() < next_deadline:
                    pass
            else:
                next_deadline = time.perf_counter()

    def _maybe_log_live_trace(
        self, *, pair_positions: Mapping[str, Sequence[float]], dt_s: float
    ) -> None:
        if self._trace_every <= 0:
            return
        if (self._iterations % self._trace_every) != 0:
            return
        compact = {
            pair_name: [round(float(v), 4) for v in values[:6]]
            for pair_name, values in pair_positions.items()
        }
        print(
            f"teleop.live dt={dt_s:.4f}s overruns={self._overruns} cmd={compact}",
            flush=True,
        )

    def _step_pairs(self) -> dict[str, tuple[float, ...]]:
        pair_positions: dict[str, tuple[float, ...]] = {}
        for pair in self._pairs:
            values = tuple(float(v) for v in pair.step())
            pair_positions[pair.name] = values
        return pair_positions

    def _publish_pair_positions(
        self,
        *,
        pair_positions: Mapping[str, Sequence[float]],
        camera_indices: Mapping[str, int],
        timestamp_ns: int,
    ) -> None:
        if self.transport.subscriber_count <= 0:
            return

        for pair_name, values in pair_positions.items():
            payload = np.asarray(values, dtype=np.float32)
            self.transport.publish(
                key=f"teleop/{pair_name}/state",
                payload=memoryview(payload),
                timestamp_ns=timestamp_ns,
                metadata={
                    "dtype": "float32",
                    "shape": [int(payload.size)],
                    "camera_frame_indices": dict(camera_indices),
                },
            )

    def _on_camera_frame(
        self,
        camera_name: str,
        frame_index: int,
        timestamp_ns: int,
        frame: np.ndarray,
    ) -> None:
        with self._state_lock:
            self._latest_camera_indices[camera_name] = int(frame_index)

        with self._record_lock:
            recording = self._active_episode_id is not None
        if recording:
            self.recorder.append_camera_frame(
                camera_name=camera_name,
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                frame=frame,
            )

        if self.transport.subscriber_count > 0:
            self.transport.publish(
                key=f"teleop/camera/{camera_name}/frame_index",
                payload=json.dumps({"frame_index": int(frame_index)}).encode("utf-8"),
                timestamp_ns=timestamp_ns,
            )

    def __enter__(self) -> BimanualSo101Session:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def run(self) -> None:
        """Run teleop until Ctrl+C using this session's arm configuration."""
        self.start()
        print("Starting teleoperation. Press Ctrl+C to stop.")
        try:
            while True:
                self.check_health()
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            self.stop()


class So101TeleopSession(BimanualSo101Session):
    """Generic SO-101 teleop session.

    Name keeps UX simple: use one session class for single-pair today, and
    multiple pairs later by changing session config only.
    """


@dataclass(frozen=True)
class SessionVars:
    """Run-level settings shared by teleop/sim session wrappers."""

    rate_hz: float | None = None
    duration_s: float | None = None
    output_dir: str | Path | None = None
    cameras: tuple[CameraStreamConfig, ...] | None = None
    transport: Any | None = None
    jit: Any | None = None
    warmup_s: float = 0.5


@dataclass(frozen=True)
class DataVars:
    """Data collection policy attached to a session run."""

    enabled: bool = False
    duration_s: float | None = None
    label: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TelemetryVars:
    """Observability settings applied before session execution."""

    otel: bool = False
    exporter: str = "console"
    sample_every: int = 100
    live_trace_every: int = 0
    otlp_endpoint: str | None = None


@dataclass(frozen=True)
class SimSpec:
    """Simulation robot spec for session.run."""

    config: str | Path | Mapping[str, Any]
    num_envs: int = 1
    backend: str = "mock"
    device: str | None = None
    policy: Callable[[dict[str, Any]], Any] | None = None


def config(
    *,
    rate_hz: float | None = 50.0,
    duration_s: float | None = None,
    output_dir: str | Path | None = None,
    cameras: Sequence[CameraStreamConfig] | None = None,
    transport: Any | None = None,
    jit: Any | None = None,
    warmup_s: float = 0.5,
) -> SessionVars:
    return SessionVars(
        rate_hz=rate_hz,
        duration_s=duration_s,
        output_dir=output_dir,
        cameras=tuple(cameras) if cameras is not None else None,
        transport=transport,
        jit=jit,
        warmup_s=warmup_s,
    )


def data(
    *,
    enabled: bool = True,
    duration_s: float | None = None,
    label: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DataVars:
    return DataVars(enabled=enabled, duration_s=duration_s, label=label, metadata=metadata)


def telemetry(
    *,
    otel: bool = True,
    exporter: str = "console",
    sample_every: int = 100,
    live_trace_every: int = 0,
    otlp_endpoint: str | None = None,
) -> TelemetryVars:
    return TelemetryVars(
        otel=otel,
        exporter=exporter,
        sample_every=sample_every,
        live_trace_every=live_trace_every,
        otlp_endpoint=otlp_endpoint,
    )


def sim(
    config: str | Path | Mapping[str, Any],
    *,
    num_envs: int = 1,
    backend: str = "mock",
    device: str | None = None,
    policy: Callable[[dict[str, Any]], Any] | None = None,
) -> SimSpec:
    return SimSpec(
        config=config,
        num_envs=num_envs,
        backend=backend,
        device=device,
        policy=policy,
    )


def _apply_telemetry(spec: TelemetryVars | None) -> None:
    if spec is None:
        return
    if spec.otel:
        os.environ["RFX_OTEL"] = "1"
    os.environ["RFX_OTEL_EXPORTER"] = str(spec.exporter)
    os.environ["RFX_OTEL_SAMPLE_EVERY"] = str(max(1, int(spec.sample_every)))
    os.environ["RFX_TELEOP_TRACE_EVERY"] = str(max(0, int(spec.live_trace_every)))
    if spec.otlp_endpoint:
        os.environ["RFX_OTEL_OTLP_ENDPOINT"] = str(spec.otlp_endpoint)


def _parse_profiles(
    variables: Any | None,
    profiles: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> tuple[SessionVars, DataVars, TelemetryVars | None]:
    session_spec = SessionVars()
    data_spec = DataVars()
    telemetry_spec: TelemetryVars | None = None

    inputs: list[Any] = []
    if variables is not None:
        inputs.append(variables)
    inputs.extend(profiles)
    for item in inputs:
        if isinstance(item, SessionVars):
            session_spec = item
        elif isinstance(item, DataVars):
            data_spec = item
        elif isinstance(item, TelemetryVars):
            telemetry_spec = item
        elif isinstance(item, Mapping):
            session_spec = SessionVars(**dict(item))

    if kwargs:
        merged = dict(session_spec.__dict__)
        merged.update(kwargs)
        session_spec = SessionVars(**merged)
    return session_spec, data_spec, telemetry_spec


def _run_teleop(
    arm_pairs: tuple[ArmPairConfig, ...],
    session_spec: SessionVars,
    data_spec: DataVars,
    telemetry_spec: TelemetryVars | None,
) -> RecordedEpisode | None:
    _apply_telemetry(telemetry_spec)
    cfg_kwargs: dict[str, Any] = {"arm_pairs": arm_pairs}
    if session_spec.rate_hz is not None:
        cfg_kwargs["rate_hz"] = session_spec.rate_hz
    if session_spec.output_dir is not None:
        cfg_kwargs["output_dir"] = Path(session_spec.output_dir)
    if session_spec.cameras is not None:
        cfg_kwargs["cameras"] = session_spec.cameras
    if session_spec.transport is not None:
        cfg_kwargs["transport"] = session_spec.transport
    if session_spec.jit is not None:
        cfg_kwargs["jit"] = session_spec.jit

    sess = BimanualSo101Session(config=TeleopSessionConfig(**cfg_kwargs))
    if data_spec.enabled:
        duration = (
            data_spec.duration_s if data_spec.duration_s is not None else session_spec.duration_s
        )
        if duration is not None:
            return sess.record_episode(
                duration_s=float(duration),
                label=data_spec.label,
                metadata=data_spec.metadata,
            )
        sess.start()
        episode = sess.start_recording(label=data_spec.label, metadata=data_spec.metadata)
        print(f"Recording episode {episode}. Press Ctrl+C to stop.")
        try:
            while True:
                sess.check_health()
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            recorded = sess.stop_recording()
            sess.stop()
        return recorded

    if session_spec.duration_s is not None:
        sess.record_episode(duration_s=float(session_spec.duration_s), label="session-run")
        return None
    sess.run()
    return None


def _run_sim_session(
    spec: SimSpec,
    session_spec: SessionVars,
) -> dict[str, float | int]:
    from ..session import Session as RuntimeSession
    from ..sim.base import SimRobot

    robot = SimRobot(
        spec.config,
        num_envs=spec.num_envs,
        backend=spec.backend,
        device=spec.device or "cpu",
    )

    policy = spec.policy
    if policy is None:
        import torch

        def _zero_policy(_obs: dict[str, Any]) -> torch.Tensor:
            return torch.zeros((robot.num_envs, robot.max_action_dim), device=robot.device)

        policy = _zero_policy

    runtime = RuntimeSession(
        robot=robot,
        policy=policy,
        rate_hz=float(session_spec.rate_hz or 50.0),
        warmup_s=float(session_spec.warmup_s),
    )
    runtime.run(duration=session_spec.duration_s)
    return runtime.stats.to_dict()


def run(
    arm: ArmPairConfig | Sequence[ArmPairConfig],
    *,
    logging: bool = False,
    rate_hz: float | None = None,
    duration_s: float | None = None,
    cameras: Sequence[CameraStreamConfig] | None = None,
    transport: Any | None = None,
    data_output: str | Path | None = None,
    lineage: str | None = None,
    scale: str | None = None,
    format: str = "native",
    metadata: Mapping[str, Any] | None = None,
    otel: bool = False,
    otel_exporter: str = "console",
    otel_sample_every: int = 100,
    otlp_endpoint: str | None = None,
) -> RecordedEpisode | dict[str, Any] | None:
    """Run a teleop session.

    Only ``arm`` is required. Everything else controls run behavior:
    - logging: live per-loop movement trace in terminal
    - data_output: enable recording to this directory
    - lineage/scale/format: run metadata + export format
    """
    if isinstance(arm, ArmPairConfig):
        arm_pairs = (arm,)
    else:
        arm_pairs = tuple(arm)
        if not arm_pairs:
            raise ValueError("arm list cannot be empty")

    run_metadata: dict[str, Any] = dict(metadata or {})
    if lineage is not None:
        run_metadata["lineage"] = lineage
    if scale is not None:
        run_metadata["scale"] = scale
    run_metadata["format"] = format

    session_spec = SessionVars(
        rate_hz=rate_hz,
        duration_s=duration_s,
        output_dir=data_output,
        # Beginner default: teleop without camera workers unless explicitly requested.
        cameras=tuple(cameras) if cameras is not None else (),
        transport=transport,
    )
    data_enabled = data_output is not None
    data_spec = DataVars(
        enabled=data_enabled,
        duration_s=duration_s,
        label="session-run",
        metadata=run_metadata if data_enabled else None,
    )
    telemetry_spec = TelemetryVars(
        otel=otel,
        exporter=otel_exporter,
        sample_every=otel_sample_every,
        live_trace_every=1 if logging else 0,
        otlp_endpoint=otlp_endpoint,
    )

    episode = _run_teleop(arm_pairs, session_spec, data_spec, telemetry_spec)
    if episode is None:
        return None

    export_format = format.strip().lower()
    if export_format in {"native", "raw"}:
        return episode
    if export_format == "mcap":
        return LeRobotRecorder(data_output or episode.episode_dir.parent).export_episode_to_mcap(
            episode,
            output_dir=data_output or Path("mcap_exports"),
        )
    if export_format == "lerobot":
        repo_id = lineage or "local/rfx-session"
        return LeRobotRecorder(data_output or episode.episode_dir.parent).export_episode_to_lerobot(
            episode,
            repo_id=repo_id,
            root=data_output or Path("lerobot_datasets"),
            push_to_hub=False,
        )
    raise ValueError("format must be one of: native, mcap, lerobot")
