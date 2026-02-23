"""
rfx.teleop - Python-first teleoperation API.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Submodule access
from . import session  # noqa: F811,E402 — public re-export
from .benchmark import JitterBenchmarkResult, assert_jitter_budget, run_jitter_benchmark
from .config import (
    ArmPairConfig,
    CameraStreamConfig,
    HybridConfig,
    JitPolicyConfig,
    TeleopSessionConfig,
    TransportConfig,
    ZenohConfig,
)
from .lerobot_writer import LeRobotExportConfig, LeRobotPackageWriter
from .recorder import LeRobotRecorder, RecordedEpisode
from .session import (
    BimanualSo101Session,
    LoopTimingStats,
    run,  # noqa: F401 — public re-export
)
from .transport import (
    HybridTransport,
    InprocTransport,
    RustSubscription,
    RustTransport,
    Subscription,
    TransportEnvelope,
    ZenohTransport,
    create_transport,
    rust_transport_available,
    zenoh_transport_available,
)

# G1 humanoid VR teleop (optional — guarded because torch may not be installed)
try:
    from .g1 import G1SafetyConfig, G1SafetyLayer, G1TeleopConfig, G1TeleopSession  # noqa: F401
    from .g1_obs import G1ObsConfig, G1ObservationBuilder  # noqa: F401
    from .retarget import G1Retargeter, RetargetBase, RetargetConfig, SimpleRetargeter  # noqa: F401
    from .vr import VRConfig, VRMotionPublisher  # noqa: F401
except ImportError:
    pass

__all__ = [
    "ArmPairConfig",
    "BimanualSo101Session",
    "CameraStreamConfig",
    "HybridConfig",
    "HybridTransport",
    "InprocTransport",
    "JitPolicyConfig",
    "JitterBenchmarkResult",
    "LeRobotExportConfig",
    "LeRobotPackageWriter",
    "LeRobotRecorder",
    "LoopTimingStats",
    "RecordedEpisode",
    "RustSubscription",
    "RustTransport",
    "Subscription",
    "TeleopSessionConfig",
    "TransportConfig",
    "TransportEnvelope",
    "ZenohConfig",
    "ZenohTransport",
    "assert_jitter_budget",
    "create_transport",
    "run",
    "run_jitter_benchmark",
    "rust_transport_available",
    "session",
    "so101",
    "zenoh_transport_available",
]


def so101(config: str | Mapping[str, Any] | None = None, **kwargs: Any) -> ArmPairConfig:
    """Beginner-friendly SO-101 arm spec factory.

    Examples:
        arm = so101()
        arm = so101(leader_port="/dev/cu.usbmodemA", follower_port="/dev/cu.usbmodemB")
        arm = so101({"name": "main", "leader_port": "...", "follower_port": "..."})
    """
    data: dict[str, Any] = {}
    if isinstance(config, Mapping):
        data.update(config)
    data.update(kwargs)

    name = str(data.get("name", "main"))
    leader_port_raw = data.get("leader_port")
    follower_port_raw = data.get("follower_port")

    if isinstance(config, str):
        # Accept a profile path marker for future extension without failing simple scripts.
        if Path(config).exists():
            pass

    if leader_port_raw is None or follower_port_raw is None:
        try:
            from ..real.so101 import _auto_pair

            auto_leader, auto_follower = _auto_pair()
            leader_port = str(leader_port_raw or auto_leader)
            follower_port = str(follower_port_raw or auto_follower)
        except Exception as exc:
            raise RuntimeError(
                "Could not auto-discover SO-101 ports. "
                "Pass both `leader_port` and `follower_port` explicitly."
            ) from exc
    else:
        leader_port = str(leader_port_raw)
        follower_port = str(follower_port_raw)

    return ArmPairConfig(name=name, leader_port=leader_port, follower_port=follower_port)
