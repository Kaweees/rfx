"""
rfx.teleop - Python-first teleoperation API.
"""

from .benchmark import JitterBenchmarkResult, assert_jitter_budget, run_jitter_benchmark
from .config import (
    ArmPairConfig,
    CameraStreamConfig,
    JitPolicyConfig,
    TeleopSessionConfig,
    TransportConfig,
    ZenohConfig,
)
from .lerobot_writer import LeRobotExportConfig, LeRobotPackageWriter
from .recorder import LeRobotRecorder, RecordedEpisode
from .session import BimanualSo101Session, LoopTimingStats
from .transport import (
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
    from .g1 import G1SafetyConfig, G1SafetyLayer, G1TeleopConfig, G1TeleopSession
    from .g1_obs import G1ObsConfig, G1ObservationBuilder
    from .retarget import G1Retargeter, RetargetBase, RetargetConfig, SimpleRetargeter
    from .vr import VRConfig, VRMotionPublisher
except ImportError:
    pass

__all__ = [
    "ArmPairConfig",
    "BimanualSo101Session",
    "CameraStreamConfig",
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
    "create_transport",
    "rust_transport_available",
    "zenoh_transport_available",
    "assert_jitter_budget",
    "run_jitter_benchmark",
]
