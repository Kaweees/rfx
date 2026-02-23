from __future__ import annotations

import pytest

from rfx.teleop import HybridConfig
from rfx.teleop.transport import HybridTransport, TransportEnvelope
from rfx.transport_policy import policy_from_hybrid_config


class _FakeTransport:
    def __init__(self) -> None:
        self.published: list[str] = []
        self.subscriber_count = 0

    def subscribe(self, pattern: str, _capacity: int):
        self.subscriber_count += 1
        return type("_Sub", (), {"pattern": pattern, "try_recv": lambda self: None})()

    def unsubscribe(self, _sub) -> bool:
        return True

    def publish(self, key: str, payload, *, metadata=None, timestamp_ns=None) -> TransportEnvelope:
        self.published.append(key)
        return TransportEnvelope(
            key=key,
            sequence=len(self.published),
            timestamp_ns=timestamp_ns or 0,
            payload=payload,
            metadata=metadata or {},
        )


def test_policy_routes_local_and_control_planes() -> None:
    policy = policy_from_hybrid_config(HybridConfig())
    assert policy.should_mirror_to_zenoh("data/camera/front") is False
    assert policy.should_mirror_to_zenoh("control/session/start") is True
    assert policy.should_mirror_to_zenoh("rfx/go2/state") is True


def test_hybrid_transport_uses_local_for_data_and_zenoh_for_control() -> None:
    local = _FakeTransport()
    zenoh = _FakeTransport()
    transport = HybridTransport(local=local, zenoh=zenoh, config=HybridConfig())

    transport.publish("data/camera/front", b"x")
    transport.publish("control/session/start", b"y")

    assert local.published == ["data/camera/front", "control/session/start"]
    assert zenoh.published == ["control/session/start"]


def test_hybrid_transport_fails_fast_when_zenoh_missing_for_control_key() -> None:
    local = _FakeTransport()
    transport = HybridTransport(local=local, zenoh=None, config=HybridConfig())

    with pytest.raises(RuntimeError, match="Zenoh control plane is unavailable"):
        transport.publish("control/session/start", b"x")
