"""Tests for in-process teleop transport semantics."""

from __future__ import annotations

import json

import pytest

import rfx.teleop.transport as transport_mod
from rfx.teleop import InprocTransport, TransportConfig


def test_publish_subscribe_with_key_patterns() -> None:
    transport = InprocTransport()
    sub_all = transport.subscribe("teleop/**")
    sub_left = transport.subscribe("teleop/left/*")

    env = transport.publish("teleop/left/state", {"q": [1, 2, 3]})
    got_all = sub_all.recv(timeout_s=0.1)
    got_left = sub_left.recv(timeout_s=0.1)

    assert got_all is not None
    assert got_left is not None
    assert got_all.sequence == env.sequence
    assert got_left.key == "teleop/left/state"


def test_zero_copy_bytes_payload_preserves_memoryview() -> None:
    transport = InprocTransport()
    sub = transport.subscribe("teleop/**")

    payload = bytearray(b"abc")
    env = transport.publish("teleop/frame/raw", payload)
    got = sub.recv(timeout_s=0.1)

    assert isinstance(env.payload, memoryview)
    assert got is not None
    assert isinstance(got.payload, memoryview)
    assert got.payload.obj is payload


def test_rust_transport_adapter_uses_native_shape(monkeypatch) -> None:
    class _FakeEnvelope:
        def __init__(self):
            self.key = "teleop/left/state"
            self.sequence = 1
            self.timestamp_ns = 123
            self.payload = b"abc"
            self.metadata_json = json.dumps({"x": 1})

    class _FakeSubscription:
        id = 7
        pattern = "teleop/**"

        def __init__(self):
            self._env = _FakeEnvelope()

        def recv(self):
            return self._env

        def recv_timeout(self, _timeout):
            return self._env

        def try_recv(self):
            return None

        def __len__(self):
            return 1

        def is_empty(self):
            return False

    class _FakeTransport:
        subscriber_count = 0

        def __init__(self):
            self.subscriber_count = 0

        def subscribe(self, _pattern, _capacity):
            self.subscriber_count += 1
            return _FakeSubscription()

        def unsubscribe(self, _sub_id):
            self.subscriber_count = max(0, self.subscriber_count - 1)
            return True

        def publish(self, key, payload, metadata_json):
            env = _FakeEnvelope()
            env.key = key
            env.payload = payload
            env.metadata_json = metadata_json
            return env

    monkeypatch.setattr(transport_mod, "_RustTransport", _FakeTransport)

    transport = transport_mod.RustTransport()
    sub = transport.subscribe("teleop/**")
    env = transport.publish("teleop/left/state", {"v": 1}, metadata={"robot": "so101"})
    got = sub.recv(timeout_s=0.1)

    assert env.key == "teleop/left/state"
    assert isinstance(env.payload, memoryview)
    assert got is not None
    assert got.metadata["x"] == 1


def test_create_transport_prefers_rust_backend_when_available(monkeypatch) -> None:
    class _FakeNativeTransport:
        def __init__(self):
            self.subscriber_count = 0

        def subscribe(self, _pattern, _capacity):
            return None

        def unsubscribe(self, _id):
            return True

        def publish(self, _key, _payload, _metadata_json):
            return None

    monkeypatch.setattr(transport_mod, "_RustTransport", _FakeNativeTransport)
    transport = transport_mod.create_transport(
        TransportConfig(backend="inproc", zero_copy_hot_path=True)
    )
    assert isinstance(transport, transport_mod.RustTransport)


def test_create_transport_falls_back_to_python_inproc(monkeypatch) -> None:
    monkeypatch.setattr(transport_mod, "_RustTransport", None)
    transport = transport_mod.create_transport(
        TransportConfig(backend="inproc", zero_copy_hot_path=True)
    )
    assert isinstance(transport, transport_mod.InprocTransport)


def test_create_transport_rejects_unknown_backends() -> None:
    with pytest.raises(ValueError, match="Unsupported transport backend"):
        transport_mod.create_transport(TransportConfig(backend="dds"))  # type: ignore[arg-type]


def test_create_transport_rejects_invalid_backend_string() -> None:
    with pytest.raises(ValueError, match="Unsupported transport backend"):
        transport_mod.create_transport(TransportConfig(backend="grpc"))  # type: ignore[arg-type]


def test_create_transport_zenoh_raises_without_bindings(monkeypatch) -> None:
    """Zenoh factory raises RuntimeError when Rust bindings are missing."""
    monkeypatch.setattr(transport_mod, "_RustTransport", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        transport_mod.create_transport(TransportConfig(backend="zenoh"))


def test_create_transport_hybrid_routes_control_to_zenoh(monkeypatch) -> None:
    class _LocalSub:
        pattern = "teleop/**"

        def try_recv(self):
            return None

    class _ZenohSub:
        pattern = "teleop/control/**"

        def try_recv(self):
            return None

    class _FakeLocalTransport:
        def __init__(self):
            self.subscriber_count = 0
            self.published: list[str] = []

        def subscribe(self, pattern, capacity):
            self.subscriber_count += 1
            return _LocalSub()

        def unsubscribe(self, _sub):
            return True

        def publish(self, key, payload, metadata=None, timestamp_ns=None):
            self.published.append(key)
            return transport_mod.TransportEnvelope(
                key=key,
                sequence=1,
                timestamp_ns=timestamp_ns or 0,
                payload=payload,
                metadata=metadata or {},
            )

    class _FakeZenohTransport:
        def __init__(self, **_kwargs):
            self.subscriber_count = 0
            self.published: list[str] = []

        def subscribe(self, pattern, capacity):
            self.subscriber_count += 1
            return _ZenohSub()

        def unsubscribe(self, _sub):
            return True

        def publish(self, key, payload, metadata=None, timestamp_ns=None):
            self.published.append(key)
            return transport_mod.TransportEnvelope(
                key=key,
                sequence=1,
                timestamp_ns=timestamp_ns or 0,
                payload=payload,
                metadata=metadata or {},
            )

    monkeypatch.setattr(transport_mod, "RustTransport", _FakeLocalTransport)
    monkeypatch.setattr(transport_mod, "ZenohTransport", _FakeZenohTransport)
    monkeypatch.setattr(transport_mod, "rust_transport_available", lambda: True)

    transport = transport_mod.create_transport(TransportConfig(backend="hybrid"))
    assert isinstance(transport, transport_mod.HybridTransport)

    local = transport._local
    zenoh = transport._zenoh

    transport.publish("teleop/left/state", b"x")
    assert local.published == ["teleop/left/state"]
    assert zenoh.published == []

    transport.publish("teleop/control/start", b"x")
    assert local.published[-1] == "teleop/control/start"
    assert zenoh.published == ["teleop/control/start"]

    sub = transport.subscribe("teleop/control/**")
    assert len(sub.subscriptions) == 2
