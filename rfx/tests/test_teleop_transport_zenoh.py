"""Tests for Zenoh transport integration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import rfx.teleop.transport as transport_mod
from rfx.teleop import TransportConfig, ZenohConfig


def test_zenoh_available_reports_bool() -> None:
    """zenoh_transport_available() returns a bool regardless of compilation."""
    result = transport_mod.zenoh_transport_available()
    assert isinstance(result, bool)


def test_zenoh_raises_when_not_compiled(monkeypatch) -> None:
    """ZenohTransport raises RuntimeError when zenoh feature is not compiled."""

    class _FakeTransport:
        @staticmethod
        def zenoh_available():
            return False

    monkeypatch.setattr(transport_mod, "_RustTransport", _FakeTransport)
    with pytest.raises(RuntimeError, match="not compiled"):
        transport_mod.ZenohTransport()


def test_zenoh_raises_when_no_rust(monkeypatch) -> None:
    """ZenohTransport raises RuntimeError when Rust bindings are missing."""
    monkeypatch.setattr(transport_mod, "_RustTransport", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        transport_mod.ZenohTransport()


def test_create_transport_zenoh_factory(monkeypatch) -> None:
    """Factory correctly wires zenoh backend through to ZenohTransport."""

    class _FakeEnvelope:
        def __init__(self):
            self.key = "test/key"
            self.sequence = 1
            self.timestamp_ns = 123
            self.payload = b"data"
            self.metadata_json = None

    class _FakeSubscription:
        id = 1
        pattern = "test/**"

        def recv(self):
            return _FakeEnvelope()

        def recv_timeout(self, _timeout):
            return _FakeEnvelope()

        def try_recv(self):
            return None

        def __len__(self):
            return 0

        def is_empty(self):
            return True

    class _FakeNativeTransport:
        subscriber_count = 0

        @staticmethod
        def zenoh_available():
            return True

        @staticmethod
        def zenoh(connect, listen, shared_memory, key_prefix):
            t = _FakeNativeTransport()
            return t

        def subscribe(self, pattern, capacity):
            return _FakeSubscription()

        def unsubscribe(self, sub_id):
            return True

        def publish(self, key, payload, metadata_json):
            env = _FakeEnvelope()
            env.key = key
            env.payload = payload
            env.metadata_json = metadata_json
            return env

    monkeypatch.setattr(transport_mod, "_RustTransport", _FakeNativeTransport)
    config = TransportConfig(backend="zenoh", zenoh=ZenohConfig())
    transport = transport_mod.create_transport(config)
    assert isinstance(transport, transport_mod.ZenohTransport)


def test_create_transport_zenoh_with_custom_config(monkeypatch) -> None:
    """Factory passes ZenohConfig fields through to the native constructor."""
    captured = {}

    class _FakeNativeTransport:
        subscriber_count = 0

        @staticmethod
        def zenoh_available():
            return True

        @staticmethod
        def zenoh(connect, listen, shared_memory, key_prefix):
            captured["connect"] = connect
            captured["listen"] = listen
            captured["shared_memory"] = shared_memory
            captured["key_prefix"] = key_prefix
            return _FakeNativeTransport()

        def subscribe(self, pattern, capacity):
            return None

        def unsubscribe(self, sub_id):
            return True

        def publish(self, key, payload, metadata_json):
            return None

    monkeypatch.setattr(transport_mod, "_RustTransport", _FakeNativeTransport)
    config = TransportConfig(
        backend="zenoh",
        zenoh=ZenohConfig(
            connect=("tcp/192.168.1.1:7447",),
            listen=("tcp/0.0.0.0:7447",),
            shared_memory=False,
            key_prefix="rfx/robot1",
        ),
    )
    transport_mod.create_transport(config)
    assert captured["connect"] == ["tcp/192.168.1.1:7447"]
    assert captured["listen"] == ["tcp/0.0.0.0:7447"]
    assert captured["shared_memory"] is False
    assert captured["key_prefix"] == "rfx/robot1"


@pytest.mark.skipif(
    not transport_mod.zenoh_transport_available(),
    reason="Zenoh not compiled",
)
def test_zenoh_roundtrip_integration() -> None:
    """Full pub/sub roundtrip through the real Zenoh backend."""
    transport = transport_mod.ZenohTransport()
    sub = transport.subscribe("integration/test/**")
    env = transport.publish(
        "integration/test/hello",
        b"world",
        metadata={"robot": "so101"},
    )

    got = sub.recv(timeout_s=2.0)
    assert got is not None
    assert got.key == "integration/test/hello"
    assert bytes(got.payload) == b"world"
    assert got.metadata.get("robot") == "so101"
