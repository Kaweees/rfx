"""Tests for parameter system patterns using InprocTransport.

The core parameter server lives in Rust (rfx_core::comm::params).
These tests validate the Python-side parameter declare/get/set pattern
and parameter change notification via transport subscription.
"""

from __future__ import annotations

import json
import time

import pytest

from rfx.teleop.transport import InprocTransport


# ---------------------------------------------------------------------------
# Parameter server pattern (Python-level)
# ---------------------------------------------------------------------------


class SimpleParamStore:
    """Minimal Python parameter store mirroring the Rust param server protocol."""

    def __init__(self, transport: InprocTransport, namespace: str = "test_node") -> None:
        self._transport = transport
        self._ns = namespace
        self._params: dict[str, object] = {}

    def declare(self, name: str, default: object) -> None:
        self._params[name] = default
        self._publish_event("declare", name, default)

    def get(self, name: str) -> object:
        return self._params.get(name)

    def set(self, name: str, value: object) -> None:
        old = self._params.get(name)
        self._params[name] = value
        self._publish_event("set", name, value, old_value=old)

    def _publish_event(
        self, event: str, name: str, value: object, *, old_value: object = None
    ) -> None:
        payload = {
            "event": event,
            "node": self._ns,
            "name": name,
            "value": value,
        }
        if old_value is not None:
            payload["old_value"] = old_value
        self._transport.publish(
            f"rfx/params/{self._ns}/{name}",
            json.dumps(payload),
        )


def test_param_declare_and_get() -> None:
    transport = InprocTransport()
    store = SimpleParamStore(transport)

    store.declare("max_speed", 1.5)
    assert store.get("max_speed") == 1.5


def test_param_set_updates_value() -> None:
    transport = InprocTransport()
    store = SimpleParamStore(transport)

    store.declare("rate_hz", 50.0)
    store.set("rate_hz", 100.0)
    assert store.get("rate_hz") == 100.0


def test_param_get_unknown_returns_none() -> None:
    transport = InprocTransport()
    store = SimpleParamStore(transport)
    assert store.get("nonexistent") is None


# ---------------------------------------------------------------------------
# Change notifications via transport
# ---------------------------------------------------------------------------


def test_param_change_notification() -> None:
    """Setting a parameter publishes a change event that subscribers receive."""
    transport = InprocTransport()
    sub = transport.subscribe("rfx/params/test_node/*")
    store = SimpleParamStore(transport)

    store.declare("kp", 1.0)
    store.set("kp", 2.0)

    # First event: declare
    env1 = sub.recv(timeout_s=0.5)
    assert env1 is not None
    data1 = json.loads(env1.payload)
    assert data1["event"] == "declare"
    assert data1["name"] == "kp"
    assert data1["value"] == 1.0

    # Second event: set
    env2 = sub.recv(timeout_s=0.5)
    assert env2 is not None
    data2 = json.loads(env2.payload)
    assert data2["event"] == "set"
    assert data2["name"] == "kp"
    assert data2["value"] == 2.0
    assert data2["old_value"] == 1.0


def test_param_notification_key_is_namespaced() -> None:
    """Each parameter event is keyed under rfx/params/{node}/{param}."""
    transport = InprocTransport()
    sub = transport.subscribe("rfx/params/my_node/speed")
    store = SimpleParamStore(transport, namespace="my_node")

    store.declare("speed", 0.5)
    env = sub.recv(timeout_s=0.5)
    assert env is not None
    assert env.key == "rfx/params/my_node/speed"
