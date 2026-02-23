"""Tests for rfx.tf - Transform system."""

from __future__ import annotations

import json

import pytest

from rfx.teleop.transport import InprocTransport
from rfx.tf import (
    TransformBroadcaster,
    TransformBuffer,
    TransformStamped,
)

# ---------------------------------------------------------------------------
# TransformStamped creation and serialization
# ---------------------------------------------------------------------------


def test_transform_stamped_creation() -> None:
    tf = TransformStamped(
        parent_frame="world",
        child_frame="base_link",
        timestamp_ns=1000,
        translation=(1.0, 2.0, 3.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )
    assert tf.parent_frame == "world"
    assert tf.child_frame == "base_link"
    assert tf.translation == (1.0, 2.0, 3.0)
    assert tf.rotation == (0.0, 0.0, 0.0, 1.0)


def test_transform_stamped_to_dict_from_dict_roundtrip() -> None:
    """to_dict -> from_dict preserves all fields."""
    original = TransformStamped(
        parent_frame="map",
        child_frame="odom",
        timestamp_ns=123456789,
        translation=(0.5, -1.0, 0.0),
        rotation=(0.0, 0.0, 0.707, 0.707),
    )
    d = original.to_dict()
    restored = TransformStamped.from_dict(d)

    assert restored.parent_frame == original.parent_frame
    assert restored.child_frame == original.child_frame
    assert restored.timestamp_ns == original.timestamp_ns
    assert restored.translation == pytest.approx(original.translation)
    assert restored.rotation == pytest.approx(original.rotation)


def test_transform_stamped_identity() -> None:
    tf = TransformStamped.identity("world", "base")
    assert tf.parent_frame == "world"
    assert tf.child_frame == "base"
    assert tf.translation == (0.0, 0.0, 0.0)
    assert tf.rotation == (0.0, 0.0, 0.0, 1.0)


def test_to_dict_includes_schema_version() -> None:
    tf = TransformStamped.identity("a", "b")
    d = tf.to_dict()
    assert d["_schema_version"] == "1.0"


# ---------------------------------------------------------------------------
# TransformBuffer set_transform and lookup
# ---------------------------------------------------------------------------


def test_buffer_set_and_lookup_direct() -> None:
    """A directly stored transform is returned by lookup."""
    buf = TransformBuffer()
    tf = TransformStamped(
        parent_frame="world",
        child_frame="base",
        timestamp_ns=1000,
        translation=(1.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )
    buf.set_transform(tf)

    # lookup(target, source) -> transform with parent_frame=target, child_frame=source
    # BFS composes source->target, so lookup("base", "world") traverses world->base
    result = buf.lookup("base", "world")
    assert result is not None
    assert result.parent_frame == "base"
    assert result.child_frame == "world"
    assert result.translation == pytest.approx((1.0, 0.0, 0.0))


def test_buffer_lookup_identity() -> None:
    """Looking up same frame as source returns identity."""
    buf = TransformBuffer()
    result = buf.lookup("world", "world")
    assert result is not None
    assert result.translation == (0.0, 0.0, 0.0)
    assert result.rotation == (0.0, 0.0, 0.0, 1.0)


def test_buffer_lookup_unknown_returns_none() -> None:
    """Lookup for non-existent frames returns None."""
    buf = TransformBuffer()
    assert buf.lookup("world", "nonexistent") is None


# ---------------------------------------------------------------------------
# TransformBuffer lookup_chain (parent -> child chain)
# ---------------------------------------------------------------------------


def test_buffer_lookup_chain() -> None:
    """BFS chain: world -> base -> sensor composes correctly."""
    buf = TransformBuffer()

    # world -> base: translate 1m in x
    buf.set_transform(
        TransformStamped(
            parent_frame="world",
            child_frame="base",
            timestamp_ns=1000,
            translation=(1.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )
    )
    # base -> sensor: translate 0.5m in y
    buf.set_transform(
        TransformStamped(
            parent_frame="base",
            child_frame="sensor",
            timestamp_ns=1000,
            translation=(0.0, 0.5, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )
    )

    # lookup(target, source): BFS from source (world) to target (sensor)
    # Traverses world->base->sensor, composing: (1.0, 0.0, 0.0) + (0.0, 0.5, 0.0)
    result = buf.lookup("sensor", "world")
    assert result is not None
    assert result.translation == pytest.approx((1.0, 0.5, 0.0), abs=1e-9)


def test_buffer_all_frames() -> None:
    buf = TransformBuffer()
    buf.set_transform(TransformStamped.identity("world", "base"))
    buf.set_transform(TransformStamped.identity("base", "sensor"))

    frames = buf.all_frames()
    assert "world" in frames
    assert "base" in frames
    assert "sensor" in frames


# ---------------------------------------------------------------------------
# TransformBroadcaster sends to transport
# ---------------------------------------------------------------------------


def test_broadcaster_publishes_to_transport() -> None:
    """TransformBroadcaster.send_transform publishes on rfx/tf/{parent}/{child}."""
    transport = InprocTransport()
    sub = transport.subscribe("rfx/tf/world/base")
    broadcaster = TransformBroadcaster(transport)

    tf = TransformStamped(
        parent_frame="world",
        child_frame="base",
        timestamp_ns=42000,
        translation=(1.0, 2.0, 3.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )
    broadcaster.send_transform(tf)

    env = sub.recv(timeout_s=0.5)
    assert env is not None
    data = json.loads(env.payload)
    assert data["parent_frame"] == "world"
    assert data["child_frame"] == "base"
    assert data["translation"] == [1.0, 2.0, 3.0]


def test_broadcaster_with_buffer_also_stores() -> None:
    """When given a buffer, broadcaster stores the transform locally too."""
    transport = InprocTransport()
    buf = TransformBuffer()
    broadcaster = TransformBroadcaster(transport, buffer=buf)

    tf = TransformStamped.identity("odom", "base_link")
    broadcaster.send_transform(tf)

    assert "odom" in buf.all_frames()
    assert "base_link" in buf.all_frames()
