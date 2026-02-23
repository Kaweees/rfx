"""Tests for discovery patterns using InprocTransport.

Discovery is in the Rust layer (rfx_core::comm::discovery), so these tests
validate the Python-side pattern of declaring nodes and tracking topics
via the InprocTransport pub/sub mechanism.
"""

from __future__ import annotations

import json

from rfx.teleop.transport import InprocTransport

# ---------------------------------------------------------------------------
# Simple node registry pattern
# ---------------------------------------------------------------------------


def test_node_declaration_visible_via_transport() -> None:
    """A node announces itself on rfx/discovery/nodes and a subscriber picks it up."""
    transport = InprocTransport()
    sub = transport.subscribe("rfx/discovery/nodes")

    # Simulate a node declaring itself
    node_info = {"name": "camera_node", "package": "perception", "pid": 12345}
    transport.publish("rfx/discovery/nodes", json.dumps(node_info))

    env = sub.recv(timeout_s=0.5)
    assert env is not None
    data = json.loads(env.payload)
    assert data["name"] == "camera_node"
    assert data["package"] == "perception"


def test_multiple_nodes_announced() -> None:
    """Multiple node announcements are received in order."""
    transport = InprocTransport()
    sub = transport.subscribe("rfx/discovery/nodes")

    for i in range(3):
        transport.publish(
            "rfx/discovery/nodes",
            json.dumps({"name": f"node_{i}", "package": "pkg"}),
        )

    names = []
    for _ in range(3):
        env = sub.recv(timeout_s=0.5)
        assert env is not None
        names.append(json.loads(env.payload)["name"])

    assert names == ["node_0", "node_1", "node_2"]


# ---------------------------------------------------------------------------
# Topic info tracking
# ---------------------------------------------------------------------------


def test_topic_info_tracking() -> None:
    """Topic advertisements are published and tracked on rfx/discovery/topics."""
    transport = InprocTransport()
    sub = transport.subscribe("rfx/discovery/topics")

    topic_info = {
        "topic": "rfx/sensor/lidar",
        "message_type": "PointCloud2",
        "publisher": "lidar_node",
    }
    transport.publish("rfx/discovery/topics", json.dumps(topic_info))

    env = sub.recv(timeout_s=0.5)
    assert env is not None
    data = json.loads(env.payload)
    assert data["topic"] == "rfx/sensor/lidar"
    assert data["message_type"] == "PointCloud2"


def test_topic_wildcard_discovery() -> None:
    """Subscribing with a wildcard catches all discovery sub-topics."""
    transport = InprocTransport()
    sub = transport.subscribe("rfx/discovery/*")

    transport.publish("rfx/discovery/nodes", json.dumps({"name": "n1"}))
    transport.publish("rfx/discovery/topics", json.dumps({"topic": "/cam"}))

    msgs = []
    for _ in range(2):
        env = sub.recv(timeout_s=0.5)
        assert env is not None
        msgs.append(env.key)

    assert "rfx/discovery/nodes" in msgs
    assert "rfx/discovery/topics" in msgs
