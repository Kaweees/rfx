"""Tests for wire format contracts - envelope metadata, service schemas, parameter events."""

from __future__ import annotations

import json

from rfx.teleop.service import ServiceRequest, ServiceResponse
from rfx.teleop.transport import InprocTransport

# ---------------------------------------------------------------------------
# Envelope metadata key injection
# ---------------------------------------------------------------------------


def test_envelope_metadata_type_injection() -> None:
    """Publishing with metadata containing _type and _schema_version preserves them."""
    transport = InprocTransport()
    sub = transport.subscribe("rfx/sensor/imu")

    meta = {"_type": "sensor_msgs/Imu", "_schema_version": "1.0"}
    env = transport.publish("rfx/sensor/imu", b"data", metadata=meta)

    assert env.metadata["_type"] == "sensor_msgs/Imu"
    assert env.metadata["_schema_version"] == "1.0"

    received = sub.recv(timeout_s=0.5)
    assert received is not None
    assert received.metadata["_type"] == "sensor_msgs/Imu"
    assert received.metadata["_schema_version"] == "1.0"


def test_envelope_metadata_empty_by_default() -> None:
    """Without metadata, envelope metadata is an empty dict."""
    transport = InprocTransport()
    env = transport.publish("rfx/test", b"payload")
    assert env.metadata == {}


def test_envelope_sequence_increments() -> None:
    """Each publish increments the sequence number."""
    transport = InprocTransport()
    env1 = transport.publish("rfx/a", b"1")
    env2 = transport.publish("rfx/a", b"2")
    assert env2.sequence == env1.sequence + 1


def test_envelope_timestamp_ns_passthrough() -> None:
    """Caller-provided timestamp_ns is used verbatim."""
    transport = InprocTransport()
    env = transport.publish("rfx/test", b"data", timestamp_ns=42)
    assert env.timestamp_ns == 42


# ---------------------------------------------------------------------------
# Service request/response JSON schema
# ---------------------------------------------------------------------------


def test_service_request_json_schema() -> None:
    """ServiceRequest serialized as JSON has required fields."""
    req = ServiceRequest(request_id=1, timeout_ms=3000, payload={"cmd": "start"})
    req_dict = {
        "request_id": req.request_id,
        "timeout_ms": req.timeout_ms,
        "payload": req.payload,
    }
    data = json.loads(json.dumps(req_dict))

    assert "request_id" in data
    assert "timeout_ms" in data
    assert "payload" in data
    assert data["request_id"] == 1
    assert data["timeout_ms"] == 3000


def test_service_response_json_schema() -> None:
    """ServiceResponse serialized as JSON has required fields."""
    resp = ServiceResponse.ok(1, payload={"result": 42})
    resp_dict = {
        "request_id": resp.request_id,
        "status": resp.status,
        "error_code": resp.error_code,
        "error_message": resp.error_message,
        "payload": resp.payload,
    }
    data = json.loads(json.dumps(resp_dict))

    assert data["request_id"] == 1
    assert data["status"] == "ok"
    assert data["error_code"] == 0
    assert data["error_message"] == ""
    assert data["payload"] == {"result": 42}


def test_service_error_response_json_schema() -> None:
    """Error response has non-zero error_code and error status."""
    resp = ServiceResponse.error(2, code=500, message="internal")
    resp_dict = {
        "request_id": resp.request_id,
        "status": resp.status,
        "error_code": resp.error_code,
        "error_message": resp.error_message,
        "payload": resp.payload,
    }
    data = json.loads(json.dumps(resp_dict))

    assert data["status"] == "error"
    assert data["error_code"] == 500
    assert data["error_message"] == "internal"


# ---------------------------------------------------------------------------
# Parameter event schema
# ---------------------------------------------------------------------------


def test_parameter_event_schema() -> None:
    """Parameter events published to transport follow expected JSON shape."""
    transport = InprocTransport()
    sub = transport.subscribe("rfx/params/node1/kp")

    event = {
        "event": "set",
        "node": "node1",
        "name": "kp",
        "value": 2.5,
        "old_value": 1.0,
    }
    transport.publish("rfx/params/node1/kp", json.dumps(event))

    env = sub.recv(timeout_s=0.5)
    assert env is not None
    data = json.loads(env.payload)

    # Required fields
    assert "event" in data
    assert "node" in data
    assert "name" in data
    assert "value" in data
    assert data["event"] == "set"
    assert data["node"] == "node1"
    assert data["name"] == "kp"
    assert data["value"] == 2.5
    assert data["old_value"] == 1.0


def test_parameter_declare_event_schema() -> None:
    """Declare events have event=declare and contain value."""
    transport = InprocTransport()
    sub = transport.subscribe("rfx/params/*")

    event = {
        "event": "declare",
        "node": "ctrl",
        "name": "max_vel",
        "value": 1.5,
    }
    transport.publish("rfx/params/ctrl", json.dumps(event))

    env = sub.recv(timeout_s=0.5)
    assert env is not None
    data = json.loads(env.payload)
    assert data["event"] == "declare"
    assert data["value"] == 1.5
    assert "old_value" not in data
