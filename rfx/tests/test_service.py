"""Tests for rfx.teleop.service - Service client/server round-trip."""

from __future__ import annotations

import time

from rfx.teleop.service import (
    ServiceClient,
    ServiceRequest,
    ServiceResponse,
    ServiceServer,
)
from rfx.teleop.transport import InprocTransport

# ---------------------------------------------------------------------------
# ServiceResponse constructors
# ---------------------------------------------------------------------------


def test_service_response_ok() -> None:
    resp = ServiceResponse.ok(42, payload={"result": "done"})
    assert resp.request_id == 42
    assert resp.status == "ok"
    assert resp.error_code == 0
    assert resp.payload == {"result": "done"}


def test_service_response_error() -> None:
    resp = ServiceResponse.error(7, code=404, message="not found")
    assert resp.request_id == 7
    assert resp.status == "error"
    assert resp.error_code == 404
    assert resp.error_message == "not found"


def test_service_response_timeout() -> None:
    resp = ServiceResponse.timeout(99)
    assert resp.request_id == 99
    assert resp.status == "timeout"
    assert resp.error_code == -1
    assert "timed out" in resp.error_message


# ---------------------------------------------------------------------------
# Round-trip via InprocTransport
# ---------------------------------------------------------------------------


def test_service_round_trip() -> None:
    """ServiceClient.call() → ServiceServer handler → response arrives."""
    transport = InprocTransport()

    def echo_handler(req: ServiceRequest) -> ServiceResponse:
        return ServiceResponse.ok(req.request_id, payload={"echo": req.payload})

    server = ServiceServer(transport, "echo", echo_handler)
    try:
        # Give the server thread a moment to start
        time.sleep(0.05)

        client = ServiceClient(transport, "echo")
        resp = client.call(payload="hello", timeout_s=0.5)

        assert resp.status == "ok"
        assert resp.payload == {"echo": "hello"}
    finally:
        server.stop()


def test_service_round_trip_with_error_handler() -> None:
    """Server returns an error response."""
    transport = InprocTransport()

    def fail_handler(req: ServiceRequest) -> ServiceResponse:
        return ServiceResponse.error(req.request_id, code=500, message="internal")

    server = ServiceServer(transport, "fail", fail_handler)
    try:
        time.sleep(0.05)
        client = ServiceClient(transport, "fail")
        resp = client.call(timeout_s=0.5)

        assert resp.status == "error"
        assert resp.error_code == 500
        assert resp.error_message == "internal"
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Timeout when no server is running
# ---------------------------------------------------------------------------


def test_service_call_timeout_no_server() -> None:
    """When no server is listening, call() returns a timeout response."""
    transport = InprocTransport()
    client = ServiceClient(transport, "nonexistent")

    t0 = time.monotonic()
    resp = client.call(payload="ping", timeout_s=0.3)
    elapsed = time.monotonic() - t0

    assert resp.status == "timeout"
    assert resp.error_code == -1
    # Should have actually waited roughly the timeout
    assert elapsed >= 0.2
