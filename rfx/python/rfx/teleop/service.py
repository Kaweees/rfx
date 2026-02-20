"""
rfx.teleop.service - Python service client/server wrappers.

Provides a pure-Python service pattern on top of the transport layer.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .transport import InprocTransport, TransportEnvelope, TransportLike


@dataclass
class ServiceRequest:
    request_id: int
    timeout_ms: int = 5000
    payload: Any = None


@dataclass
class ServiceResponse:
    request_id: int
    status: str = "ok"
    error_code: int = 0
    error_message: str = ""
    payload: Any = None

    @staticmethod
    def ok(request_id: int, payload: Any = None) -> ServiceResponse:
        return ServiceResponse(request_id=request_id, payload=payload)

    @staticmethod
    def error(request_id: int, code: int, message: str) -> ServiceResponse:
        return ServiceResponse(
            request_id=request_id, status="error", error_code=code, error_message=message
        )

    @staticmethod
    def timeout(request_id: int) -> ServiceResponse:
        return ServiceResponse(
            request_id=request_id,
            status="timeout",
            error_code=-1,
            error_message="service call timed out",
        )


ServiceHandler = Callable[[ServiceRequest], ServiceResponse]


class ServiceServer:
    """Registers a service handler on a transport-based request/response channel."""

    def __init__(
        self,
        transport: TransportLike,
        name: str,
        handler: ServiceHandler,
    ) -> None:
        self.name = name
        self._handler = handler
        self._transport = transport
        self._sub = transport.subscribe(f"rfx/srv/{name}/request", capacity=64)
        self._running = True
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def _serve_loop(self) -> None:
        while self._running:
            env = self._sub.recv(timeout_s=0.5)
            if env is None:
                continue
            try:
                payload = env.payload
                if isinstance(payload, memoryview):
                    payload = bytes(payload)
                req_data = json.loads(payload)
                request = ServiceRequest(
                    request_id=req_data.get("request_id", 0),
                    timeout_ms=req_data.get("timeout_ms", 5000),
                    payload=req_data.get("payload"),
                )
                response = self._handler(request)
                resp_data = {
                    "request_id": response.request_id,
                    "status": response.status,
                    "error_code": response.error_code,
                    "error_message": response.error_message,
                    "payload": response.payload,
                }
                self._transport.publish(
                    f"rfx/srv/{self.name}/response",
                    json.dumps(resp_data),
                )
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)

    def __del__(self) -> None:
        self.stop()


class ServiceClient:
    """Calls a service via transport request/response channels."""

    def __init__(self, transport: TransportLike, name: str) -> None:
        self.name = name
        self._transport = transport
        self._sub = transport.subscribe(f"rfx/srv/{name}/response", capacity=64)
        self._next_id = 0
        self._lock = threading.Lock()

    def call(
        self,
        payload: Any = None,
        timeout_s: float = 5.0,
    ) -> ServiceResponse:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id

        req_data = {
            "request_id": request_id,
            "timeout_ms": int(timeout_s * 1000),
            "payload": payload,
        }
        self._transport.publish(
            f"rfx/srv/{self.name}/request",
            json.dumps(req_data),
        )

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            env = self._sub.recv(timeout_s=min(remaining, 0.5))
            if env is None:
                continue
            try:
                raw = env.payload
                if isinstance(raw, memoryview):
                    raw = bytes(raw)
                resp_data = json.loads(raw)
                if resp_data.get("request_id") == request_id:
                    return ServiceResponse(
                        request_id=request_id,
                        status=resp_data.get("status", "ok"),
                        error_code=resp_data.get("error_code", 0),
                        error_message=resp_data.get("error_message", ""),
                        payload=resp_data.get("payload"),
                    )
            except Exception:
                continue

        return ServiceResponse.timeout(request_id)
