"""
rfx.runtime.health - Watchdog, reconnect policy, and health monitoring.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Watchdog:
    """Daemon-thread watchdog timer. Calls on_timeout if not kicked within timeout_s."""

    def __init__(self, timeout_s: float, on_timeout: Callable[[], None]) -> None:
        self._timeout_s = timeout_s
        self._on_timeout = on_timeout
        self._last_kick = time.monotonic()
        self._running = True
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def kick(self) -> None:
        with self._lock:
            self._last_kick = time.monotonic()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=self._timeout_s + 1.0)

    def _run(self) -> None:
        while self._running:
            time.sleep(min(self._timeout_s / 4.0, 0.5))
            with self._lock:
                elapsed = time.monotonic() - self._last_kick
            if elapsed > self._timeout_s:
                logger.warning("Watchdog timeout after %.1fs", elapsed)
                try:
                    self._on_timeout()
                except Exception:
                    logger.exception("Error in watchdog timeout callback")
                # Reset after firing
                with self._lock:
                    self._last_kick = time.monotonic()


@dataclass
class ReconnectPolicy:
    """Exponential backoff policy for reconnection attempts."""

    max_retries: int = 5
    base_delay_s: float = 1.0
    backoff_factor: float = 2.0
    max_delay_s: float = 30.0

    def delays(self):
        """Yield delay values for each retry attempt."""
        delay = self.base_delay_s
        for _ in range(self.max_retries):
            yield delay
            delay = min(delay * self.backoff_factor, self.max_delay_s)

    def execute(self, action: Callable[[], bool]) -> bool:
        """
        Try action() with exponential backoff until it returns True or retries exhausted.
        Returns True if action succeeded, False if all retries failed.
        """
        for i, delay in enumerate(self.delays()):
            try:
                if action():
                    logger.info("Reconnect succeeded on attempt %d", i + 1)
                    return True
            except Exception:
                logger.exception("Reconnect attempt %d failed", i + 1)
            logger.info("Retrying in %.1fs...", delay)
            time.sleep(delay)
        logger.error("All %d reconnect attempts failed", self.max_retries)
        return False


class HealthMonitor:
    """Periodic health check publisher."""

    def __init__(
        self,
        transport: Any,
        node_name: str,
        check_interval_s: float = 5.0,
        health_check: Callable[[], dict] | None = None,
    ) -> None:
        self._transport = transport
        self._node_name = node_name
        self._interval = check_interval_s
        self._health_check = health_check or (lambda: {"status": "ok"})
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import json

        while self._running:
            try:
                health = self._health_check()
                health["node"] = self._node_name
                health["timestamp_ns"] = time.time_ns()
                self._transport.publish(
                    f"rfx/health/{self._node_name}",
                    json.dumps(health),
                )
            except Exception:
                logger.exception("Health check failed")
            time.sleep(self._interval)

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=self._interval + 1.0)
