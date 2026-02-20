"""
rfx.drivers - Robot driver plugin registry.

Register and discover robot drivers via entry_points or explicit registration.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Global driver registry
_DRIVER_REGISTRY: dict[str, type] = {}


@runtime_checkable
class RobotDriver(Protocol):
    """Protocol that all robot drivers must implement."""

    def connect(self, **kwargs: Any) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def read_state(self) -> dict[str, Any]: ...
    def send_command(self, command: dict[str, Any]) -> None: ...

    @property
    def name(self) -> str: ...

    @property
    def num_joints(self) -> int: ...


def register_driver(name: str, factory: type) -> None:
    """Register a driver factory by name."""
    _DRIVER_REGISTRY[name] = factory
    logger.debug("Registered driver: %s", name)


def get_driver(name: str) -> type | None:
    """Get a registered driver factory by name."""
    return _DRIVER_REGISTRY.get(name)


def list_drivers() -> list[str]:
    """List all registered driver names."""
    return sorted(_DRIVER_REGISTRY.keys())


def _discover_entry_points() -> None:
    """Auto-discover drivers from entry_points(group='rfx.drivers')."""
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        rfx_eps = eps.select(group="rfx.drivers") if hasattr(eps, "select") else eps.get("rfx.drivers", [])
        for ep in rfx_eps:
            try:
                driver_cls = ep.load()
                register_driver(ep.name, driver_cls)
            except Exception:
                logger.debug("Failed to load driver entry point: %s", ep.name)
    except Exception:
        pass


# Auto-discover on import
_discover_entry_points()
