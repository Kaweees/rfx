"""
rfx.robot.discovery - Port scanning and hardware discovery.

    >>> from rfx.robot.discovery import discover_ports, discover_so101_ports
    >>> ports = discover_so101_ports()
"""

from __future__ import annotations

from typing import Any


def discover_ports() -> list[dict[str, Any]]:
    """Scan serial ports for known robot hardware.

    Returns a list of dicts with keys: port, description, robot_type, hwid.
    """
    from ..node import discover_ports as _discover_ports

    return _discover_ports()


def discover_so101_ports() -> list[str]:
    """Return serial port paths that look like SO-101 arms."""
    from ..node import discover_so101_ports as _discover_so101_ports

    return _discover_so101_ports()


__all__ = ["discover_ports", "discover_so101_ports"]
