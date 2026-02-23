"""
rfx.robot.graph - Zenoh transport graph and node discovery.

    >>> from rfx.robot.graph import discover_nodes
    >>> nodes = discover_nodes()
"""

from __future__ import annotations

# Re-export topic key helpers from node module
from ..node import TOPIC_PREFIX, cmd_key, state_key


def discover_nodes(timeout_s: float = 2.0) -> list[str]:
    """List live robot nodes on the Zenoh network.

    Uses liveliness queries to find nodes that have declared
    ``rfx/alive/node/{name}`` tokens.
    """
    from ..node import discover_nodes as _discover_nodes

    return _discover_nodes(timeout_s=timeout_s)


__all__ = ["TOPIC_PREFIX", "state_key", "cmd_key", "discover_nodes"]
