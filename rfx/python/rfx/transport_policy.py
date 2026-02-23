"""
rfx.transport_policy - Plane routing policy for local data and distributed control.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

DEFAULT_LOCAL_ONLY_PATTERNS = ("data/**",)
DEFAULT_ZENOH_REQUIRED_PATTERNS = (
    "control/**",
    "rfx/**",
    # Backward-compatible teleop keys.
    "teleop/control/**",
    "teleop/service/**",
)


def _pattern_prefix(pattern: str) -> str:
    for idx, ch in enumerate(pattern):
        if ch in ("*", "?", "["):
            return pattern[:idx]
    return pattern


def _patterns_overlap(left: str, right: str) -> bool:
    if fnmatchcase(left, right) or fnmatchcase(right, left):
        return True
    left_prefix = _pattern_prefix(left)
    right_prefix = _pattern_prefix(right)
    return left_prefix.startswith(right_prefix) or right_prefix.startswith(left_prefix)


@dataclass(frozen=True)
class KeySpacePolicy:
    """Routing policy for key-spaces across local and Zenoh planes."""

    local_only_patterns: tuple[str, ...] = DEFAULT_LOCAL_ONLY_PATTERNS
    zenoh_required_patterns: tuple[str, ...] = DEFAULT_ZENOH_REQUIRED_PATTERNS

    def validate_key(self, key: str) -> None:
        if not key or key.startswith("/") or "//" in key:
            raise ValueError(
                f"Invalid transport key {key!r}. Keys must be non-empty and slash-delimited."
            )

    def is_local_only(self, key: str) -> bool:
        return any(fnmatchcase(key, pattern) for pattern in self.local_only_patterns)

    def requires_zenoh(self, key: str) -> bool:
        return any(fnmatchcase(key, pattern) for pattern in self.zenoh_required_patterns)

    def should_mirror_to_zenoh(self, key: str) -> bool:
        self.validate_key(key)
        if self.is_local_only(key):
            return False
        return self.requires_zenoh(key)

    def pattern_routes_to_zenoh(self, pattern: str) -> bool:
        self.validate_key(pattern)
        if any(_patterns_overlap(pattern, local) for local in self.local_only_patterns):
            # Local-only wins.
            if not any(
                _patterns_overlap(pattern, remote) for remote in self.zenoh_required_patterns
            ):
                return False
        return any(_patterns_overlap(pattern, remote) for remote in self.zenoh_required_patterns)


def policy_from_hybrid_config(config: Any) -> KeySpacePolicy:
    """Build a key-space policy from teleop HybridConfig-like objects."""
    local_patterns = tuple(getattr(config, "local_only_patterns", DEFAULT_LOCAL_ONLY_PATTERNS))
    required_patterns = tuple(getattr(config, "zenoh_required_patterns", ()))
    # Backward-compatible field in existing configs.
    control_patterns = tuple(getattr(config, "control_patterns", ()))
    if control_patterns:
        required_patterns = tuple(
            dict.fromkeys((*required_patterns, *DEFAULT_ZENOH_REQUIRED_PATTERNS, *control_patterns))
        )
    if not required_patterns:
        required_patterns = DEFAULT_ZENOH_REQUIRED_PATTERNS
    return KeySpacePolicy(
        local_only_patterns=local_patterns,
        zenoh_required_patterns=required_patterns,
    )
