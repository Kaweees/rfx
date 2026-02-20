"""Tests for rfx.drivers - Robot driver plugin registry."""

from __future__ import annotations

from typing import Any

import pytest

import rfx.drivers as drivers_mod
from rfx.drivers import get_driver, list_drivers, register_driver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDriver:
    """Minimal driver-like class for registry testing."""

    _name = "fake"
    _num_joints = 6

    def connect(self, **kwargs: Any) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return False

    def read_state(self) -> dict[str, Any]:
        return {}

    def send_command(self, command: dict[str, Any]) -> None:
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_joints(self) -> int:
        return self._num_joints


@pytest.fixture(autouse=True)
def _clean_registry():
    """Remove test drivers from the global registry before/after each test."""
    for key in ("test_fake", "test_alpha", "test_beta"):
        drivers_mod._DRIVER_REGISTRY.pop(key, None)
    yield
    for key in ("test_fake", "test_alpha", "test_beta"):
        drivers_mod._DRIVER_REGISTRY.pop(key, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_register_and_get_driver() -> None:
    """register_driver + get_driver round-trip."""
    register_driver("test_fake", _FakeDriver)
    result = get_driver("test_fake")
    assert result is _FakeDriver


def test_get_driver_returns_none_for_unknown() -> None:
    assert get_driver("unknown_xyz_12345") is None


def test_list_drivers_returns_registered_names() -> None:
    register_driver("test_alpha", _FakeDriver)
    register_driver("test_beta", _FakeDriver)

    names = list_drivers()
    assert "test_alpha" in names
    assert "test_beta" in names


def test_register_driver_overwrites() -> None:
    """Re-registering the same name replaces the factory."""

    class AnotherDriver(_FakeDriver):
        pass

    register_driver("test_fake", _FakeDriver)
    register_driver("test_fake", AnotherDriver)
    assert get_driver("test_fake") is AnotherDriver
