"""Tests for rfx.runtime.lifecycle - Managed node lifecycle state machine."""

from __future__ import annotations

from rfx.runtime.lifecycle import (
    LifecycleState,
    LifecycleTransition,
    ManagedNode,
)
from rfx.runtime.node import NodeContext
from rfx.teleop.transport import InprocTransport


def _make_ctx(name: str = "test_node") -> NodeContext:
    return NodeContext(name=name, package="test_pkg", transport=InprocTransport())


# ---------------------------------------------------------------------------
# Valid transitions
# ---------------------------------------------------------------------------


def test_valid_transition_configure() -> None:
    """UNCONFIGURED -> INACTIVE via CONFIGURE."""
    node = ManagedNode(_make_ctx())
    assert node.lifecycle_state == LifecycleState.UNCONFIGURED
    assert node.trigger_transition(LifecycleTransition.CONFIGURE) is True
    assert node.lifecycle_state == LifecycleState.INACTIVE


def test_valid_full_lifecycle() -> None:
    """Walk through the full happy path: UNCONFIGURED -> INACTIVE -> ACTIVE -> INACTIVE -> FINALIZED."""
    node = ManagedNode(_make_ctx())

    assert node.trigger_transition(LifecycleTransition.CONFIGURE) is True
    assert node.lifecycle_state == LifecycleState.INACTIVE

    assert node.trigger_transition(LifecycleTransition.ACTIVATE) is True
    assert node.lifecycle_state == LifecycleState.ACTIVE

    assert node.trigger_transition(LifecycleTransition.DEACTIVATE) is True
    assert node.lifecycle_state == LifecycleState.INACTIVE

    assert node.trigger_transition(LifecycleTransition.SHUTDOWN) is True
    assert node.lifecycle_state == LifecycleState.FINALIZED


def test_transition_history_recorded() -> None:
    """transition_history tracks every successful transition."""
    node = ManagedNode(_make_ctx())
    node.trigger_transition(LifecycleTransition.CONFIGURE)
    node.trigger_transition(LifecycleTransition.ACTIVATE)

    history = node.transition_history
    assert len(history) == 2
    assert history[0] == (LifecycleTransition.CONFIGURE, LifecycleState.INACTIVE)
    assert history[1] == (LifecycleTransition.ACTIVATE, LifecycleState.ACTIVE)


# ---------------------------------------------------------------------------
# Invalid transitions
# ---------------------------------------------------------------------------


def test_invalid_transition_returns_false() -> None:
    """Attempting ACTIVATE from UNCONFIGURED (invalid) returns False and keeps state."""
    node = ManagedNode(_make_ctx())
    assert node.lifecycle_state == LifecycleState.UNCONFIGURED
    assert node.trigger_transition(LifecycleTransition.ACTIVATE) is False
    assert node.lifecycle_state == LifecycleState.UNCONFIGURED


def test_double_configure_invalid() -> None:
    """Cannot configure twice in a row (INACTIVE + CONFIGURE is invalid)."""
    node = ManagedNode(_make_ctx())
    node.trigger_transition(LifecycleTransition.CONFIGURE)
    assert node.trigger_transition(LifecycleTransition.CONFIGURE) is False


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_callback_exception_enters_error_processing() -> None:
    """If on_configure raises, node transitions to ERROR_PROCESSING."""

    class BadNode(ManagedNode):
        def on_configure(self) -> bool:
            raise RuntimeError("setup boom")

    node = BadNode(_make_ctx())
    result = node.trigger_transition(LifecycleTransition.CONFIGURE)
    assert result is False
    assert node.lifecycle_state == LifecycleState.ERROR_PROCESSING


def test_callback_returning_false_enters_error_processing() -> None:
    """If a callback returns False, node enters ERROR_PROCESSING."""

    class RejectNode(ManagedNode):
        def on_activate(self) -> bool:
            return False

    node = RejectNode(_make_ctx())
    node.trigger_transition(LifecycleTransition.CONFIGURE)
    result = node.trigger_transition(LifecycleTransition.ACTIVATE)
    assert result is False
    assert node.lifecycle_state == LifecycleState.ERROR_PROCESSING


# ---------------------------------------------------------------------------
# Recovery from ERROR_PROCESSING
# ---------------------------------------------------------------------------


def test_recover_from_error_processing() -> None:
    """ERROR_PROCESSING -> UNCONFIGURED via RECOVER."""

    class BadNode(ManagedNode):
        def on_configure(self) -> bool:
            raise RuntimeError("boom")

    node = BadNode(_make_ctx())
    node.trigger_transition(LifecycleTransition.CONFIGURE)
    assert node.lifecycle_state == LifecycleState.ERROR_PROCESSING

    assert node.trigger_transition(LifecycleTransition.RECOVER) is True
    assert node.lifecycle_state == LifecycleState.UNCONFIGURED


def test_shutdown_from_error_processing() -> None:
    """ERROR_PROCESSING -> FINALIZED via SHUTDOWN."""

    class BadNode(ManagedNode):
        def on_configure(self) -> bool:
            raise RuntimeError("boom")

    node = BadNode(_make_ctx())
    node.trigger_transition(LifecycleTransition.CONFIGURE)
    assert node.lifecycle_state == LifecycleState.ERROR_PROCESSING

    assert node.trigger_transition(LifecycleTransition.SHUTDOWN) is True
    assert node.lifecycle_state == LifecycleState.FINALIZED


# ---------------------------------------------------------------------------
# Auto-lifecycle run()
# ---------------------------------------------------------------------------


def test_run_with_max_steps() -> None:
    """run() auto-transitions through configure/activate, ticks, then deactivates/shuts down."""
    ticks: list[int] = []

    class CountNode(ManagedNode):
        def tick(self) -> bool:
            ticks.append(len(ticks))
            return True

    node = CountNode(_make_ctx())
    steps = node.run(rate_hz=1000, max_steps=5)

    assert steps == 5
    assert len(ticks) == 5
    assert node.lifecycle_state == LifecycleState.FINALIZED


def test_run_tick_returning_false_stops_early() -> None:
    """If tick() returns False the run loop stops and node finalizes."""

    class OneShotNode(ManagedNode):
        def tick(self) -> bool:
            return False

    node = OneShotNode(_make_ctx())
    steps = node.run(rate_hz=1000, max_steps=100)

    assert steps == 1
    assert node.lifecycle_state == LifecycleState.FINALIZED
