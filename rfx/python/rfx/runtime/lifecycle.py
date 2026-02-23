"""
rfx.runtime.lifecycle - ROS 2-style managed node lifecycle.

Lifecycle states: UNCONFIGURED → INACTIVE → ACTIVE → FINALIZED
Error in any callback → ERROR_PROCESSING → (recover to UNCONFIGURED or shutdown)
"""

from __future__ import annotations

import enum
import logging
import time

from .node import Node, NodeContext

logger = logging.getLogger(__name__)


class LifecycleState(enum.Enum):
    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FINALIZED = "finalized"
    ERROR_PROCESSING = "error_processing"


class LifecycleTransition(enum.Enum):
    CONFIGURE = "configure"
    ACTIVATE = "activate"
    DEACTIVATE = "deactivate"
    CLEANUP = "cleanup"
    SHUTDOWN = "shutdown"
    ERROR = "error"
    RECOVER = "recover"


# Valid transitions: (from_state, transition) → to_state
_TRANSITIONS: dict[tuple[LifecycleState, LifecycleTransition], LifecycleState] = {
    (LifecycleState.UNCONFIGURED, LifecycleTransition.CONFIGURE): LifecycleState.INACTIVE,
    (LifecycleState.INACTIVE, LifecycleTransition.ACTIVATE): LifecycleState.ACTIVE,
    (LifecycleState.ACTIVE, LifecycleTransition.DEACTIVATE): LifecycleState.INACTIVE,
    (LifecycleState.INACTIVE, LifecycleTransition.CLEANUP): LifecycleState.UNCONFIGURED,
    (LifecycleState.UNCONFIGURED, LifecycleTransition.SHUTDOWN): LifecycleState.FINALIZED,
    (LifecycleState.INACTIVE, LifecycleTransition.SHUTDOWN): LifecycleState.FINALIZED,
    (LifecycleState.ACTIVE, LifecycleTransition.SHUTDOWN): LifecycleState.FINALIZED,
    (LifecycleState.ERROR_PROCESSING, LifecycleTransition.RECOVER): LifecycleState.UNCONFIGURED,
    (LifecycleState.ERROR_PROCESSING, LifecycleTransition.SHUTDOWN): LifecycleState.FINALIZED,
}


class ManagedNode(Node):
    """
    A node with ROS 2-style lifecycle management.

    Subclass and override on_configure/on_activate/on_deactivate/on_shutdown/on_error.
    The ``run()`` method auto-transitions: unconfigured → inactive → active → tick loop → shutdown.
    """

    def __init__(self, context: NodeContext):
        super().__init__(context)
        self._lifecycle_state = LifecycleState.UNCONFIGURED
        self._transition_history: list[tuple[LifecycleTransition, LifecycleState]] = []

    @property
    def lifecycle_state(self) -> LifecycleState:
        return self._lifecycle_state

    @property
    def transition_history(self) -> list[tuple[LifecycleTransition, LifecycleState]]:
        return list(self._transition_history)

    def trigger_transition(self, transition: LifecycleTransition) -> bool:
        key = (self._lifecycle_state, transition)
        target = _TRANSITIONS.get(key)
        if target is None:
            logger.warning(
                "Invalid transition %s from state %s",
                transition.value,
                self._lifecycle_state.value,
            )
            return False

        callback_map = {
            LifecycleTransition.CONFIGURE: self.on_configure,
            LifecycleTransition.ACTIVATE: self.on_activate,
            LifecycleTransition.DEACTIVATE: self.on_deactivate,
            LifecycleTransition.CLEANUP: self.on_cleanup,
            LifecycleTransition.SHUTDOWN: self.on_shutdown,
            LifecycleTransition.RECOVER: self.on_recover,
        }

        callback = callback_map.get(transition)
        try:
            if callback is not None:
                success = callback()
                if not success:
                    logger.warning("Transition %s callback returned False", transition.value)
                    self._enter_error()
                    return False
        except Exception:
            logger.exception("Error during transition %s", transition.value)
            self._enter_error()
            return False

        self._lifecycle_state = target
        self._transition_history.append((transition, target))
        logger.debug("Transitioned to %s via %s", target.value, transition.value)
        return True

    def _enter_error(self) -> None:
        self._lifecycle_state = LifecycleState.ERROR_PROCESSING
        try:
            self.on_error()
        except Exception:
            logger.exception("Error in on_error callback")

    # --- Override these in subclasses ---

    def on_configure(self) -> bool:
        """Called during UNCONFIGURED → INACTIVE. Bridge to setup()."""
        self.setup()
        return True

    def on_activate(self) -> bool:
        """Called during INACTIVE → ACTIVE."""
        return True

    def on_deactivate(self) -> bool:
        """Called during ACTIVE → INACTIVE."""
        return True

    def on_cleanup(self) -> bool:
        """Called during INACTIVE → UNCONFIGURED."""
        return True

    def on_shutdown(self) -> bool:
        """Called during any → FINALIZED. Bridge to shutdown()."""
        self.shutdown()
        return True

    def on_error(self) -> bool:
        """Called when entering ERROR_PROCESSING."""
        return True

    def on_recover(self) -> bool:
        """Called during ERROR_PROCESSING → UNCONFIGURED."""
        return True

    def run(self, rate_hz: float = 50.0, max_steps: int | None = None) -> int:
        """
        Auto-lifecycle run: configure → activate → tick loop → deactivate → shutdown.
        """
        # Auto-transition: unconfigured → inactive → active
        if self._lifecycle_state == LifecycleState.UNCONFIGURED:
            if not self.trigger_transition(LifecycleTransition.CONFIGURE):
                return 0
        if self._lifecycle_state == LifecycleState.INACTIVE:
            if not self.trigger_transition(LifecycleTransition.ACTIVATE):
                return 0

        if self._lifecycle_state != LifecycleState.ACTIVE:
            logger.warning("Cannot run: not in ACTIVE state (got %s)", self._lifecycle_state.value)
            return 0

        period = 1.0 / max(rate_hz, 1.0)
        self._running = True
        steps = 0
        try:
            while self._running and self._lifecycle_state == LifecycleState.ACTIVE:
                t0 = time.perf_counter()
                try:
                    keep_going = bool(self.tick())
                except Exception:
                    logger.exception("Error in tick()")
                    self._enter_error()
                    break
                steps += 1
                if not keep_going:
                    break
                if max_steps is not None and steps >= max_steps:
                    break
                dt = time.perf_counter() - t0
                if dt < period:
                    time.sleep(period - dt)
        finally:
            self._running = False
            # Auto-transition: active → inactive → finalized
            if self._lifecycle_state == LifecycleState.ACTIVE:
                self.trigger_transition(LifecycleTransition.DEACTIVATE)
            if self._lifecycle_state == LifecycleState.INACTIVE:
                self.trigger_transition(LifecycleTransition.SHUTDOWN)

        return steps
