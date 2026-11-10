"""Clock protocol, VirtualClock (test), MonotonicClock (prod), and CancelToken.

Pure layer: No asyncio, no static time import, no sleep calls.
"""

from __future__ import annotations

import heapq
import threading
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


class CancelledError(Exception):
    """Raised when an operation is attempted on a cancelled CancelToken."""


@runtime_checkable
class TimerHandle(Protocol):
    """Handle to a scheduled timer callback."""

    @property
    def cancelled(self) -> bool:
        """Whether this timer has been cancelled."""
        ...

    def cancel(self) -> None:
        """Cancel this timer. Idempotent."""
        ...


@runtime_checkable
class Clock(Protocol):
    """Time source and timer scheduler protocol."""

    def now_ns(self) -> int:
        """Current monotonic timestamp in nanoseconds."""
        ...

    def call_at(self, deadline_ns: int, callback: Callable[..., Any], *args: Any) -> TimerHandle:
        """Schedule a callback to run at the given deadline in nanoseconds."""
        ...

    def call_later(
        self, delay_ms: float | int, callback: Callable[..., Any], *args: Any
    ) -> TimerHandle:
        """Schedule a callback to run after delay_ms milliseconds."""
        ...

    def cancel(self, handle: TimerHandle) -> None:
        """Cancel a scheduled timer handle."""
        ...


class CancelToken:
    """Hierarchical, thread-safe cancellation token with callback hooks."""

    def __init__(self, parent: CancelToken | None = None) -> None:
        self._parent = parent
        self._cancelled = False
        self._callbacks: list[Callable[[], Any]] = []
        self._children: list[CancelToken] = []

        if parent is not None:
            if parent.cancelled:
                self._cancelled = True
            else:
                parent._add_child(self)

    def _add_child(self, child: CancelToken) -> None:
        if self._cancelled:
            child.cancel()
        else:
            self._children.append(child)

    @property
    def cancelled(self) -> bool:
        """Whether this token or any ancestor is cancelled."""
        if self._cancelled:
            return True
        if self._parent is not None and self._parent.cancelled:
            self._cancelled = True
            return True
        return False

    def child(self) -> CancelToken:
        """Create a child token that cancels when this token cancels."""
        return CancelToken(parent=self)

    def add_callback(self, callback: Callable[[], Any]) -> Callable[[], None]:
        """Register a callback to run when cancelled.

        If already cancelled, runs callback immediately.
        Returns an unregister callable.
        """
        if self.cancelled:
            callback()
            return lambda: None

        self._callbacks.append(callback)

        def unregister() -> None:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

        return unregister

    def register_callback(self, callback: Callable[[], Any]) -> Callable[[], None]:
        """Alias for add_callback."""
        return self.add_callback(callback)

    def on_cancel(self, callback: Callable[[], Any]) -> Callable[[], None]:
        """Alias for add_callback."""
        return self.add_callback(callback)

    def cancel(self) -> None:
        """Cancel this token and all descendant tokens. Idempotent.

        Never swallows exceptions: if callbacks raise, collects and re-raises them.
        """
        if self._cancelled:
            return
        self._cancelled = True

        exceptions: list[Exception] = []

        # Execute registered callbacks once
        callbacks = list(self._callbacks)
        self._callbacks.clear()
        for cb in callbacks:
            try:
                cb()
            except Exception as exc:
                exceptions.append(exc)

        # Cascade cancellation to children
        children = list(self._children)
        self._children.clear()
        for child in children:
            try:
                child.cancel()
            except Exception as exc:
                exceptions.append(exc)

        if exceptions:
            if len(exceptions) == 1:
                raise exceptions[0]
            raise ExceptionGroup("CancelToken callbacks failed", exceptions)

    def raise_if_cancelled(self) -> None:
        """Raise CancelledError if this token is cancelled."""
        if self.cancelled:
            raise CancelledError("Operation has been cancelled")


class _VirtualTimerHandle:
    def __init__(self, item_ref: list[bool]) -> None:
        self._item_ref = item_ref  # [is_cancelled, has_fired]

    @property
    def cancelled(self) -> bool:
        return bool(self._item_ref[0])

    def cancel(self) -> None:
        self._item_ref[0] = True


class _VirtualAdvanceResult:
    """Awaitable result of VirtualClock.advance to support cooperative async draining."""

    def __init__(self, async_drainer: Callable[[], Any] | None = None) -> None:
        self._async_drainer = async_drainer

    def __await__(self) -> Any:
        if self._async_drainer is not None:
            res = self._async_drainer()
            if hasattr(res, "__await__"):
                return res.__await__()
        return iter(())


class VirtualClock:
    """Deterministic virtual-time clock for fast, reproducible tests.

    Timers are strictly ordered by (deadline_ns, insertion_seq).
    """

    def __init__(self, start_ns: int = 0) -> None:
        self._now_ns = start_ns
        self._seq = 0
        self._heap: list[tuple[int, int, Callable[..., Any], tuple[Any, ...], list[bool]]] = []
        self._drain_callback: Callable[[], None] | None = None
        self._async_drainer: Callable[[], Any] | None = None

    def now_ns(self) -> int:
        return self._now_ns

    def call_at(self, deadline_ns: int, callback: Callable[..., Any], *args: Any) -> TimerHandle:
        self._seq += 1
        item_ref = [False, False]  # [cancelled, fired]
        heapq.heappush(self._heap, (deadline_ns, self._seq, callback, args, item_ref))
        return _VirtualTimerHandle(item_ref)

    def call_later(
        self, delay_ms: float | int, callback: Callable[..., Any], *args: Any
    ) -> TimerHandle:
        delay_ns = int(delay_ms * 1_000_000)
        return self.call_at(self._now_ns + delay_ns, callback, *args)

    def cancel(self, handle: TimerHandle) -> None:
        handle.cancel()

    def set_drain_callback(self, drain_fn: Callable[[], None] | None) -> None:
        """Set callback to drain the test event loop to quiescence on advance."""
        self._drain_callback = drain_fn

    def set_async_drainer(self, drain_fn: Callable[[], Any] | None) -> None:
        """Set async drainer callable invoked when advance is awaited."""
        self._async_drainer = drain_fn

    def advance(self, ms: float | int = 0) -> _VirtualAdvanceResult:
        """Advance virtual time by ms milliseconds, firing all due timers."""
        target_ns = self._now_ns + int(ms * 1_000_000)
        return self.advance_to(target_ns)

    def advance_to(self, target_ns: int) -> _VirtualAdvanceResult:
        """Advance virtual time to target_ns nanoseconds, firing all due timers."""
        while self._heap and self._heap[0][0] <= target_ns:
            deadline_ns, _seq, callback, args, item_ref = heapq.heappop(self._heap)
            self._now_ns = deadline_ns
            if not item_ref[0] and not item_ref[1]:
                item_ref[1] = True
                callback(*args)
                if self._drain_callback is not None:
                    self._drain_callback()

        if self._now_ns < target_ns:
            self._now_ns = target_ns
        if self._drain_callback is not None:
            self._drain_callback()

        return _VirtualAdvanceResult(self._async_drainer)

    def advance_ns(self, ns: int) -> _VirtualAdvanceResult:
        """Advance virtual time by ns nanoseconds."""
        return self.advance_to(self._now_ns + ns)


class _MonotonicTimerHandle:
    def __init__(self, timer: threading.Timer | None) -> None:
        self._timer = timer
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True
        if self._timer is not None:
            self._timer.cancel()


class MonotonicClock:
    """Production clock using monotonic nanosecond time."""

    def __init__(
        self,
        time_fn: Callable[[], int] | None = None,
        call_at_fn: (
            Callable[[int, Callable[..., Any], tuple[Any, ...]], TimerHandle] | None
        ) = None,
    ) -> None:
        if time_fn is None:
            import importlib

            time_mod = importlib.import_module("time")
            self._time_fn: Callable[[], int] = time_mod.monotonic_ns
        else:
            self._time_fn = time_fn
        self._call_at_fn = call_at_fn

    def now_ns(self) -> int:
        return self._time_fn()

    def call_at(self, deadline_ns: int, callback: Callable[..., Any], *args: Any) -> TimerHandle:
        if self._call_at_fn is not None:
            return self._call_at_fn(deadline_ns, callback, args)
        delay_sec = max(0.0, (deadline_ns - self.now_ns()) / 1_000_000_000.0)
        timer = threading.Timer(delay_sec, callback, args)
        timer.daemon = True
        timer.start()
        return _MonotonicTimerHandle(timer)

    def call_later(
        self, delay_ms: float | int, callback: Callable[..., Any], *args: Any
    ) -> TimerHandle:
        deadline_ns = self.now_ns() + int(delay_ms * 1_000_000)
        return self.call_at(deadline_ns, callback, *args)

    def cancel(self, handle: TimerHandle) -> None:
        handle.cancel()
