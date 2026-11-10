"""Unit tests for Clock, VirtualClock, MonotonicClock, and CancelToken."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

from overlap.core.clock import (
    CancelledError,
    CancelToken,
    MonotonicClock,
    TimerHandle,
    VirtualClock,
)

_helpers_dir = str(Path(__file__).resolve().parent.parent.parent / "helpers")
if _helpers_dir not in sys.path:
    sys.path.insert(0, _helpers_dir)

from virtual import run_virtual  # noqa: E402


def test_virtual_clock_deterministic_ordering() -> None:
    """VirtualClock fires timers ordered by (deadline_ns, insertion_seq)."""
    clock = VirtualClock()
    fired: list[str] = []

    # Schedule multiple timers at the exact same deadline
    clock.call_at(100_000_000, lambda: fired.append("first_at_100"))
    clock.call_at(100_000_000, lambda: fired.append("second_at_100"))
    clock.call_at(50_000_000, lambda: fired.append("at_50"))
    clock.call_at(100_000_000, lambda: fired.append("third_at_100"))
    clock.call_at(200_000_000, lambda: fired.append("at_200"))

    clock.advance(100)  # advance 100 ms = 100_000_000 ns

    assert fired == ["at_50", "first_at_100", "second_at_100", "third_at_100"]
    assert clock.now_ns() == 100_000_000

    clock.advance(100)  # advance another 100 ms -> 200 ms
    assert fired == ["at_50", "first_at_100", "second_at_100", "third_at_100", "at_200"]


def test_virtual_clock_past_timers() -> None:
    """Timers scheduled in the past fire immediately on next advance with deterministic ordering."""
    clock = VirtualClock(start_ns=1_000_000_000)  # at 1000 ms
    fired: list[str] = []

    # Schedule with deadline in the past or negative delay
    clock.call_at(500_000_000, lambda: fired.append("past_500"))
    clock.call_later(-10, lambda: fired.append("neg_delay_1"))
    clock.call_later(-5, lambda: fired.append("neg_delay_2"))

    assert fired == []

    # Advance 0 ms: should trigger all past due timers in order
    clock.advance(0)
    assert fired == ["past_500", "neg_delay_1", "neg_delay_2"]


def test_virtual_clock_cancellation() -> None:
    """VirtualClock handles cancellation via handle.cancel() and clock.cancel()."""
    clock = VirtualClock()
    fired: list[str] = []

    h1 = clock.call_later(50, lambda: fired.append("h1"))
    h2 = clock.call_later(50, lambda: fired.append("h2"))
    h3 = clock.call_later(100, lambda: fired.append("h3"))

    assert not h1.cancelled
    assert not h2.cancelled

    h1.cancel()
    assert h1.cancelled
    clock.cancel(h2)
    assert h2.cancelled

    # Cancelling again must be idempotent
    h1.cancel()
    clock.cancel(h2)

    clock.advance(100)
    assert fired == ["h3"]

    # Cancelling an already fired timer is a safe no-op
    assert not h3.cancelled
    h3.cancel()


def test_virtual_clock_cascading_timers_within_advance() -> None:
    """Timers scheduled during an advance step fire if their deadline is within target."""
    clock = VirtualClock()
    fired: list[str] = []

    def on_step1() -> None:
        fired.append("step1")
        # Schedule step 2 at 80 ms (target is 100 ms)
        clock.call_at(80_000_000, lambda: fired.append("step2"))
        # Schedule step 3 at 150 ms (beyond target)
        clock.call_at(150_000_000, lambda: fired.append("step3"))

    clock.call_at(50_000_000, on_step1)
    clock.advance(100)

    assert fired == ["step1", "step2"]
    assert clock.now_ns() == 100_000_000

    clock.advance(50)
    assert fired == ["step1", "step2", "step3"]


def test_10_minute_virtual_time_under_200ms() -> None:
    """Evidence requirement: 10-minute virtual-time test finishing in < 0.2 s real time."""
    clock = VirtualClock()
    ticks: list[int] = []

    # Schedule a recurring 1-second timer across 10 minutes (600 seconds)
    def schedule_next(count: int) -> None:
        ticks.append(count)
        if count < 600:
            clock.call_later(1000, lambda: schedule_next(count + 1))

    clock.call_later(1000, lambda: schedule_next(1))

    t_start = time.perf_counter()
    clock.advance(600 * 1000)  # advance 10 minutes = 600,000 ms
    elapsed = time.perf_counter() - t_start

    assert len(ticks) == 600
    assert clock.now_ns() == 600_000_000_000
    assert elapsed < 0.2, f"10-min virtual time test took {elapsed:.4f} s (budget: < 0.2 s)"


def test_monotonic_clock() -> None:
    """Test MonotonicClock basic time and scheduling operations."""
    clock = MonotonicClock()
    t1 = clock.now_ns()
    assert isinstance(t1, int)
    assert t1 > 0

    # Custom time source
    fixed_clock = MonotonicClock(time_fn=lambda: 987654321)
    assert fixed_clock.now_ns() == 987654321

    # Call later and cancellation
    fired = []
    handle = clock.call_later(10000, lambda: fired.append("never"))
    assert isinstance(handle, TimerHandle)
    assert not handle.cancelled
    handle.cancel()
    assert handle.cancelled


def test_cancel_token_hierarchy() -> None:
    """Parent cancellation cascades to children; children do not cancel parents."""
    parent = CancelToken()
    child1 = parent.child()
    child2 = parent.child()
    grandchild = child1.child()

    assert not parent.cancelled
    assert not child1.cancelled
    assert not child2.cancelled
    assert not grandchild.cancelled

    # Cancelling child2 cancels only child2
    child2.cancel()
    assert child2.cancelled
    assert not parent.cancelled
    assert not child1.cancelled
    assert not grandchild.cancelled

    # Cancelling parent cascades to child1 and grandchild
    parent.cancel()
    assert parent.cancelled
    assert child1.cancelled
    assert grandchild.cancelled

    # A new child created from an already cancelled parent is cancelled immediately
    late_child = parent.child()
    assert late_child.cancelled


def test_cancel_token_callbacks_and_idempotence() -> None:
    """Cancel callbacks run once and idempotent cancel does not re-run them."""
    token = CancelToken()
    fired: list[str] = []

    token.add_callback(lambda: fired.append("cb1"))
    token.register_callback(lambda: fired.append("cb2"))
    unsub_to_cancel = token.on_cancel(lambda: fired.append("cb_unsubbed"))
    unsub_to_cancel()  # remove before cancel

    assert fired == []
    token.cancel()
    assert fired == ["cb1", "cb2"]

    # Calling cancel() again does not re-run callbacks
    token.cancel()
    assert fired == ["cb1", "cb2"]

    # Registering on an already cancelled token runs callback immediately
    token.add_callback(lambda: fired.append("cb_late"))
    assert fired == ["cb1", "cb2", "cb_late"]


def test_cancel_token_exceptions_not_swallowed() -> None:
    """Exceptions in cancel callbacks are not swallowed, and all callbacks are executed."""
    token = CancelToken()
    fired: list[str] = []

    def bad_callback() -> None:
        fired.append("bad")
        raise ValueError("Boom in cancel callback")

    token.add_callback(bad_callback)
    token.add_callback(lambda: fired.append("good"))

    with pytest.raises((ValueError, ExceptionGroup)):
        token.cancel()

    # Both callbacks had a chance to execute
    assert "bad" in fired
    assert "good" in fired
    assert token.cancelled


def test_cancel_token_raise_if_cancelled() -> None:
    """raise_if_cancelled raises CancelledError when cancelled."""
    token = CancelToken()
    token.raise_if_cancelled()  # does not raise

    token.cancel()
    with pytest.raises(CancelledError):
        token.raise_if_cancelled()


def test_run_virtual_simple() -> None:
    """Test run_virtual executes async coroutine and returns its result."""

    async def sample() -> int:
        return 42

    result = run_virtual(sample())
    assert result == 42


def test_run_virtual_advance_drains_to_quiescence() -> None:
    """Ensure advance() drains the loop to quiescence even across chained tasks."""
    clock = VirtualClock()
    state: list[str] = []

    async def worker() -> None:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()

        # Timer fires after 50 ms virtual time
        clock.call_later(50, lambda: fut.set_result(None))

        await fut
        state.append("resumed")

        # Create a chained background task that does more work
        async def background_step() -> None:
            state.append("chained_background")

        asyncio.create_task(background_step())

    async def main() -> None:
        asyncio.create_task(worker())
        await asyncio.sleep(0)
        assert state == []

        # Advancing 50 ms should fire timer, resolve fut, resume worker,
        # and drain the chained background task so it executes BEFORE advance() returns!
        await clock.advance(50)
        assert state == ["resumed", "chained_background"]

    run_virtual(main(), clock=clock)
