"""Async test helper hosting coroutines on a VirtualClock with quiescent loop draining."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

from overlap.core.clock import VirtualClock

T = TypeVar("T")


class VirtualEventLoop(asyncio.SelectorEventLoop):
    """Hermetic event loop that does not create socketpairs for self-pipes."""

    def _make_self_pipe(self) -> None:
        self._ssock = None
        self._csock = None

    def _close_self_pipe(self) -> None:
        pass

    def _write_to_self(self) -> None:
        pass


async def advance_virtual(clock: VirtualClock, ms: float | int = 0) -> None:
    """Advance VirtualClock and drain the event loop to quiescence."""
    await clock.advance(ms)


def run_virtual(
    coro: Coroutine[Any, Any, T],
    clock: VirtualClock | None = None,
) -> T:
    """Run an async coroutine with a VirtualClock, with no real sleeping and quiescent draining."""
    if clock is None:
        clock = VirtualClock()

    loop = VirtualEventLoop()
    asyncio.set_event_loop(loop)

    async def drain_async() -> None:
        while bool(getattr(loop, "_ready", None)):
            await asyncio.sleep(0)

    clock.set_async_drainer(drain_async)

    try:
        return loop.run_until_complete(coro)
    finally:
        clock.set_async_drainer(None)
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        asyncio.set_event_loop(None)
