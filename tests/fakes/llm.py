"""Fake implementation of LlmPort for testing and scenarios."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Sequence

from overlap.core.clock import CancelToken, VirtualClock
from overlap.ports.llm import LlmPort, LlmStream, Message


class FakeLlmStream(LlmStream):
    """Fake LLM stream producing scripted text deltas."""

    def __init__(
        self,
        deltas: list[str],
        epoch: int = 1,
        cancel: CancelToken | None = None,
        clock: VirtualClock | None = None,
    ) -> None:
        self.epoch = epoch
        self._deltas = deque(deltas)
        self._cancel = cancel
        self._clock = clock
        self._aborted = False
        self._notify_event = asyncio.Event()

        if cancel is not None:
            if cancel.cancelled:
                self._handle_abort()
            else:
                cancel.on_cancel(self._handle_abort)

    def _handle_abort(self) -> None:
        self._aborted = True
        self._deltas.clear()
        self._notify_event.set()

    async def deltas(self) -> AsyncIterator[str]:
        """Yield scripted text deltas."""
        while not self._aborted:
            if self._cancel is not None and self._cancel.cancelled:
                self._handle_abort()
                break

            if self._deltas:
                yield self._deltas.popleft()
                continue

            break

    async def abort(self) -> None:
        """Abort generation idempotently."""
        if self._aborted:
            return
        self._handle_abort()


class FakeLlm(LlmPort):
    """Fake LLM engine implementing LlmPort."""

    def __init__(
        self,
        deltas: list[str] | None = None,
        clock: VirtualClock | None = None,
    ) -> None:
        self.deltas = list(deltas) if deltas is not None else ["Haan, ", "main ", "sun raha hoon."]
        self.clock = clock

    async def stream(
        self,
        messages: Sequence[Message],
        epoch: int = 1,
        cancel: CancelToken | None = None,
    ) -> LlmStream:
        """Open a streaming response."""
        return FakeLlmStream(
            deltas=list(self.deltas),
            epoch=epoch,
            cancel=cancel,
            clock=self.clock,
        )
