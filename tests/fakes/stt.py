"""Fake implementation of SttPort for testing and scenarios."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from overlap.core.clock import CancelToken, VirtualClock
from overlap.core.types import AudioFrame, SttEvent
from overlap.ports.stt import SttPort, SttStream

if TYPE_CHECKING:
    pass


class FakeSttStream(SttStream):
    """In-memory scriptable stream for FakeStt."""

    def __init__(
        self,
        timeline: list[tuple[int, SttEvent]],
        cancel: CancelToken | None = None,
        clock: VirtualClock | None = None,
        max_queue_size: int = 100,
    ) -> None:
        self._timeline = sorted(timeline, key=lambda x: x[0])
        self._cancel = cancel
        self._clock = clock
        self._max_queue_size = max_queue_size
        self._audio_queue: deque[AudioFrame] = deque(maxlen=max_queue_size)
        self._event_queue: deque[SttEvent] = deque()
        self._notify_event = asyncio.Event()
        self._closed = False
        self._cancelled = False

        if cancel is not None:
            if cancel.cancelled:
                self._handle_cancel()
            else:
                cancel.on_cancel(self._handle_cancel)

        # Schedule timeline events
        for t_ms, evt in self._timeline:
            if clock is not None:
                # Schedule in virtual time
                clock.call_later(t_ms, self._schedule_event_cb, evt)
            else:
                self.push_event(evt)

    def _schedule_event_cb(self, evt: SttEvent) -> None:
        if not self._closed and not self._cancelled:
            self.push_event(evt)

    def _handle_cancel(self) -> None:
        self._cancelled = True
        self._event_queue.clear()
        self._notify_event.set()

    def push_event(self, event: SttEvent) -> None:
        """Push an STT event into the output queue."""
        if self._closed or self._cancelled:
            return
        self._event_queue.append(event)
        self._notify_event.set()

    async def send_audio(self, frame: AudioFrame) -> None:
        """Send an audio frame into the stream. Drops oldest if full."""
        if self._closed or self._cancelled:
            return
        self._audio_queue.append(frame)

    async def events(self) -> AsyncIterator[SttEvent]:
        """Async iterator yielding STT events."""
        while not self._closed and not self._cancelled:
            if self._cancel is not None and self._cancel.cancelled:
                self._handle_cancel()
                break

            if self._event_queue:
                yield self._event_queue.popleft()
                continue

            # Wait for next event or close/cancel
            self._notify_event.clear()
            await self._notify_event.wait()

    async def close(self) -> None:
        """Close the stream idempotently."""
        if self._closed:
            return
        self._closed = True
        self._event_queue.clear()
        self._notify_event.set()


class FakeStt(SttPort):
    """Fake STT engine implementing SttPort with scriptable event timeline."""

    def __init__(
        self,
        timeline: list[tuple[int, SttEvent]] | None = None,
        partial_latency_ms: float = 0.0,
        clock: VirtualClock | None = None,
    ) -> None:
        self._timeline: list[tuple[int, SttEvent]] = list(timeline or [])
        self.partial_latency_ms = partial_latency_ms
        self.clock = clock

    def add_event(self, t_ms: int, event: SttEvent) -> None:
        """Add a scripted STT event at t_ms."""
        self._timeline.append((t_ms, event))

    async def open(
        self,
        *,
        lang_hint: str | None = None,
        cancel: CancelToken | None = None,
    ) -> SttStream:
        """Open a new STT stream."""
        return FakeSttStream(
            timeline=self._timeline,
            cancel=cancel,
            clock=self.clock,
        )
