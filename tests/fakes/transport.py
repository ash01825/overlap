"""Fake implementation of AudioTransport for testing and scenarios."""

from __future__ import annotations

import asyncio
import math
import struct
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

from overlap.core.clock import VirtualClock
from overlap.core.types import AudioFrame, Tag
from overlap.ports.transport import AudioTransport


def _calc_pcm16_rms(pcm: bytes) -> float:
    """Calculate linear RMS of 16-bit little-endian PCM audio."""
    num_samples = len(pcm) // 2
    if num_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{num_samples}h", pcm)
    sum_squares = sum(s * s for s in samples)
    return math.sqrt(sum_squares / num_samples)


@dataclass(frozen=True, slots=True)
class SentFrameRecord:
    """Record of a frame sent via FakeTransport."""

    ts: int
    tags: Tag
    rms: float
    frame: AudioFrame


class FakeTransport(AudioTransport):
    """Fake audio transport recording sent frames, modeling downstream latency and flush."""

    def __init__(
        self,
        downstream_latency_ms: int = 0,
        clock: VirtualClock | None = None,
    ) -> None:
        self.downstream_latency_ms = downstream_latency_ms
        self.clock = clock

        self.sent_frames: list[SentFrameRecord] = []
        self._output_queue: deque[AudioFrame] = deque()
        self.flushes: int = 0
        self.flush_history: list[int] = []

        self._incoming_user_frames: deque[AudioFrame] = deque()
        self._user_frame_event = asyncio.Event()
        self._closed = False

    def send_frame(self, frame: AudioFrame) -> None:
        """Record and enqueue an outbound audio frame synchronously."""
        rms = _calc_pcm16_rms(frame.pcm)
        rec = SentFrameRecord(
            ts=frame.ts_ns,
            tags=frame.tags,
            rms=rms,
            frame=frame,
        )
        self.sent_frames.append(rec)
        self._output_queue.append(frame)

    def flush_output(self) -> None:
        """Clear queued audio synchronously."""
        self._output_queue.clear()
        self.flushes += 1
        now_ms = (self.clock.now_ns() // 1_000_000) if self.clock is not None else 0
        self.flush_history.append(now_ms)

    def output_queue_ms(self) -> int:
        """Return estimated output queue duration in ms including downstream latency."""
        if not self._output_queue:
            return 0
        total_samples = sum(len(f.pcm) // 2 for f in self._output_queue)
        sample_rate = self._output_queue[0].sample_rate if self._output_queue else 16000
        buffered_ms = int(round((total_samples / sample_rate) * 1000))
        return buffered_ms + self.downstream_latency_ms

    def playout_position_ms(self) -> int | None:
        """Return playout position in ms."""
        if self.clock is not None:
            return self.clock.now_ns() // 1_000_000
        return 0

    def push_user_frame(self, frame: AudioFrame) -> None:
        """Helper to feed an inbound user audio frame."""
        self._incoming_user_frames.append(frame)
        self._user_frame_event.set()

    async def user_frames(self) -> AsyncIterator[AudioFrame]:
        """Async iterator yielding incoming user audio frames."""
        while not self._closed:
            if self._incoming_user_frames:
                yield self._incoming_user_frames.popleft()
                continue
            self._user_frame_event.clear()
            await self._user_frame_event.wait()

    def close(self) -> None:
        """Close the transport."""
        self._closed = True
        self._user_frame_event.set()
