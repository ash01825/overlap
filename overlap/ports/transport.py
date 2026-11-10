"""Audio transport port protocol definitions (e.g. LiveKit).

Pure port interface: typing.Protocol only, no implementation logic, no asyncio import.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from overlap.core.types import AudioFrame


@runtime_checkable
class AudioTransport(Protocol):
    """Bi-directional audio transport interface."""

    def user_frames(self) -> AsyncIterator[AudioFrame]:
        """Asynchronous stream of incoming user audio frames (20 ms cadence)."""
        ...

    def send_frame(self, frame: AudioFrame) -> None:
        """Synchronously enqueue or transmit an agent audio frame (fast path, non-blocking)."""
        ...

    def flush_output(self) -> None:
        """Drop all audio queued downstream in the transport."""
        ...

    def output_queue_ms(self) -> int:
        """Duration in milliseconds of agent audio currently queued in transport."""
        ...

    def playout_position_ms(self) -> int | None:
        """Best estimate of played position in ms heard by user, or None if unavailable."""
        ...
