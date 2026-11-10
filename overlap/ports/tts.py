"""TTS (Text-to-Speech) port, stream protocol, and chunk dataclasses.

Pure port interface: typing.Protocol only, no implementation logic, no asyncio import.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from overlap.core.clock import CancelToken
from overlap.core.types import WordTiming


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """Synthesized audio chunk carrying epoch and word timings."""

    pcm: bytes
    sample_rate: int
    epoch: int
    words: tuple[WordTiming, ...] = ()
    is_last: bool = False


@dataclass(frozen=True, slots=True)
class AudioClip:
    """Pre-synthesized audio clip for backchannels or cache."""

    pcm: bytes
    sample_rate: int


@runtime_checkable
class TtsStream(Protocol):
    """Active streaming TTS synthesis session."""

    def send_text(self, clause: str, *, ssml: bool = False) -> Coroutine[Any, Any, None]:
        """Enqueue text clause for synthesis."""
        ...

    def finish(self) -> Coroutine[Any, Any, None]:
        """Signal that no more text will be sent for this stream."""
        ...

    def chunks(self) -> AsyncIterator[AudioChunk]:
        """Asynchronous iterator yielding synthesized audio chunks."""
        ...

    def cancel(self) -> Coroutine[Any, Any, None]:
        """Cancel synthesis stream. Idempotent; must stop chunk delivery within 1 loop turn."""
        ...


@runtime_checkable
class TtsPort(Protocol):
    """Port for streaming TTS and batch synthesis."""

    def open_stream(
        self, *, voice_id: str, lang: str, epoch: int, cancel: CancelToken
    ) -> Coroutine[Any, Any, TtsStream]:
        """Open a streaming TTS session."""
        ...

    def synthesize(self, text: str, *, voice_id: str, lang: str) -> Coroutine[Any, Any, AudioClip]:
        """Batch synthesize a complete clip (used by ClipCache)."""
        ...
