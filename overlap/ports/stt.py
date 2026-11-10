"""STT (Speech-to-Text) port and stream protocol definitions.

Pure port interface: typing.Protocol only, no implementation logic, no asyncio import.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Coroutine
from typing import Any, Protocol, runtime_checkable

from overlap.core.clock import CancelToken
from overlap.core.types import AudioFrame, SttEvent


@runtime_checkable
class SttStream(Protocol):
    """Active STT recognition stream session."""

    def send_audio(self, frame: AudioFrame) -> Coroutine[Any, Any, None]:
        """Enqueue an audio frame for speech recognition.

        Non-blocking: drops oldest on overflow to ensure fast-path safety.
        """
        ...

    def events(self) -> AsyncIterator[SttEvent]:
        """Asynchronous iterator yielding partial and final transcription events in order."""
        ...

    def close(self) -> Coroutine[Any, Any, None]:
        """Close the stream session. Must be idempotent."""
        ...


@runtime_checkable
class SttPort(Protocol):
    """Port for establishing speech-to-text streams."""

    def open(
        self, *, lang_hint: str | None = None, cancel: CancelToken
    ) -> Coroutine[Any, Any, SttStream]:
        """Open a new STT stream with optional language hint and cancel token."""
        ...
