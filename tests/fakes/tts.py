"""Fake implementation of TtsPort for testing and scenarios."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator

from overlap.core.clock import CancelToken, VirtualClock
from overlap.core.types import WordTiming
from overlap.ports.tts import AudioChunk, AudioClip, TtsPort, TtsStream


class FakeTtsStream(TtsStream):
    """Fake streaming TTS output producing deterministic PCM chunks."""

    def __init__(
        self,
        epoch: int,
        voice_id: str,
        lang: str,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 20,
        cadence_ms: float = 0.0,
        cancel: CancelToken | None = None,
        clock: VirtualClock | None = None,
    ) -> None:
        self.epoch = epoch
        self.voice_id = voice_id
        self.lang = lang
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.cadence_ms = cadence_ms
        self._cancel = cancel
        self._clock = clock

        self._cancelled = False
        self._finished = False
        self._pending_chunks: deque[AudioChunk] = deque()
        self._notify_event = asyncio.Event()
        self._text_buffer: list[str] = []

        if cancel is not None:
            if cancel.cancelled:
                self._handle_cancel()
            else:
                cancel.on_cancel(self._handle_cancel)

    def _handle_cancel(self) -> None:
        self._cancelled = True
        self._pending_chunks.clear()
        self._notify_event.set()

    async def send_text(self, clause: str, *, ssml: bool = False) -> None:
        """Enqueue text to synthesize into chunks."""
        if self._cancelled:
            return
        self._text_buffer.append(clause)

        # Generate deterministic chunks
        words = clause.split()
        bytes_per_chunk = int(self.sample_rate * 2 * (self.chunk_duration_ms / 1000.0))
        # Total chunks based on text length: at least 1 chunk per word or min 2 chunks
        chunks_count = max(len(words), 2)

        for i in range(chunks_count):
            # Compute word timings if words available
            word_timings: list[WordTiming] = []
            if i < len(words):
                w = words[i]
                start_ms = i * self.chunk_duration_ms
                end_ms = start_ms + self.chunk_duration_ms
                word_timings.append(WordTiming(word=w, start_ms=start_ms, end_ms=end_ms))

            # Deterministic dummy PCM pattern
            pcm = bytes([(i + j) % 256 for j in range(bytes_per_chunk)])
            chunk = AudioChunk(
                pcm=pcm,
                sample_rate=self.sample_rate,
                epoch=self.epoch,
                words=tuple(word_timings),
                is_last=False,
            )

            if self.cadence_ms > 0 and self._clock is not None:
                # Schedule in virtual time
                delay_ms = (i + 1) * self.cadence_ms
                self._clock.call_later(delay_ms, self._enqueue_chunk, chunk)
            else:
                self._enqueue_chunk(chunk)

    def _enqueue_chunk(self, chunk: AudioChunk) -> None:
        if self._cancelled:
            return
        self._pending_chunks.append(chunk)
        self._notify_event.set()

    async def finish(self) -> None:
        """Mark end of input text; marks final chunk."""
        if self._cancelled:
            return
        self._finished = True
        if self._pending_chunks:
            last = self._pending_chunks.pop()
            # replace with is_last=True
            self._pending_chunks.append(
                AudioChunk(
                    pcm=last.pcm,
                    sample_rate=last.sample_rate,
                    epoch=last.epoch,
                    words=last.words,
                    is_last=True,
                )
            )
        self._notify_event.set()

    async def chunks(self) -> AsyncIterator[AudioChunk]:
        """Yield generated audio chunks."""
        while not self._cancelled:
            if self._cancel is not None and self._cancel.cancelled:
                self._handle_cancel()
                break

            if self._pending_chunks:
                chunk = self._pending_chunks.popleft()
                yield chunk
                continue

            if self._finished and not self._pending_chunks:
                break

            self._notify_event.clear()
            await self._notify_event.wait()

    async def cancel(self) -> None:
        """Cancel the stream idempotently and halt emissions."""
        if self._cancelled:
            return
        self._handle_cancel()


class FakeTts(TtsPort):
    """Fake TTS engine implementing TtsPort."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 20,
        cadence_ms: float = 0.0,
        clock: VirtualClock | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.cadence_ms = cadence_ms
        self.clock = clock

    async def open_stream(
        self,
        voice_id: str,
        lang: str,
        epoch: int,
        cancel: CancelToken | None = None,
    ) -> TtsStream:
        """Open a streaming TTS synthesis session."""
        return FakeTtsStream(
            epoch=epoch,
            voice_id=voice_id,
            lang=lang,
            sample_rate=self.sample_rate,
            chunk_duration_ms=self.chunk_duration_ms,
            cadence_ms=self.cadence_ms,
            cancel=cancel,
            clock=self.clock,
        )

    async def synthesize(self, text: str, voice_id: str = "v1", lang: str = "hi") -> AudioClip:
        """Batch synthesize text into an AudioClip."""
        # 16-bit PCM (2 bytes per sample), 100ms
        sample_count = int(self.sample_rate * 0.1)
        pcm = b"\x00" * (sample_count * 2)
        return AudioClip(pcm=pcm, sample_rate=self.sample_rate)
