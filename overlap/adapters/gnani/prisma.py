"""Prisma STT adapter implementing SttPort and SttStream protocols.

Full-duplex WebSocket streaming, interim partials, auto-detected language,
drop-oldest bounded send queue, and backoff reconnect with audio replay buffer.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

from overlap.adapters.gnani.wire import (
    DEFAULT_USER_AGENT,
    PRISMA_FRAME_MS,
    PRISMA_STT_WS_URL,
    AsyncWebSocket,
    GnaniAuthenticationError,
    build_prisma_headers,
)
from overlap.core.clock import CancelToken, Clock
from overlap.core.types import AudioFrame, SttEvent, WordTiming
from overlap.ports.stt import SttPort, SttStream
from overlap.ports.telemetry import TelemetrySink

logger = logging.getLogger(__name__)


class PrismaSttStream(SttStream):
    """Active streaming STT session connected to Gnani Prisma."""

    def __init__(
        self,
        *,
        ws: Any,
        cancel: CancelToken,
        ws_connect_fn: Callable[[], Coroutine[Any, Any, Any]] | None = None,
        telemetry: TelemetrySink | None = None,
        clock: Clock | None = None,
        language_code: str = "indic-auto",
        max_reconnect_attempts: int = 3,
        reconnect_backoff_initial_ms: int = 100,
        reconnect_backoff_max_ms: int = 2000,
        max_queue_frames: int = 100,
        replay_buffer_ms: int = 2000,
    ) -> None:
        self._ws = ws
        self._cancel = cancel
        self._ws_connect_fn = ws_connect_fn
        self._telemetry = telemetry
        self._clock = clock
        self._language_code = language_code
        self._current_lang: str = language_code
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_backoff_initial_ms = reconnect_backoff_initial_ms
        self._reconnect_backoff_max_ms = reconnect_backoff_max_ms
        self._max_queue_frames = max_queue_frames

        # Calculate replay buffer size in frames (2000 ms / 20 ms = 100 frames)
        max_replay_frames = max(1, replay_buffer_ms // PRISMA_FRAME_MS)
        self._replay_buffer: deque[AudioFrame] = deque(maxlen=max_replay_frames)

        self._send_queue: deque[AudioFrame] = deque()
        self._send_event = asyncio.Event()
        self._closed: bool = False
        self._dropped_frames_total: int = 0
        self._reconnect_attempts: int = 0
        self._last_emitted_ts_ns: int = 0

        self._finalized_utterances: set[str] = set()
        self._last_partial_seq: dict[str, int] = {}
        self._error_events: list[dict[str, Any]] = []
        self._last_error: dict[str, Any] | None = None

        self._send_task: asyncio.Task[None] | None = asyncio.create_task(self._send_loop())

    @property
    def dropped_frames_total(self) -> int:
        """Total number of audio frames dropped due to buffer overflow."""
        return self._dropped_frames_total

    @property
    def reconnect_attempts(self) -> int:
        """Total number of reconnect attempts made."""
        return self._reconnect_attempts

    @property
    def last_error(self) -> dict[str, Any] | None:
        """Last error message dictionary received or surfaced."""
        return self._last_error

    @property
    def has_error(self) -> bool:
        """Whether any error frame or fatal condition was encountered."""
        return self._last_error is not None or len(self._error_events) > 0

    @property
    def error_events(self) -> list[dict[str, Any]]:
        """List of all error frames received during the session."""
        return list(self._error_events)

    async def _send_loop(self) -> None:
        """Background worker continuously draining send queue to the WebSocket."""
        while not self._closed and not self._cancel.cancelled:
            while self._send_queue and not self._closed and not self._cancel.cancelled:
                frame = self._send_queue.popleft()
                try:
                    await self._ws.send_bytes(frame.pcm)
                except Exception:
                    # Send errors will be handled or reconnected by the main recv/reconnect loop
                    break

            self._send_event.clear()
            try:
                await self._send_event.wait()
            except asyncio.CancelledError:
                break

    async def send_audio(self, frame: AudioFrame) -> None:
        """Enqueue an audio frame for speech recognition.

        Non-blocking: drops oldest frame on queue overflow to ensure fast-path safety.
        """
        if self._closed or self._cancel.cancelled:
            return

        # Keep sliding history in replay buffer for reconnects (last <= 2 s)
        self._replay_buffer.append(frame)

        # Enforce bounded send queue
        if len(self._send_queue) >= self._max_queue_frames:
            self._send_queue.popleft()
            self._dropped_frames_total += 1
            if self._telemetry is not None:
                self._telemetry.emit(
                    "stt.dropped_frames",
                    ts_ns=frame.ts_ns,
                    count=1,
                    total=self._dropped_frames_total,
                )

        self._send_queue.append(frame)
        self._send_event.set()

    def _extract_words(self, words_raw: list[Any], audio_ms: int) -> tuple[WordTiming, ...]:
        """Convert raw word dictionaries to typed WordTiming tuples."""
        results: list[WordTiming] = []
        for item in words_raw:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("word") or "")
                start_ms = int(item.get("start_ms") or item.get("start") or 0)
                end_ms = int(item.get("end_ms") or item.get("end") or audio_ms)
                results.append(WordTiming(word=text, start_ms=start_ms, end_ms=end_ms))
        return tuple(results)

    def _process_partial(self, data: dict[str, Any]) -> SttEvent | None:
        """Process partial_transcript frame, filtering out-of-order or late partials."""
        utterance_id = str(data.get("utterance_id", ""))
        if utterance_id and utterance_id in self._finalized_utterances:
            # Drop partial arriving after final transcript
            return None

        seq = data.get("seq")
        if seq is not None and isinstance(seq, int):
            last_seq = self._last_partial_seq.get(utterance_id, -1)
            if seq <= last_seq:
                # Drop duplicate or out-of-order partial
                return None
            self._last_partial_seq[utterance_id] = seq

        text = str(data.get("text", ""))
        lang = data.get("detected_language_code") or data.get("lang") or self._current_lang

        conf = float(data["confidence"]) if "confidence" in data else None

        audio_ms = data.get("audio_ms") or data.get("t_ms")
        if audio_ms is not None:
            ts_ns = int(audio_ms) * 1_000_000
        elif self._clock is not None:
            ts_ns = self._clock.now_ns()
        else:
            ts_ns = self._last_emitted_ts_ns + 1

        ts_ns = max(ts_ns, self._last_emitted_ts_ns)
        self._last_emitted_ts_ns = ts_ns

        words = self._extract_words(data.get("words", []), audio_ms=int(audio_ms or 0))

        return SttEvent(kind="partial", text=text, lang=lang, conf=conf, ts_ns=ts_ns, words=words)

    def _process_final(self, data: dict[str, Any]) -> SttEvent | None:
        """Process final_transcript frame, marking utterance finalized."""
        utterance_id = str(data.get("utterance_id", ""))
        if utterance_id:
            self._finalized_utterances.add(utterance_id)

        text = str(data.get("text", ""))
        lang = data.get("detected_language_code") or data.get("lang") or self._current_lang
        if lang:
            self._current_lang = lang

        conf = float(data["confidence"]) if "confidence" in data else None

        duration_ms = data.get("audio_duration_ms") or data.get("audio_ms")
        if duration_ms is not None:
            ts_ns = int(float(duration_ms)) * 1_000_000
        elif self._clock is not None:
            ts_ns = self._clock.now_ns()
        else:
            ts_ns = self._last_emitted_ts_ns + 1

        ts_ns = max(ts_ns, self._last_emitted_ts_ns)
        self._last_emitted_ts_ns = ts_ns

        words = self._extract_words(data.get("words", []), audio_ms=int(duration_ms or 0))

        return SttEvent(kind="final", text=text, lang=lang, conf=conf, ts_ns=ts_ns, words=words)

    def _handle_error_frame(self, data: dict[str, Any]) -> None:
        """Handle server error frame, surfacing to telemetry and stream error state."""
        logger.error("Prisma STT server error: %s", data)
        self._error_events.append(data)
        self._last_error = data
        if self._telemetry is not None:
            msg = str(data.get("message") or data.get("error") or "Unknown server error")
            self._telemetry.emit(
                "stt.error",
                message=msg,
                raw=data,
                ts_ns=self._last_emitted_ts_ns,
            )

    async def _attempt_reconnect(self) -> bool:
        """Attempt WebSocket reconnection with exponential backoff and audio buffer replay."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(
                "STT max reconnect attempts reached (%d); halting stream",
                self._max_reconnect_attempts,
            )
            if self._telemetry is not None:
                self._telemetry.emit(
                    "stt.reconnect_exhausted",
                    attempts=self._reconnect_attempts,
                    ts_ns=self._last_emitted_ts_ns,
                )
            self._last_error = {
                "type": "reconnect_exhausted",
                "attempts": self._reconnect_attempts,
            }
            return False

        self._reconnect_attempts += 1
        delay_ms = min(
            self._reconnect_backoff_max_ms,
            self._reconnect_backoff_initial_ms * (2 ** (self._reconnect_attempts - 1)),
        )
        logger.info(
            "STT reconnecting (attempt %d/%d) after %d ms delay",
            self._reconnect_attempts,
            self._max_reconnect_attempts,
            delay_ms,
        )

        await asyncio.sleep(delay_ms / 1000.0)

        if self._closed or self._cancel.cancelled:
            return False

        if self._ws_connect_fn is None:
            return False

        try:
            new_ws = await self._ws_connect_fn()
            self._ws = new_ws
        except Exception as exc:
            logger.error("STT reconnect socket creation failed: %s", exc)
            return False

        # Replay buffered frames (<= 2 s audio) to restore server VAD/STT context
        replayed = list(self._replay_buffer)
        for frame in replayed:
            try:
                await self._ws.send_bytes(frame.pcm)
            except Exception as exc:
                logger.warning("Failed replaying frame during reconnect: %s", exc)
                break

        if self._telemetry is not None:
            self._telemetry.emit(
                "stt.reconnected",
                attempt=self._reconnect_attempts,
                replayed_frames=len(replayed),
                ts_ns=self._last_emitted_ts_ns,
            )

        return True

    async def events(self) -> AsyncIterator[SttEvent]:
        """Asynchronous iterator yielding partial and final transcription events in order."""
        while not self._closed and not self._cancel.cancelled:
            # For replay sockets (like FakeWebSocket), terminate if open but queue exhausted
            is_ws_closed = getattr(self._ws, "_closed", False) or getattr(
                self._ws, "is_closed", False
            )
            if (
                hasattr(self._ws, "inbound_queue")
                and not self._ws.inbound_queue
                and not is_ws_closed
            ):
                break

            try:
                raw_msg = await self._ws.recv()
            except asyncio.IncompleteReadError:
                if self._closed or self._cancel.cancelled:
                    break
                reconnected = await self._attempt_reconnect()
                if not reconnected:
                    break
                continue
            except Exception:
                if self._closed or self._cancel.cancelled:
                    break
                reconnected = await self._attempt_reconnect()
                if not reconnected:
                    break
                continue

            if self._closed or self._cancel.cancelled:
                break

            if isinstance(raw_msg, bytes):
                try:
                    text_payload = raw_msg.decode("utf-8")
                except UnicodeDecodeError:
                    logger.warning("Received non-UTF8 binary frame from Prisma, ignoring")
                    continue
            else:
                text_payload = raw_msg

            try:
                data = json.loads(text_payload)
            except json.JSONDecodeError:
                logger.warning(
                    "Received malformed JSON from Prisma STT, ignoring: %s",
                    text_payload[:100],
                )
                continue

            if not isinstance(data, dict):
                logger.warning("Prisma payload is not a JSON object: %s", type(data))
                continue

            msg_type = data.get("type")
            if not msg_type:
                if "error" in data:
                    self._handle_error_frame(data)
                else:
                    logger.warning("Prisma frame missing 'type': %s", data)
                continue

            if msg_type == "connected":
                cfg = data.get("config")
                if isinstance(cfg, dict):
                    lang = cfg.get("language_code")
                    if lang:
                        self._current_lang = str(lang)
                continue

            elif msg_type == "speech_start":
                continue

            elif msg_type == "speech_end":
                continue

            elif msg_type == "partial_transcript":
                evt = self._process_partial(data)
                if evt is not None:
                    if self._closed or self._cancel.cancelled:
                        break
                    yield evt

            elif msg_type == "final_transcript":
                evt = self._process_final(data)
                if evt is not None:
                    if self._closed or self._cancel.cancelled:
                        break
                    yield evt

            elif msg_type == "error":
                self._handle_error_frame(data)

            else:
                logger.warning("Unknown Prisma frame type ignored: %s", msg_type)

    async def close(self) -> None:
        """Close the stream session. Must be idempotent."""
        if self._closed:
            return
        self._closed = True
        self._send_event.set()

        if self._send_task is not None and not self._send_task.done():
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass

        try:
            await self._ws.close()
        except Exception:
            pass


class PrismaSttPort(SttPort):
    """Port for establishing speech-to-text streams via Gnani Prisma."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str = PRISMA_STT_WS_URL,
        model: str = "prisma-v3.0",
        language_code: str = "indic-auto",
        format_style: str = "verbatim",
        telemetry: TelemetrySink | None = None,
        clock: Clock | None = None,
        ws_factory: Callable[..., Any] | None = None,
        max_reconnect_attempts: int = 3,
        reconnect_backoff_initial_ms: int = 100,
        reconnect_backoff_max_ms: int = 2000,
        max_queue_frames: int = 100,
        replay_buffer_ms: int = 2000,
    ) -> None:
        self._api_key = api_key or os.environ.get("GNANI_API_KEY", "")
        self._url = url
        self._model = model
        self._language_code = language_code
        self._format_style = format_style
        self._telemetry = telemetry
        self._clock = clock
        self._ws_factory = ws_factory
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_backoff_initial_ms = reconnect_backoff_initial_ms
        self._reconnect_backoff_max_ms = reconnect_backoff_max_ms
        self._max_queue_frames = max_queue_frames
        self._replay_buffer_ms = replay_buffer_ms

    async def open(
        self,
        *,
        lang_hint: str | None = None,
        cancel: CancelToken,
    ) -> PrismaSttStream:
        """Open a new STT stream with optional language hint and cancel token."""
        active_lang = lang_hint or self._language_code
        headers = build_prisma_headers(
            api_key=self._api_key,
            model=self._model,
            language_code=active_lang,
            format_style=self._format_style,
            user_agent=DEFAULT_USER_AGENT,
        )

        async def _connect_ws() -> Any:
            if self._ws_factory is not None:
                try:
                    res = self._ws_factory(url=self._url, headers=headers)
                except TypeError:
                    res = self._ws_factory()
                if inspect.isawaitable(res):
                    return await res
                return res

            if not self._api_key:
                raise GnaniAuthenticationError(
                    "GNANI_API_KEY is required to connect to Prisma STT WebSocket"
                )
            return await AsyncWebSocket.connect(self._url, headers=headers)

        ws = await _connect_ws()

        return PrismaSttStream(
            ws=ws,
            cancel=cancel,
            ws_connect_fn=_connect_ws,
            telemetry=self._telemetry,
            clock=self._clock,
            language_code=active_lang,
            max_reconnect_attempts=self._max_reconnect_attempts,
            reconnect_backoff_initial_ms=self._reconnect_backoff_initial_ms,
            reconnect_backoff_max_ms=self._reconnect_backoff_max_ms,
            max_queue_frames=self._max_queue_frames,
            replay_buffer_ms=self._replay_buffer_ms,
        )
