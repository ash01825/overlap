"""Live full-duplex OVERLAP demo server (`aiohttp` + WebSocket + `OverlapEngine`).

Bridges browser microphone / Web Audio playout (16 kHz int16 mono PCM at 20 ms cadence)
directly to `OverlapEngine`, Gnani Prisma STT, Gnani Timbre TTS, and Evon / Gemini LLM,
with automatic hermetic fallback so `python -m overlap.demo` stays responsive even without
external credentials or network connectivity.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.resources
import inspect
import logging
import math
import os
import re
import struct
import threading
import webbrowser
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from aiohttp import WSMsgType, web

from overlap.adapters.gnani.evon import EvonConfig, EvonLlmAdapter
from overlap.adapters.gnani.prisma import PrismaSttPort, PrismaSttStream
from overlap.adapters.gnani.timbre import TimbreTtsAdapter, TimbreTtsConfig
from overlap.core.audio import EnergyVad, rms_dbfs
from overlap.core.clock import CancelToken, Clock, MonotonicClock
from overlap.core.fsm import FSM
from overlap.core.lexicon import analyze
from overlap.core.types import (
    AgentAudioDrained,
    AudioFrame,
    FsmState,
    Label,
    OverlapFeatures,
    SttEvent,
    Tag,
    Tuning,
    VadOnset,
    Verdict,
)
from overlap.ports.llm import LlmPort, LlmStream, Message
from overlap.ports.stt import SttPort, SttStream
from overlap.ports.telemetry import TelemetrySink
from overlap.ports.transport import AudioTransport
from overlap.ports.tts import AudioChunk, AudioClip, TtsPort, TtsStream
from overlap.ports.vad import VadResult
from overlap.runtime.engine import OverlapEngine
from overlap.runtime.turn import TurnResult

logger = logging.getLogger(__name__)

DEMO_SYSTEM_PROMPT: str = (
    "You are OVERLAP, a calm, natural full-duplex voice assistant built on the Gnani stack. "
    "Keep spoken replies concise (2 to 3 clear sentences), conversational, and direct. "
    "Never use markdown, bullet lists, or asterisks."
)

SUPPORTED_LANGUAGES: dict[str, dict[str, str]] = {
    "auto": {
        "label": "Indic Auto",
        "bcp47_stt": "indic-auto",
        "bcp47_tts": "hi-IN",
        "lexicon": "hi",
        "name": "auto-detected Indian language or English",
    },
    "hi": {
        "label": "Hindi · हिन्दी",
        "bcp47_stt": "indic-auto",
        "bcp47_tts": "hi-IN",
        "lexicon": "hi",
        "name": "Hindi / Hinglish",
    },
    "en": {
        "label": "English · IN",
        "bcp47_stt": "en-IN",
        "bcp47_tts": "en-IN",
        "lexicon": "en",
        "name": "Indian English",
    },
    "ta": {
        "label": "Tamil · தமிழ்",
        "bcp47_stt": "indic-auto",
        "bcp47_tts": "ta-IN",
        "lexicon": "ta",
        "name": "Tamil",
    },
    "te": {
        "label": "Telugu · తెలుగు",
        "bcp47_stt": "indic-auto",
        "bcp47_tts": "te-IN",
        "lexicon": "te",
        "name": "Telugu",
    },
    "kn": {
        "label": "Kannada · ಕನ್ನಡ",
        "bcp47_stt": "indic-auto",
        "bcp47_tts": "kn-IN",
        "lexicon": "kn",
        "name": "Kannada",
    },
    "ml": {
        "label": "Malayalam · മലയാളം",
        "bcp47_stt": "indic-auto",
        "bcp47_tts": "ml-IN",
        "lexicon": "ml",
        "name": "Malayalam",
    },
    "bn": {
        "label": "Bengali · বাংলা",
        "bcp47_stt": "indic-auto",
        "bcp47_tts": "bn-IN",
        "lexicon": "bn",
        "name": "Bengali",
    },
    "mr": {
        "label": "Marathi · मराठी",
        "bcp47_stt": "indic-auto",
        "bcp47_tts": "mr-IN",
        "lexicon": "mr",
        "name": "Marathi",
    },
    "gu": {
        "label": "Gujarati · ગુજરાતી",
        "bcp47_stt": "indic-auto",
        "bcp47_tts": "gu-IN",
        "lexicon": "gu",
        "name": "Gujarati",
    },
    "pa": {
        "label": "Punjabi · ਪੰਜਾਬੀ",
        "bcp47_stt": "indic-auto",
        "bcp47_tts": "pa-IN",
        "lexicon": "pa",
        "name": "Punjabi",
    },
    "or": {
        "label": "Odia · ଓଡ଼ିଆ",
        "bcp47_stt": "indic-auto",
        "bcp47_tts": "or-IN",
        "lexicon": "or",
        "name": "Odia",
    },
}

DemoMode = Literal["overlap", "stock"]


# =============================================================================
# Environment Helpers
# =============================================================================


def load_dotenv_if_present(env_path: Path | None = None) -> dict[str, str]:
    """Load key-value pairs from `.env` into `os.environ` without overwriting existing values."""
    loaded: dict[str, str] = {}
    candidates = (
        [env_path]
        if env_path is not None
        else [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if key.startswith("export "):
                key = key[7:].strip()
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            if key and key not in os.environ:
                os.environ[key] = val
                loaded[key] = val
        break
    return loaded


def _normalize_short_lang(lang: str) -> str:
    raw = (lang or "en").strip().lower()
    if raw in ("auto", "indic-auto", "indic_auto"):
        return "auto"
    base = raw.split("-")[0].split("_")[0]
    return base if base in SUPPORTED_LANGUAGES else "en"


def _map_stt_lang(short_lang: str) -> str:
    return SUPPORTED_LANGUAGES[_normalize_short_lang(short_lang)]["bcp47_stt"]


def _map_tts_lang(short_lang: str) -> str:
    return SUPPORTED_LANGUAGES[_normalize_short_lang(short_lang)]["bcp47_tts"]


def _map_lexicon_lang(short_lang: str) -> str:
    return SUPPORTED_LANGUAGES[_normalize_short_lang(short_lang)]["lexicon"]


def _build_system_prompt(short_lang: str) -> str:
    code = _normalize_short_lang(short_lang)
    if code == "en":
        return (
            f"{DEMO_SYSTEM_PROMPT} CRITICAL INSTRUCTION: Respond STRICTLY in natural English "
            "(Latin script only). Even if earlier turns in the conversation were in Hindi or "
            "another language, or if the user's transcript is transliterated in Devanagari/Indic "
            "script, you MUST reply 100% in clear spoken English."
        )
    if code == "auto":
        return (
            f"{DEMO_SYSTEM_PROMPT} Detect the language of the caller's LATEST message: "
            "if the caller is speaking English (even if transcribed in Indic script), reply "
            "in natural English; if they are speaking Hindi/Hinglish or another Indian language "
            "(Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia), "
            "reply in that same language."
        )
    name = SUPPORTED_LANGUAGES[code]["name"]
    return (
        f"{DEMO_SYSTEM_PROMPT} Respond naturally in spoken {name}. "
        "However, if the caller explicitly asks you to speak in English or asks a full "
        "question in English, reply in the language they used."
    )


def _map_voice_id(voice_id: str) -> str:
    cleaned = (voice_id or "").strip()
    if not cleaned or cleaned == "v1":
        return "Nalini"
    return cleaned


def _pcm_peak_norm(pcm: bytes) -> float:
    """Normalized [0.0, 1.0] perceptual loudness indicator for 16-bit mono PCM."""
    if not pcm:
        return 0.0
    db = rms_dbfs(pcm)
    if db <= -60.0:
        return 0.0
    if db >= -6.0:
        return 1.0
    return max(0.0, min(1.0, (db + 60.0) / 54.0))


def _synthesize_tone_pcm(
    *,
    duration_ms: int,
    sample_rate: int = 16000,
    freq_hz: float = 195.0,
    harmonic_hz: float = 390.0,
    amplitude: int = 5200,
    fade_ms: int = 25,
) -> bytes:
    """Synthesize smooth int16 mono harmonic speech-like PCM without external calls."""
    n_samples = max(1, (sample_rate * duration_ms) // 1000)
    fade_samples = max(1, min(n_samples // 2, (sample_rate * fade_ms) // 1000))
    out = bytearray(n_samples * 2)
    for i in range(n_samples):
        t = i / sample_rate
        # Gentle prosodic tremolo + formant pair for pleasant audible monitoring
        mod = 0.72 + 0.28 * math.sin(2.0 * math.pi * 4.2 * t)
        sig = 0.65 * math.sin(2.0 * math.pi * freq_hz * t) + 0.35 * math.sin(
            2.0 * math.pi * harmonic_hz * t
        )
        env = 1.0
        if i < fade_samples:
            env = i / fade_samples
        elif i >= n_samples - fade_samples:
            env = (n_samples - 1 - i) / fade_samples
        val = int(max(-32767, min(32767, round(amplitude * mod * env * sig))))
        struct.pack_into("<h", out, i * 2, val)
    return bytes(out)


# =============================================================================
# Browser WebSocket Audio Transport
# =============================================================================


class BrowserWsTransport(AudioTransport):
    """Bi-directional 16 kHz int16 PCM bridge between `OverlapEngine` and a WebSocket."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        tuning: Tuning | None = None,
        on_flush: Callable[[], Any] | None = None,
    ) -> None:
        self._clock: Clock = clock if clock is not None else MonotonicClock()
        self._tuning: Tuning = tuning if tuning is not None else Tuning()
        self._on_flush = on_flush

        self._frame_ms: int = self._tuning.frame_ms
        self._sample_rate: int = self._tuning.in_rate_hz
        self._samples_per_frame: int = (self._sample_rate * self._frame_ms) // 1000
        self._bytes_per_frame: int = self._samples_per_frame * 2

        self._in_buf = bytearray()
        self._in_queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue(maxsize=250)
        self._stream_start_ns: int | None = None
        self._emitted_in_frames: int = 0
        self._closed: bool = False

        self._outbound_queue: deque[AudioFrame] = deque(maxlen=200)
        self._outbound_event: asyncio.Event = asyncio.Event()
        self._client_buffered_ms: int = 0
        self._total_emitted_ms: int = 0
        self._last_agent_level: float = 0.0
        self._last_agent_frame_ns: int = 0
        self.flush_count: int = 0

    @property
    def last_agent_level(self) -> float:
        """Most recent normalized [0..1] outbound agent level (decays to 0 when idle)."""
        if self._clock.now_ns() - self._last_agent_frame_ns > 150_000_000:
            return 0.0
        return self._last_agent_level

    def push_mic_pcm(self, pcm_bytes: bytes) -> list[AudioFrame]:
        """Accept raw 16 kHz int16 PCM bytes from browser and slice into canonical 20 ms frames."""
        if self._closed or not pcm_bytes:
            return []
        if self._stream_start_ns is None:
            self._stream_start_ns = self._clock.now_ns()

        self._in_buf.extend(pcm_bytes)
        frames: list[AudioFrame] = []
        frame_dur_ns = self._frame_ms * 1_000_000

        while len(self._in_buf) >= self._bytes_per_frame:
            chunk = bytes(self._in_buf[: self._bytes_per_frame])
            del self._in_buf[: self._bytes_per_frame]
            ts_ns = self._stream_start_ns + (self._emitted_in_frames * frame_dur_ns)
            # Align ahead if paused across long gaps so clock & frame ts stay close
            clock_now = self._clock.now_ns()
            if ts_ns < clock_now - (40 * 1_000_000):
                ts_ns = clock_now
                self._stream_start_ns = clock_now - (self._emitted_in_frames * frame_dur_ns)
            self._emitted_in_frames += 1

            frame = AudioFrame(
                pcm=chunk,
                sample_rate=self._sample_rate,
                ts_ns=ts_ns,
                tags=Tag.NONE,
            )
            if self._in_queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._in_queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self._in_queue.put_nowait(frame)
            frames.append(frame)

        return frames

    def update_client_buffer_ms(self, buffered_ms: int) -> None:
        """Update latest client-reported Web Audio queue depth in milliseconds."""
        self._client_buffered_ms = max(0, buffered_ms)

    async def user_frames(self) -> AsyncIterator[AudioFrame]:
        """Yield canonical 20 ms user audio frames."""
        while not self._closed:
            frame = await self._in_queue.get()
            if frame is None:
                break
            yield frame

    def send_frame(self, frame: AudioFrame) -> None:
        """Synchronously enqueue an outbound 20 ms agent PCM frame."""
        if self._closed or not frame.pcm:
            return
        self._outbound_queue.append(frame)
        self._outbound_event.set()
        dur_ms = int((len(frame.pcm) / max(1, frame.sample_rate * 2)) * 1000)
        self._total_emitted_ms += dur_ms
        self._last_agent_level = _pcm_peak_norm(frame.pcm)
        self._last_agent_frame_ns = self._clock.now_ns()

    def flush_output(self) -> None:
        """Drop all queued outbound frames and signal immediate Web Audio stop."""
        self.flush_count += 1
        self._outbound_queue.clear()
        self._client_buffered_ms = 0
        self._total_emitted_ms = 0
        self._last_agent_level = 0.0
        if self._on_flush is not None:
            self._on_flush()

    def reset_playout_tracking(self) -> None:
        """Reset turn-level cumulative playout counters."""
        self._total_emitted_ms = 0
        self._client_buffered_ms = 0

    def output_queue_ms(self) -> int:
        """Queued outbound PCM duration in ms."""
        queued = len(self._outbound_queue) * self._frame_ms
        return queued + self._client_buffered_ms

    def playout_position_ms(self) -> int | None:
        """Estimated ms already played at client Web Audio output."""
        if self._total_emitted_ms == 0:
            return 0
        return max(0, self._total_emitted_ms - self.output_queue_ms())

    async def pop_outbound_frames(self) -> list[AudioFrame]:
        """Wait for and drain pending outbound PCM frames."""
        if not self._outbound_queue and not self._closed:
            self._outbound_event.clear()
            if not self._outbound_queue:
                await self._outbound_event.wait()
        frames = list(self._outbound_queue)
        self._outbound_queue.clear()
        return frames

    def close(self) -> None:
        """Stop queues and close transport cleanly."""
        self._closed = True
        self._outbound_queue.clear()
        self._outbound_event.set()
        if self._in_queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._in_queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            self._in_queue.put_nowait(None)


# =============================================================================
# Resilient Demo Adapters (Prisma STT, Timbre TTS, Evon/Gemini LLM)
# =============================================================================


class DemoPrismaSttStream(SttStream):
    """Multiplexed STT stream uniting live Prisma WebSocket and injected typed events."""

    def __init__(
        self,
        *,
        prisma_port: PrismaSttPort | None,
        lang: str,
        cancel: CancelToken,
        status_cb: Callable[[str, str], Any] | None = None,
        use_live: bool = True,
        frame_gate: Callable[[AudioFrame], bool] | None = None,
    ) -> None:
        self._prisma_port = prisma_port
        self._lang = _normalize_short_lang(lang)
        self._cancel = cancel
        self._status_cb = status_cb
        self._use_live = use_live
        self._frame_gate = frame_gate

        self._queue: asyncio.Queue[SttEvent | None] = asyncio.Queue()
        self._upstream: PrismaSttStream | None = None
        self._upstream_task: asyncio.Task[None] | None = None
        self._mic_active: bool = False
        self._closed: bool = False
        self._connect_lock: asyncio.Lock = asyncio.Lock()
        self._last_connect_attempt_ns: int = 0

    def inject_event(self, event: SttEvent) -> None:
        """Enqueue an STT event synchronously (used for typed text & deterministic tests)."""
        if not self._closed and not self._cancel.cancelled:
            self._queue.put_nowait(event)

    async def set_mic_active(self, active: bool) -> None:
        """Connect upstream Prisma STT when mic capture starts; detach cleanly when stopped."""
        self._mic_active = active
        if active:
            await self._ensure_upstream()
        else:
            await self._close_upstream()

    async def set_language(self, lang: str) -> None:
        """Switch STT language hint and restart upstream Prisma stream if connected."""
        new_lang = _normalize_short_lang(lang)
        if new_lang == self._lang:
            return
        self._lang = new_lang
        if self._upstream is not None:
            await self._close_upstream()
            if self._mic_active:
                await self._ensure_upstream()

    async def _ensure_upstream(self) -> None:
        if self._closed or self._cancel.cancelled:
            return
        if not self._use_live or self._prisma_port is None:
            if self._status_cb is not None:
                self._status_cb("stt", "local")
            return
        if self._upstream is not None:
            return

        async with self._connect_lock:
            if self._upstream is not None or self._closed or self._cancel.cancelled:
                return
            self._last_connect_attempt_ns = int(asyncio.get_running_loop().time() * 1_000_000_000)
            try:
                stream = await self._prisma_port.open(
                    lang_hint=_map_stt_lang(self._lang),
                    cancel=self._cancel,
                )
            except Exception as exc:
                logger.warning("Prisma STT unavailable, staying on local queue: %s", exc)
                if self._status_cb is not None:
                    self._status_cb("stt", "fallback")
                return

            self._upstream = stream
            if self._status_cb is not None:
                self._status_cb("stt", "prisma")
            self._upstream_task = asyncio.create_task(
                self._pump_upstream(stream),
                name="demo_prisma_pump",
            )

    async def _pump_upstream(self, stream: PrismaSttStream) -> None:
        try:
            async for evt in stream.events():
                if self._closed or self._cancel.cancelled:
                    break
                await self._queue.put(evt)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("Prisma STT stream interrupted: %s", exc)
            if self._status_cb is not None:
                self._status_cb("stt", "fallback")
        finally:
            if self._upstream is stream:
                self._upstream = None

    async def _close_upstream(self) -> None:
        if self._upstream_task is not None and not self._upstream_task.done():
            self._upstream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._upstream_task
        self._upstream_task = None
        if self._upstream is not None:
            with contextlib.suppress(Exception):
                await self._upstream.close()
            self._upstream = None

    async def send_audio(self, frame: AudioFrame) -> None:
        """Forward 20 ms mic PCM frame to upstream Prisma stream when live mic is active."""
        if self._closed or self._cancel.cancelled:
            return
        if self._mic_active and self._upstream is None:
            now_ns = int(asyncio.get_running_loop().time() * 1_000_000_000)
            if now_ns - self._last_connect_attempt_ns >= 500_000_000:
                await self._ensure_upstream()
        if self._mic_active and self._upstream is not None:
            outbound_frame = frame
            if self._frame_gate is not None and not self._frame_gate(frame):
                outbound_frame = AudioFrame(
                    pcm=b"\x00" * len(frame.pcm),
                    sample_rate=frame.sample_rate,
                    ts_ns=frame.ts_ns,
                    tags=frame.tags,
                )
            with contextlib.suppress(Exception):
                await self._upstream.send_audio(outbound_frame)

    async def events(self) -> AsyncIterator[SttEvent]:
        """Yield live Prisma transcripts and typed/injected STT events."""
        while not self._closed and not self._cancel.cancelled:
            evt = await self._queue.get()
            if evt is None:
                break
            yield evt

    async def close(self) -> None:
        """Close stream and terminate any upstream Prisma connection."""
        if self._closed:
            return
        self._closed = True
        await self._close_upstream()
        self._queue.put_nowait(None)


class DemoPrismaSttPort(SttPort):
    """`SttPort` wrapper exposing latest active stream for UI language/mic controls."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        use_live: bool = True,
        status_cb: Callable[[str, str], Any] | None = None,
        telemetry: TelemetrySink | None = None,
        clock: Clock | None = None,
        frame_gate: Callable[[AudioFrame], bool] | None = None,
    ) -> None:
        resolved_key = (api_key or os.environ.get("GNANI_API_KEY", "")).strip()
        self._use_live = use_live and bool(resolved_key)
        self._status_cb = status_cb
        self.frame_gate = frame_gate
        self._prisma_port: PrismaSttPort | None = (
            PrismaSttPort(
                api_key=resolved_key,
                language_code="indic-auto",
                telemetry=telemetry,
                clock=clock,
            )
            if self._use_live
            else None
        )
        self.active_stream: DemoPrismaSttStream | None = None

    @property
    def is_live_configured(self) -> bool:
        """True if Gnani Prisma live credentials are enabled."""
        return self._use_live

    async def open(self, *, lang_hint: str | None = None, cancel: CancelToken) -> SttStream:
        """Open a multiplexed demo STT stream."""
        stream = DemoPrismaSttStream(
            prisma_port=self._prisma_port,
            lang=lang_hint or "hi",
            cancel=cancel,
            status_cb=self._status_cb,
            use_live=self._use_live,
            frame_gate=self.frame_gate,
        )
        self.active_stream = stream
        return stream


class DemoTimbreTtsStream(TtsStream):
    """Multi-clause wrapper over `TimbreTtsAdapter` with fast local speech fallback."""

    def __init__(
        self,
        *,
        adapter: TimbreTtsAdapter | None,
        voice_id: str,
        lang: str,
        epoch: int,
        cancel: CancelToken,
        use_live: bool,
        status_cb: Callable[[str, str], Any] | None = None,
        on_clause: Callable[[str], None] | None = None,
    ) -> None:
        self._adapter = adapter
        self._voice_id = _map_voice_id(voice_id)
        self._lang = _map_tts_lang(lang)
        self._epoch = epoch
        self._cancel = cancel
        self._use_live = use_live
        self._status_cb = status_cb
        self._on_clause = on_clause

        self._clause_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._cancelled: bool = False
        self._active_upstream: TtsStream | None = None

    async def send_text(self, clause: str, *, ssml: bool = False) -> None:
        """Queue a clause for synthesis."""
        del ssml
        cleaned = clause.strip()
        if not self._cancelled and not self._cancel.cancelled and cleaned:
            if self._on_clause is not None:
                self._on_clause(cleaned)
            await self._clause_queue.put(cleaned)

    async def finish(self) -> None:
        """Signal no further clauses will be sent for this turn."""
        await self._clause_queue.put(None)

    async def cancel(self) -> None:
        """Cancel active upstream stream and clear queued clauses immediately."""
        self._cancelled = True
        upstream = self._active_upstream
        if upstream is not None:
            with contextlib.suppress(Exception):
                await upstream.cancel()
        with contextlib.suppress(asyncio.QueueFull):
            self._clause_queue.put_nowait(None)

    async def _synthesize_clause_fallback(self, clause: str) -> AsyncIterator[AudioChunk]:
        # Duration scales proportionally with character count (~65 ms per char, clamped)
        dur_ms = max(900, min(3200, len(clause) * 65))
        pcm = _synthesize_tone_pcm(
            duration_ms=dur_ms,
            sample_rate=16000,
            freq_hz=210.0,
            harmonic_hz=420.0,
            amplitude=4600,
            fade_ms=30,
        )
        # Emit in 100 ms chunks so cancellation stops delivery immediately
        chunk_bytes = 3200  # 100 ms @ 16 kHz int16
        total = len(pcm)
        offset = 0
        while offset < total and not self._cancelled and not self._cancel.cancelled:
            end = min(total, offset + chunk_bytes)
            yield AudioChunk(
                pcm=pcm[offset:end],
                sample_rate=16000,
                epoch=self._epoch,
                words=(),
                is_last=False,
            )
            offset = end
            await asyncio.sleep(0)

    def _clause_tts_lang(self, clause: str) -> str:
        """Select Timbre language matching the actual clause script."""
        if self._lang == "en-IN" and any("\u0900" <= ch <= "\u097f" for ch in clause):
            return "hi-IN"
        alpha_chars = [ch for ch in clause if ch.isalpha()]
        if self._lang == "hi-IN" and alpha_chars and all(ord(ch) < 128 for ch in alpha_chars):
            return "en-IN"
        return self._lang

    async def chunks(self) -> AsyncIterator[AudioChunk]:
        """Stream PCM audio chunks across all clauses sent by `TurnPipeline`."""
        while not self._cancelled and not self._cancel.cancelled:
            clause = await self._clause_queue.get()
            if clause is None or self._cancelled or self._cancel.cancelled:
                break

            tts_lang = self._clause_tts_lang(clause)
            candidate_langs = [tts_lang, "hi-IN"] if tts_lang == "mr-IN" else [tts_lang]
            yielded_for_clause = False
            if self._use_live and self._adapter is not None:
                for cand_lang in candidate_langs:
                    if yielded_for_clause or self._cancelled or self._cancel.cancelled:
                        break
                    try:
                        stream = await self._adapter.open_stream(
                            voice_id=self._voice_id,
                            lang=cand_lang,
                            epoch=self._epoch,
                            cancel=self._cancel,
                        )
                        self._active_upstream = stream
                        await stream.send_text(clause)
                        await stream.finish()
                        async for chunk in stream.chunks():
                            if self._cancelled or self._cancel.cancelled:
                                break
                            if chunk.pcm:
                                yielded_for_clause = True
                                yield AudioChunk(
                                    pcm=chunk.pcm,
                                    sample_rate=chunk.sample_rate,
                                    epoch=self._epoch,
                                    words=chunk.words,
                                    is_last=False,
                                )
                        if self._status_cb is not None and yielded_for_clause:
                            self._status_cb("tts", "timbre")
                    except Exception as exc:
                        logger.warning(
                            "Timbre TTS clause failed (%s), falling back: %s",
                            cand_lang,
                            exc,
                        )
                    finally:
                        self._active_upstream = None

                if (
                    not yielded_for_clause
                    and not self._cancelled
                    and not self._cancel.cancelled
                    and self._status_cb is not None
                ):
                    self._status_cb("tts", "fallback")

            if not yielded_for_clause and not self._cancelled and not self._cancel.cancelled:
                async for fb_chunk in self._synthesize_clause_fallback(clause):
                    yield fb_chunk

        if not self._cancelled and not self._cancel.cancelled:
            yield AudioChunk(
                pcm=b"",
                sample_rate=16000,
                epoch=self._epoch,
                words=(),
                is_last=True,
            )


class DemoTimbreTtsPort(TtsPort):
    """`TtsPort` using real Gnani Timbre streaming + instant local clip synthesis."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        use_live: bool = True,
        status_cb: Callable[[str, str], Any] | None = None,
        telemetry: TelemetrySink | None = None,
        on_clause: Callable[[str], None] | None = None,
    ) -> None:
        resolved_key = (api_key or os.environ.get("GNANI_API_KEY", "")).strip()
        self._use_live = use_live and bool(resolved_key)
        self._status_cb = status_cb
        self.on_clause = on_clause
        self._adapter: TimbreTtsAdapter | None = (
            TimbreTtsAdapter(
                TimbreTtsConfig(api_key=resolved_key, sample_rate=16000),
                telemetry=telemetry,
            )
            if self._use_live
            else None
        )

    async def open_stream(
        self,
        *,
        voice_id: str,
        lang: str,
        epoch: int,
        cancel: CancelToken,
    ) -> TtsStream:
        """Open a multi-clause streaming TTS session."""
        return DemoTimbreTtsStream(
            adapter=self._adapter,
            voice_id=voice_id,
            lang=lang,
            epoch=epoch,
            cancel=cancel,
            use_live=self._use_live,
            status_cb=self._status_cb,
            on_clause=self.on_clause,
        )

    async def synthesize(self, text: str, *, voice_id: str, lang: str) -> AudioClip:
        """Produce a short, non-blocking local backchannel confirmation clip for `ClipCache`."""
        del text, voice_id, lang
        pcm = _synthesize_tone_pcm(
            duration_ms=140,
            sample_rate=16000,
            freq_hz=230.0,
            harmonic_hz=345.0,
            amplitude=2400,
            fade_ms=35,
        )
        return AudioClip(pcm=pcm, sample_rate=16000)


class _LocalLlmStream(LlmStream):
    """Streaming fallback LLM responder when live API keys or network are unavailable."""

    def __init__(self, text: str, *, cancel: CancelToken) -> None:
        self._text = text
        self._cancel = cancel
        self._aborted = False

    async def deltas(self) -> AsyncIterator[str]:
        words = self._text.split()
        for idx, word in enumerate(words):
            if self._aborted or self._cancel.cancelled:
                break
            yield ("" if idx == 0 else " ") + word
            await asyncio.sleep(0)

    async def abort(self) -> None:
        self._aborted = True


class DemoEvonLlmPort(LlmPort):
    """`LlmPort` wrapping `EvonLlmAdapter` (Gemini) with a crisp full-duplex system prompt."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        lang: str = "en",
        use_live: bool = True,
        status_cb: Callable[[str, str], Any] | None = None,
        telemetry: TelemetrySink | None = None,
        clock: Clock | None = None,
    ) -> None:
        resolved_key = (
            api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GNANI_API_KEY", "")
        ).strip()
        self._lang = _normalize_short_lang(lang)
        self._use_live = use_live and bool(resolved_key)
        self._status_cb = status_cb
        self._adapter: EvonLlmAdapter | None = (
            EvonLlmAdapter(
                config=EvonConfig(
                    api_key=resolved_key,
                    connect_timeout_s=8.0,
                    first_token_timeout_s=10.0,
                    token_timeout_s=10.0,
                ),
                clock=clock,
                telemetry=telemetry,
            )
            if self._use_live
            else None
        )

    def set_language(self, lang: str) -> None:
        """Update target response language for subsequent turns."""
        self._lang = _normalize_short_lang(lang)

    def _local_fallback_reply(self, messages: Sequence[Message]) -> str:
        if self._lang == "en":
            return (
                "Got it, I am running in Indian English full-duplex mode now. "
                "While I explain this, brief acknowledgements softly duck me "
                "and let me finish, whereas saying wait or asking a question stops me cleanly."
            )
        last_user = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user = msg.content.strip()
                break
        words = set(re.findall(r"[a-z]+", last_user.lower()))
        is_hindi = (
            self._lang == "hi"
            or bool(words & {"kya", "hai", "batao", "hindi", "namaste", "haan", "theek"})
            or (self._lang == "auto" and any("\u0900" <= ch <= "\u097f" for ch in last_user))
        )
        if is_hindi:
            return (
                "Ji bilkul, main abhi full-duplex mode mein sun raha hoon. "
                "Jab main bol raha hoon tab aap haan ya achha bol sakte hain "
                "aur main nahi rukunga. "
                "Lekin agar aap rukiye ya koi naya sawal poochenge toh main turant ruk jaunga."
            )
        lang_name = SUPPORTED_LANGUAGES.get(self._lang, {}).get("name", "full-duplex")
        return (
            f"Got it, I am running in {lang_name} full-duplex mode now. "
            "While I explain this, brief acknowledgements softly duck me "
            "and let me finish, whereas saying wait or asking a question stops me cleanly."
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        epoch: int,
        cancel: CancelToken,
    ) -> LlmStream:
        """Stream LLM response deltas, prepending system persona once."""
        prompt_messages = list(messages)
        sys_prompt = _build_system_prompt(self._lang)
        if not prompt_messages or prompt_messages[0].role != "system":
            prompt_messages.insert(0, Message(role="system", content=sys_prompt))
        else:
            prompt_messages[0] = Message(role="system", content=sys_prompt)

        for idx in range(len(prompt_messages) - 1, -1, -1):
            if prompt_messages[idx].role == "user":
                if self._lang == "en":
                    directive = "\n[Reply strictly in English.]"
                elif self._lang == "auto":
                    directive = "\n[Reply in the same language as this message.]"
                else:
                    lang_name = SUPPORTED_LANGUAGES[self._lang]["name"]
                    directive = f"\n[Reply in {lang_name}.]"
                prompt_messages[idx] = Message(
                    role="user",
                    content=prompt_messages[idx].content + directive,
                )
                break

        if self._use_live and self._adapter is not None:
            try:
                live_stream = await self._adapter.stream(
                    prompt_messages,
                    epoch=epoch,
                    cancel=cancel,
                )

                class _ResilientStream(LlmStream):
                    def __init__(
                        self,
                        inner: LlmStream,
                        fallback_text: str,
                        status_cb: Callable[[str, str], Any] | None,
                    ) -> None:
                        self._inner = inner
                        self._fallback_text = fallback_text
                        self._status_cb = status_cb

                    async def deltas(self) -> AsyncIterator[str]:
                        emitted = False
                        try:
                            async for delta in self._inner.deltas():
                                emitted = True
                                yield delta
                            if emitted and self._status_cb is not None:
                                self._status_cb("llm", "evon")
                        except Exception as exc:
                            logger.warning("Live LLM stream failed: %s", exc)
                            if self._status_cb is not None:
                                self._status_cb("llm", "fallback")
                            if not emitted:
                                fb = _LocalLlmStream(self._fallback_text, cancel=cancel)
                                async for delta in fb.deltas():
                                    yield delta

                    async def abort(self) -> None:
                        await self._inner.abort()

                return _ResilientStream(
                    live_stream,
                    self._local_fallback_reply(prompt_messages),
                    self._status_cb,
                )
            except Exception as exc:
                logger.warning("Failed opening live LLM stream: %s", exc)
                if self._status_cb is not None:
                    self._status_cb("llm", "fallback")

        if self._status_cb is not None:
            self._status_cb("llm", "local")
        return _LocalLlmStream(self._local_fallback_reply(prompt_messages), cancel=cancel)


# =============================================================================
# Telemetry Sink & Session Coordinator
# =============================================================================


class _DemoTelemetrySink(TelemetrySink):
    """Synchronous non-blocking telemetry forwarder into `DemoSession`."""

    def __init__(self, on_event: Callable[[str, int, dict[str, Any]], None]) -> None:
        self._on_event = on_event

    def emit(self, name: str, ts_ns: int, **fields: Any) -> None:
        self._on_event(name, ts_ns, dict(fields))


class DemoSession:
    """Single live full-duplex demo session orchestrating `OverlapEngine` and WS updates."""

    def __init__(
        self,
        *,
        send_json: Callable[[dict[str, Any]], Any] | None = None,
        send_bytes: Callable[[bytes], Any] | None = None,
        lang: str = "en",
        voice_id: str = "Nalini",
        mode: DemoMode = "overlap",
        clock: Clock | None = None,
        use_live_apis: bool = True,
        tuning: Tuning | None = None,
        stt_port: DemoPrismaSttPort | None = None,
        tts_port: TtsPort | None = None,
        llm_port: LlmPort | None = None,
    ) -> None:
        self._send_json_cb = send_json
        self._send_bytes_cb = send_bytes
        self._lang: str = _normalize_short_lang(lang)
        self._voice_id: str = _map_voice_id(voice_id)
        self._mode: DemoMode = mode
        self._clock: Clock = clock if clock is not None else MonotonicClock()
        self._tuning: Tuning = tuning if tuning is not None else Tuning()
        self._use_live_apis = use_live_apis
        self._session_start_ns: int = self._clock.now_ns()

        has_gnani = bool(os.environ.get("GNANI_API_KEY"))
        has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
        self._status: dict[str, str] = {
            "stt": "prisma" if (use_live_apis and has_gnani) else "local",
            "llm": "evon" if (use_live_apis and (has_gemini or has_gnani)) else "local",
            "tts": "timbre" if (use_live_apis and has_gnani) else "local",
        }

        self._telemetry = _DemoTelemetrySink(self._on_telemetry)
        self.transport = BrowserWsTransport(
            clock=self._clock,
            tuning=self._tuning,
            on_flush=self._on_transport_flush,
        )
        self._vad = EnergyVad(
            speech_threshold_db=15.0,
            min_noise_floor_db=-45.0,
            initial_noise_floor_db=-42.0,
            hangover_frames=3,
        )
        self._orig_vad_process = self._vad.process
        self._vad.process = self._guarded_vad_process  # type: ignore[method-assign]

        self.stt_port: DemoPrismaSttPort = (
            stt_port
            if stt_port is not None
            else DemoPrismaSttPort(
                use_live=use_live_apis,
                status_cb=self._on_adapter_status,
                telemetry=self._telemetry,
                clock=self._clock,
                frame_gate=self._should_forward_stt_frame,
            )
        )
        self.tts_port: TtsPort = (
            tts_port
            if tts_port is not None
            else DemoTimbreTtsPort(
                use_live=use_live_apis,
                status_cb=self._on_adapter_status,
                telemetry=self._telemetry,
                on_clause=self._on_agent_clause,
            )
        )
        self.llm_port: LlmPort = (
            llm_port
            if llm_port is not None
            else DemoEvonLlmPort(
                lang=self._lang,
                use_live=use_live_apis,
                status_cb=self._on_adapter_status,
                telemetry=self._telemetry,
                clock=self._clock,
            )
        )

        lex_lang = _map_lexicon_lang(self._lang)
        self.engine = OverlapEngine(
            transport=self.transport,
            vad=self._vad,
            stt=self.stt_port,
            tts=self.tts_port,
            llm=self.llm_port,
            clock=self._clock,
            telemetry=self._telemetry,
            tuning=self._tuning,
            voice_id=self._voice_id,
            lang=lex_lang,
        )

        # Track live turn lifecycle & UI state
        self._prev_fsm_state: FsmState = self.engine.state
        self._pending_user_text: str = ""
        self._active_turn_task: asyncio.Task[TurnResult] | None = None
        self._tts_streaming_clauses: bool = False
        self._mic_active: bool = False
        self._last_mic_rx_ns: int = 0
        self._simulating_user_text: bool = False
        self._latest_user_level: float = 0.0
        self._latest_verdict_info: dict[str, Any] | None = None
        self._turn_started_ns: int = 0
        self._last_user_speech_ns: int = 0
        self._vad_onset_streak: int = 0
        self._last_user_turn_text: str = ""
        self._current_agent_text: str = ""

        self._ws_outbound_queue: asyncio.Queue[tuple[Literal["json", "bytes"], Any]] = (
            asyncio.Queue(maxsize=512)
        )
        self._tx_task: asyncio.Task[None] | None = None
        self._audio_pump_task: asyncio.Task[None] | None = None
        self._meter_task: asyncio.Task[None] | None = None
        self._idle_pump_task: asyncio.Task[None] | None = None
        self._stopped: bool = False

        # Expose turn launch & chunk gating cleanly without modifying engine.py
        self._orig_classifier_fn = self.engine._classifier_fn
        self.engine._classifier_fn = self._guarded_classify

        self._orig_on_fsm_changed = self.engine._on_fsm_state_changed
        self.engine._on_fsm_state_changed = self._wrapped_on_fsm_state_changed  # type: ignore[method-assign]

        self._orig_handle_tts_chunk = self.engine._handle_tts_audio_chunk
        self.engine._handle_tts_audio_chunk = self._wrapped_handle_tts_chunk  # type: ignore[method-assign]
        self.engine.turn_pipeline._audio_chunk_handler = self._wrapped_handle_tts_chunk

        self._orig_handle_stt = self.engine.handle_stt_event
        self.engine.handle_stt_event = self._wrapped_handle_stt_event  # type: ignore[method-assign]

    @property
    def mode(self) -> DemoMode:
        """Current overlap strategy (`overlap` or `stock`)."""
        return self._mode

    @property
    def lang(self) -> str:
        """Active session language code (across all supported Indic languages + auto/en)."""
        return self._lang

    async def start(self) -> None:
        """Initialize engine workers and WebSocket broadcast pumps."""
        self._stopped = False
        self._session_start_ns = self._clock.now_ns()
        await self.engine.start()

        if self._send_json_cb is not None or self._send_bytes_cb is not None:
            self._tx_task = asyncio.create_task(self._ws_tx_worker(), name="demo_ws_tx")
        self._audio_pump_task = asyncio.create_task(self._agent_audio_pump(), name="demo_audio_tx")
        if not hasattr(self._clock, "advance"):
            self._meter_task = asyncio.create_task(self._meter_loop(), name="demo_meter")
            self._idle_pump_task = asyncio.create_task(
                self._idle_frame_pump(), name="demo_idle_pump"
            )
        self._emit_snapshot()

    async def stop(self) -> None:
        """Shut down active turns, background tasks, and engine cleanly."""
        self._stopped = True
        self.transport.close()

        for task in (self._idle_pump_task, self._meter_task, self._audio_pump_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

        if self._active_turn_task is not None and not self._active_turn_task.done():
            self._active_turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._active_turn_task

        await self.engine.stop()

        if self._tx_task is not None and not self._tx_task.done():
            self._tx_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._tx_task

    # -------------------------------------------------------------------------
    # UI Control Actions
    # -------------------------------------------------------------------------

    def set_mode(self, mode: DemoMode) -> None:
        """Switch between `overlap` (duck-then-decide) and `stock` (naive VAD stop)."""
        self._mode = "stock" if mode == "stock" else "overlap"
        self._enqueue_json(
            {
                "type": "telemetry",
                "name": "demo.mode_changed",
                "ts_ms": self._rel_ms(self._clock.now_ns()),
                "fields": {"mode": self._mode},
            }
        )
        self._emit_state()

    async def set_language(self, lang: str) -> None:
        """Switch active conversation language across STT, TTS, LLM, and lexicon."""
        prev_lang = self._lang
        new_lang = _normalize_short_lang(lang)
        lex_lang = _map_lexicon_lang(new_lang)
        self._lang = new_lang
        self.engine._lang = lex_lang
        self.engine.turn_pipeline._lang = lex_lang
        self.engine.predictor._default_lang = lex_lang
        if new_lang != prev_lang:
            self.engine.turn_pipeline._history.clear()
        set_llm_lang = getattr(self.llm_port, "set_language", None)
        if callable(set_llm_lang):
            set_llm_lang(new_lang)
        await self.engine.clip_cache.prepare([lex_lang])
        if self.stt_port.active_stream is not None:
            await self.stt_port.active_stream.set_language(new_lang)
        self._emit_snapshot()

    async def set_mic_active(self, active: bool) -> None:
        """Notify STT adapter whether live browser microphone streaming is active."""
        self._mic_active = bool(active)
        if self.stt_port.active_stream is not None:
            await self.stt_port.active_stream.set_mic_active(self._mic_active)
        if not self._mic_active:
            self._latest_user_level = 0.0
        self._emit_snapshot()

    async def reset(self) -> None:
        """Cancel active speech, reset conversation history, and return cleanly to LISTENING."""
        self.engine.supervisor.cancel_turn()
        self.engine.mixer.flush()
        self.transport.flush_output()
        if self._active_turn_task is not None and not self._active_turn_task.done():
            self._active_turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._active_turn_task
        self._active_turn_task = None
        self.engine.turn_pipeline._history.clear()
        self.engine.turn_pipeline._cancelled_user_text = ""
        self.engine._tts_finished = False
        self.engine._clear_overlap_partial_state()
        self.engine._fsm = FSM(
            tuning=self._tuning,
            initial_state=FsmState.LISTENING,
            initial_epoch=self.engine.fsm.epoch,
        )
        self._prev_fsm_state = FsmState.LISTENING
        self._latest_verdict_info = None
        self._pending_user_text = ""
        self._turn_started_ns = 0
        self._last_user_speech_ns = 0
        self._vad_onset_streak = 0
        self._last_user_turn_text = ""
        self._current_agent_text = ""
        self._enqueue_json({"type": "reset"})
        self._emit_snapshot()

    def handle_incoming_pcm(self, pcm_bytes: bytes) -> None:
        """Ingest raw binary 16 kHz int16 mono PCM bytes from browser WebSocket."""
        if self._stopped or not pcm_bytes:
            return
        self._last_mic_rx_ns = self._clock.now_ns()
        frames = self.transport.push_mic_pcm(pcm_bytes)
        if frames:
            self._latest_user_level = _pcm_peak_norm(frames[-1].pcm)

    async def handle_user_text(self, text: str) -> None:
        """Handle a typed user utterance or mid-speech overlap cleanly through `OverlapEngine`."""
        cleaned = text.strip()
        if not cleaned or self._stopped:
            return

        state = self.engine.state
        if state in (FsmState.LISTENING, FsmState.BACKCHANNELING):
            now_ns = self._clock.now_ns()
            evt = SttEvent(
                kind="final",
                text=cleaned,
                lang=_map_stt_lang(self._lang),
                conf=0.98,
                ts_ns=now_ns,
            )
            self._inject_or_handle_stt(evt)
            return

        if state == FsmState.THINKING:
            # Wait briefly (up to 400 ms) for first TTS chunk if in-flight
            for _ in range(20):
                if self.engine.state != FsmState.THINKING:
                    break
                await self._step_ms(20)
            state = self.engine.state

        if state in (FsmState.SPEAKING, FsmState.DUCKED):
            await self._simulate_spoken_overlap(cleaned)
        else:
            now_ns = self._clock.now_ns()
            evt = SttEvent(
                kind="final",
                text=cleaned,
                lang=_map_stt_lang(self._lang),
                conf=0.98,
                ts_ns=now_ns,
            )
            self._inject_or_handle_stt(evt)

    async def _simulate_spoken_overlap(self, text: str) -> None:
        """Feed real voiced PCM frames + STT partial/final through `OverlapEngine` fast path."""
        self._simulating_user_text = True
        try:
            agent_rms = self.engine.loudness_tracker._agent_rms
            target_amp = (
                max(4500, min(14000, int(agent_rms * 32767.0 * 1.8))) if agent_rms > 0.01 else 7500
            )
            voiced_pcm = _synthesize_tone_pcm(
                duration_ms=self._tuning.frame_ms,
                sample_rate=16000,
                freq_hz=175.0,
                harmonic_hz=350.0,
                amplitude=target_amp,
                fade_ms=2,
            )
            silence_pcm = b"\x00" * (16000 * 2 * self._tuning.frame_ms // 1000)

            self._latest_user_level = _pcm_peak_norm(voiced_pcm)
            frame0 = AudioFrame(
                pcm=voiced_pcm,
                sample_rate=16000,
                ts_ns=self._clock.now_ns(),
                tags=Tag.NONE,
            )
            self.engine.handle_frame(frame0)

            # In STOCK mode, VadOnset already triggers immediate cut & yield
            if self._mode == "stock" or self.engine.state == FsmState.YIELDED:
                for _ in range(10):
                    await self._step_ms(20)
                    self.engine.handle_frame(
                        AudioFrame(pcm=silence_pcm, sample_rate=16000, ts_ns=self._clock.now_ns())
                    )
                    if not self.engine._user_speaking:
                        break
                self._latest_user_level = 0.0
                if self.engine.state in (FsmState.LISTENING, FsmState.BACKCHANNELING):
                    self._emit_interrupt_or_followup(text)
                return

            # Stream voiced frames while delivering early partial transcript at ~40 ms
            partial_sent = False
            verdict_before = self.engine.last_verdict
            for step_idx in range(1, 15):
                await self._step_ms(20)
                if self.engine.state not in (FsmState.DUCKED, FsmState.SPEAKING):
                    break
                self.engine.handle_frame(
                    AudioFrame(pcm=voiced_pcm, sample_rate=16000, ts_ns=self._clock.now_ns())
                )
                if not partial_sent and step_idx >= 2:
                    partial_sent = True
                    self._inject_or_handle_stt(
                        SttEvent(
                            kind="partial",
                            text=text,
                            lang=_map_stt_lang(self._lang),
                            conf=0.92,
                            ts_ns=self._clock.now_ns(),
                        )
                    )
                    # Let the STT receive worker consume the injected partial immediately.
                    for _ in range(3):
                        await asyncio.sleep(0)
                if self.engine.last_verdict is not verdict_before:
                    # Decision reached; trail out smoothly
                    break

            # Allow EnergyVad hangover (8 frames = 160 ms) to settle cleanly to VadOffset
            self._latest_user_level = 0.0
            for _ in range(10):
                await self._step_ms(20)
                self.engine.handle_frame(
                    AudioFrame(pcm=silence_pcm, sample_rate=16000, ts_ns=self._clock.now_ns())
                )
                if not self.engine._user_speaking:
                    break

            verdict_after = self.engine.last_verdict
            if verdict_after is not verdict_before and verdict_after is not None:
                if verdict_after.label == Label.BACKCHANNEL:
                    self._enqueue_json(
                        {
                            "type": "turn",
                            "role": "user",
                            "text": text,
                            "interrupted": False,
                            "backchannel": True,
                        }
                    )
                    return

                if verdict_after.label == Label.INTERRUPT:
                    # Wait briefly if still draining yield sequence
                    for _ in range(5):
                        if self.engine.state == FsmState.LISTENING:
                            break
                        await self._step_ms(20)
                    self._emit_interrupt_or_followup(text)
                    return

            # Fallback if final settles outcome
            self._inject_or_handle_stt(
                SttEvent(
                    kind="final",
                    text=text,
                    lang=_map_stt_lang(self._lang),
                    conf=0.98,
                    ts_ns=self._clock.now_ns(),
                )
            )
        finally:
            self._latest_user_level = 0.0
            self._simulating_user_text = False

    def _emit_interrupt_or_followup(self, text: str) -> None:
        """Send final utterance as new user turn if substantive, else log bare barge-in."""
        analysis = analyze(text, _map_lexicon_lang(self._lang))
        if analysis.content_tokens or analysis.question:
            self._inject_or_handle_stt(
                SttEvent(
                    kind="final",
                    text=text,
                    lang=_map_stt_lang(self._lang),
                    conf=0.98,
                    ts_ns=self._clock.now_ns(),
                )
            )
        else:
            self._enqueue_json(
                {
                    "type": "turn",
                    "role": "user",
                    "text": text,
                    "interrupted": True,
                    "backchannel": False,
                }
            )

    def _inject_or_handle_stt(self, event: SttEvent) -> None:
        if self.stt_port.active_stream is not None:
            self.stt_port.active_stream.inject_event(event)
            # If no running loop tick drained it synchronously (in unit tests using VirtualClock),
            # direct delivery happens cleanly in asyncio next tick.
        else:
            self.engine.handle_stt_event(event)

    async def _step_ms(self, ms: int) -> None:
        advance_fn = getattr(self._clock, "advance", None)
        if callable(advance_fn):
            # Yield before and after advancing virtual time so injected STT events
            # and supervisor cancellation tasks run coherently between timer steps.
            for _ in range(3):
                await asyncio.sleep(0)
            res = advance_fn(ms)
            if inspect.isawaitable(res):
                await res
            for _ in range(3):
                await asyncio.sleep(0)
            return
        await asyncio.sleep(ms / 1000.0)

    # -------------------------------------------------------------------------
    # Engine Interception Hooks & Turn Runner
    # -------------------------------------------------------------------------

    @staticmethod
    def _unicode_words(text: str) -> list[str]:
        """Split text into lowercase words without breaking Devanagari combining marks."""
        words: list[str] = []
        for raw in re.split(r"[\s,.;:!?।|\"'()\[\]{}\-/\\]+", text.strip().lower()):
            cleaned = raw.strip()
            if len(cleaned) >= 2:
                words.append(cleaned)
        return words

    def _on_agent_clause(self, clause: str) -> None:
        self._current_agent_text = f"{self._current_agent_text} {clause}".strip()

    def _should_forward_stt_frame(self, frame: AudioFrame) -> bool:
        if (
            self.engine.state in (FsmState.THINKING, FsmState.SPEAKING, FsmState.DUCKED)
            and not self._simulating_user_text
        ):
            now_ns = self._clock.now_ns()
            recent_speech = self._last_user_speech_ns > 0 and (
                (now_ns - self._last_user_speech_ns) < 800_000_000
            )
            return bool(self.engine._user_speaking or recent_speech or rms_dbfs(frame.pcm) >= -29.0)
        return True

    def _guarded_vad_process(self, frame: AudioFrame) -> VadResult:
        res = cast(VadResult, self._orig_vad_process(frame))
        if self._mic_active and not self._simulating_user_text:
            if self.engine.state == FsmState.THINKING:
                self._vad_onset_streak = 0
                return VadResult(is_speech=False, confidence=0.0, energy_db=res.energy_db)
            if (
                self.engine.state in (FsmState.SPEAKING, FsmState.DUCKED)
                or self.transport.last_agent_level > 0.02
            ):
                if not res.is_speech or res.energy_db is None or res.energy_db < -20.5:
                    self._vad_onset_streak = 0
                    return VadResult(is_speech=False, confidence=0.0, energy_db=res.energy_db)
                if not self.engine._user_speaking:
                    self._vad_onset_streak += 1
                    if self._vad_onset_streak < 3:
                        return VadResult(is_speech=False, confidence=0.0, energy_db=res.energy_db)
            else:
                if not res.is_speech or res.energy_db is None or res.energy_db < -24.5:
                    self._vad_onset_streak = 0
                    return VadResult(is_speech=False, confidence=0.0, energy_db=res.energy_db)
            self._last_user_speech_ns = self._clock.now_ns()
        return res

    def _guarded_classify(
        self,
        features: OverlapFeatures,
        tuning: Tuning | None = None,
        *,
        decided_at_ns: int = 0,
    ) -> Verdict:
        verdict = self._orig_classifier_fn(features, tuning, decided_at_ns=decided_at_ns)
        if (
            self._mode == "overlap"
            and self._mic_active
            and not self._simulating_user_text
            and verdict.label == Label.INTERRUPT
            and (features.partial_text is None or not features.partial_text.strip())
        ):
            return Verdict(
                label=Label.BACKCHANNEL,
                confidence=0.7,
                reason="untranscribed_noise_guard",
                overlap_ms=features.overlap_ms,
                decided_at_ns=verdict.decided_at_ns,
            )
        return verdict

    def _is_echo_or_stale_stt(self, cleaned: str) -> bool:
        if self._simulating_user_text:
            return False
        if self.engine.state not in (FsmState.THINKING, FsmState.SPEAKING, FsmState.DUCKED):
            return False
        now_ns = self._clock.now_ns()
        if (
            not self.engine._user_speaking
            and self._turn_started_ns > 0
            and (now_ns - self._turn_started_ns) < 600_000_000
        ):
            return True
        stt_words = self._unicode_words(cleaned)
        if not stt_words:
            return False
        if self._last_user_turn_text:
            last_words = self._unicode_words(self._last_user_turn_text)
            if (
                last_words
                and stt_words == last_words
                and (self._turn_started_ns == 0 or (now_ns - self._turn_started_ns) < 1_200_000_000)
            ):
                return True
        if len(stt_words) >= 2 and self._current_agent_text:
            agent_words = set(self._unicode_words(self._current_agent_text))
            if agent_words:
                overlap = sum(1 for w in stt_words if w in agent_words)
                if (overlap / len(stt_words)) >= 0.6:
                    return True
        return False

    def _wrapped_handle_stt_event(self, event: SttEvent) -> None:
        cleaned = event.text.strip()
        if not cleaned:
            return
        if self._is_echo_or_stale_stt(cleaned):
            return

        analysis = analyze(cleaned, _map_lexicon_lang(self._lang))
        state_before = self.engine.state
        stripped_tok = cleaned.rstrip(".!?,;:।").strip().lower()
        is_pure_bc = analysis.is_pure_backchannel or (
            state_before in (FsmState.SPEAKING, FsmState.DUCKED)
            and (
                len(stripped_tok) <= 1 or stripped_tok in {"o", "oh", "oo", "ah", "uh", "hm", "mm"}
            )
        )
        now_ns = self._clock.now_ns()
        user_spoke_recently = self._last_user_speech_ns > 0 and (
            (now_ns - self._last_user_speech_ns) < 1_500_000_000
        )
        if (
            not self._simulating_user_text
            and state_before in (FsmState.SPEAKING, FsmState.DUCKED)
            and not self.engine._user_speaking
            and not user_spoke_recently
            and not analysis.stop_hit
            and not is_pure_bc
        ):
            return

        self._enqueue_json(
            {
                "type": "stt",
                "kind": event.kind,
                "text": cleaned,
                "lang": event.lang or _map_stt_lang(self._lang),
            }
        )
        if event.kind == "final":
            self._pending_user_text = cleaned

        if not self._simulating_user_text and state_before == FsmState.SPEAKING:
            if (
                analysis.stop_hit
                or (self._mode == "stock" and (self.engine._user_speaking or user_spoke_recently))
                or (
                    event.kind == "final"
                    and (self.engine._user_speaking or user_spoke_recently)
                    and not is_pure_bc
                )
            ):
                self.engine._apply_event_fast_path(
                    VadOnset(ts_ns=now_ns, tag=Tag.NONE, is_echo=False),
                    now_ns,
                )
            elif event.kind == "final" and is_pure_bc:
                self._enqueue_json(
                    {
                        "type": "turn",
                        "role": "user",
                        "text": cleaned,
                        "interrupted": False,
                        "backchannel": True,
                    }
                )
                return

        if (
            not self._simulating_user_text
            and state_before == FsmState.DUCKED
            and event.kind == "final"
            and is_pure_bc
            and not analysis.is_pure_backchannel
        ):
            self._orig_handle_stt(
                SttEvent(
                    kind="final",
                    text="mm-hmm",
                    lang=event.lang,
                    conf=event.conf,
                    ts_ns=event.ts_ns,
                    words=event.words,
                )
            )
            return

        self._orig_handle_stt(event)

        if (
            event.kind == "final"
            and not self._simulating_user_text
            and state_before not in (FsmState.LISTENING, FsmState.BACKCHANNELING)
        ):
            asyncio.create_task(self._complete_live_barge_in(event))

    async def _complete_live_barge_in(self, event: SttEvent) -> None:
        cleaned = event.text.strip()
        if not cleaned or self._stopped:
            return
        analysis = analyze(cleaned, _map_lexicon_lang(self._lang))
        is_substantive = (
            bool(analysis.content_tokens)
            or analysis.question
            or len(analysis.tokens) > self._tuning.bc_max_tokens
        )
        for _ in range(15):
            if self._stopped:
                return
            if self.engine.state == FsmState.LISTENING:
                if is_substantive:
                    self._pending_user_text = cleaned
                    self._orig_handle_stt(event)
                else:
                    self._enqueue_json(
                        {
                            "type": "turn",
                            "role": "user",
                            "text": cleaned,
                            "interrupted": True,
                            "backchannel": False,
                        }
                    )
                return
            await self._step_ms(20)

    def _wrapped_handle_tts_chunk(self, chunk: AudioChunk) -> None:
        self._orig_handle_tts_chunk(chunk)
        # Do not declare TTS complete between clauses while TurnPipeline is still streaming
        if self._tts_streaming_clauses and not chunk.is_last:
            self.engine._tts_finished = False
            self.engine.mixer._tts_finished = False

    def _wrapped_on_fsm_state_changed(self, now_ns: int) -> None:
        self._orig_on_fsm_changed(now_ns)
        new_state = self.engine.state
        prev_state = self._prev_fsm_state
        self._prev_fsm_state = new_state

        # In STOCK mode, naive VAD stops immediately upon entering DUCKED
        if (
            self._mode == "stock"
            and new_state == FsmState.DUCKED
            and prev_state == FsmState.SPEAKING
            and self.engine._user_speaking
        ):
            self._trigger_stock_barge_in(now_ns)
            return

        # Launch real LLM->TTS response turn when FSM transitions into THINKING
        if new_state == FsmState.THINKING and prev_state != FsmState.THINKING:
            user_text = self._pending_user_text.strip()
            self._pending_user_text = ""
            if user_text:
                self._enqueue_json(
                    {
                        "type": "turn",
                        "role": "user",
                        "text": user_text,
                        "interrupted": False,
                        "backchannel": False,
                    }
                )
            self._start_agent_turn_async(user_text=user_text or "Hello")

        self._emit_state()

    def _trigger_stock_barge_in(self, now_ns: int) -> None:
        """Simulate naive VAD stop-on-speech policy used in Stock voice pipelines."""
        verdict = Verdict(
            label=Label.INTERRUPT,
            confidence=1.0,
            reason="stock_naive_vad_stop",
            overlap_ms=0,
            decided_at_ns=now_ns,
        )
        self.engine._last_verdict = verdict
        self.engine._last_verdict_ts_ns = now_ns
        self._telemetry.emit(
            "verdict",
            now_ns,
            label=verdict.label.value,
            confidence=verdict.confidence,
            reason=verdict.reason,
            overlap_ms=0,
            onset_ts_ns=self.engine._overlap_onset_ns or now_ns,
            partial_text=self.engine._current_partial_text or "",
            epoch=self.engine.fsm.epoch,
        )
        prev = self.engine.state
        effects = self.engine.fsm.transition(verdict, now_ns=now_ns)
        curr = self.engine.state
        if curr == FsmState.YIELDED:
            self.engine.supervisor._current_epoch = self.engine.fsm.epoch
            self.engine.mixer._current_epoch = self.engine.fsm.epoch
        if curr != prev:
            self._telemetry.emit(
                "fsm.transition",
                now_ns,
                from_state=prev.value,
                to_state=curr.value,
                event="StockVadStop",
                epoch=self.engine.fsm.epoch,
            )
        from overlap.core.types import SetGain  # local pure import for inline gain check

        for eff in effects:
            if isinstance(eff, SetGain):
                self.engine.mixer.set_gain(eff.gain, eff.ramp_ms)
        if effects:
            self.engine.supervisor.execute_effects(effects)
        self._wrapped_on_fsm_state_changed(now_ns)

    def _start_agent_turn_async(self, *, user_text: str) -> None:
        """Schedule async `TurnPipeline.run_turn` outside synchronous fast path."""
        epoch = self.engine.supervisor.current_epoch
        turn_id = self.engine.supervisor.current_turn_id or f"turn_{epoch}"
        token = self.engine.supervisor.turn_token

        self._turn_started_ns = self._clock.now_ns()
        self._last_user_turn_text = user_text
        self._current_agent_text = ""

        self.transport.reset_playout_tracking()
        self.engine.mixer.start_turn(turn_id, epoch)

        prev_task = self._active_turn_task
        if prev_task is not None and not prev_task.done():
            prev_task.cancel()

        self._tts_streaming_clauses = True
        self._active_turn_task = self.engine.supervisor.spawn_task(
            self._run_turn_and_report(user_text=user_text, epoch=epoch, cancel=token),
            name=f"demo_turn_{epoch}",
            cancel=token,
        )

    async def _run_turn_and_report(
        self, *, user_text: str, epoch: int, cancel: CancelToken
    ) -> TurnResult:
        result: TurnResult | None = None
        try:
            result = await self.engine.turn_pipeline.run_turn(
                user_text=user_text,
                epoch=epoch,
                cancel=cancel,
            )
            return result
        finally:
            self._tts_streaming_clauses = False
            if result is not None:
                if not result.interrupted and not cancel.cancelled:
                    self.engine._tts_finished = True
                    self.engine.mixer._tts_finished = True
                    if not self.engine.mixer.is_playing and self.engine.state in (
                        FsmState.SPEAKING,
                        FsmState.DUCKED,
                    ):
                        self.engine._apply_event_fast_path(
                            AgentAudioDrained(), self._clock.now_ns()
                        )
                heard = result.heard_text.strip()
                full = result.full_text.strip()
                discarded = ""
                display_text = full
                if result.interrupted:
                    marker = self._tuning.truncation_marker
                    heard_base = heard.removesuffix(marker).strip() if marker else heard
                    full_norm = " ".join(full.split())
                    if heard_base and full_norm.lower().startswith(heard_base.lower()):
                        display_text = heard
                        discarded = full_norm[len(heard_base) :].strip()
                    else:
                        display_text = ""
                        discarded = full
                if display_text or discarded:
                    self._enqueue_json(
                        {
                            "type": "turn",
                            "role": "assistant",
                            "text": display_text,
                            "full_text": full,
                            "discarded_text": discarded,
                            "interrupted": bool(result.interrupted),
                            "backchannel": False,
                        }
                    )
            self._emit_state()

    # -------------------------------------------------------------------------
    # Telemetry & Real-time WebSocket Loops
    # -------------------------------------------------------------------------

    def _rel_ms(self, ts_ns: int) -> int:
        return max(0, int((ts_ns - self._session_start_ns) // 1_000_000))

    def _on_transport_flush(self) -> None:
        self._enqueue_json({"type": "flush"})

    def _on_adapter_status(self, component: str, state: str) -> None:
        self._status[component] = state
        self._emit_snapshot()

    def _on_telemetry(self, name: str, ts_ns: int, fields: dict[str, Any]) -> None:
        if name == "verdict":
            onset_ns = int(fields.get("onset_ts_ns") or 0)
            decision_ms = (
                round((ts_ns - onset_ns) / 1_000_000.0, 1)
                if onset_ns > 0 and ts_ns >= onset_ns
                else float(fields.get("overlap_ms") or 0.0)
            )
            payload = {
                "label": str(fields.get("label", "")),
                "reason": str(fields.get("reason", "")),
                "overlap_ms": int(fields.get("overlap_ms") or 0),
                "decision_ms": decision_ms,
                "text": str(fields.get("partial_text") or ""),
            }
            self._latest_verdict_info = payload
            self._enqueue_json({"type": "verdict", **payload})

        elif name == "fsm.ignored" and fields.get("reason") == "no_escalation_signal":
            # Delayed STT partial/final for a non-interrupting user backchannel while SPEAKING
            if fields.get("event") == "SttPartial":
                txt = (self.engine._current_partial_text or "").strip()
                if txt and self.engine.state == FsmState.SPEAKING:
                    self._enqueue_json(
                        {
                            "type": "turn",
                            "role": "user",
                            "text": txt,
                            "interrupted": False,
                            "backchannel": True,
                        }
                    )

        elif name == "duck.applied":
            self._emit_state()

        self._enqueue_json(
            {
                "type": "telemetry",
                "name": name,
                "ts_ms": self._rel_ms(ts_ns),
                "fields": {
                    k: (v.value if hasattr(v, "value") else v)
                    for k, v in fields.items()
                    if k != "onset_ts_ns"
                },
            }
        )

    def _emit_state(self) -> None:
        self._enqueue_json(
            {
                "type": "state",
                "state": self.engine.state.value,
                "gain": round(float(self.engine.mixer.current_gain), 2),
                "target_gain": round(float(self.engine.mixer.target_gain), 2),
                "mode": self._mode,
                "lang": self._lang,
                "epoch": self.engine.fsm.epoch,
                "vad_active": bool(self.engine._user_speaking),
                "verdict": self._latest_verdict_info,
            }
        )

    def _emit_snapshot(self) -> None:
        self._enqueue_json(
            {
                "type": "status",
                "mode": self._mode,
                "lang": self._lang,
                "voice": self._voice_id,
                "mic_active": self._mic_active,
                "providers": dict(self._status),
            }
        )
        self._emit_state()

    def _enqueue_json(self, payload: dict[str, Any]) -> None:
        if self._stopped:
            return
        if self._ws_outbound_queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._ws_outbound_queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            self._ws_outbound_queue.put_nowait(("json", payload))

    def _enqueue_bytes(self, pcm: bytes) -> None:
        if self._stopped or not pcm:
            return
        if self._ws_outbound_queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._ws_outbound_queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            self._ws_outbound_queue.put_nowait(("bytes", pcm))

    async def _ws_tx_worker(self) -> None:
        try:
            while not self._stopped:
                kind, payload = await self._ws_outbound_queue.get()
                if kind == "json" and self._send_json_cb is not None:
                    res = self._send_json_cb(payload)
                    if inspect.isawaitable(res):
                        await res
                elif kind == "bytes" and self._send_bytes_cb is not None:
                    res = self._send_bytes_cb(payload)
                    if inspect.isawaitable(res):
                        await res
        except asyncio.CancelledError:
            return

    async def _agent_audio_pump(self) -> None:
        try:
            while not self._stopped:
                frames = await self.transport.pop_outbound_frames()
                for frame in frames:
                    self._enqueue_bytes(frame.pcm)
        except asyncio.CancelledError:
            return

    async def _idle_frame_pump(self) -> None:
        """Keep `OverlapEngine` advancing smoothly at 50 Hz when live mic isn't streaming."""
        silence_pcm = b"\x00" * (16000 * 2 * self._tuning.frame_ms // 1000)
        try:
            while not self._stopped:
                await asyncio.sleep(self._tuning.frame_ms / 1000.0)
                if self._stopped or self._simulating_user_text:
                    continue
                now_ns = self._clock.now_ns()
                if self._mic_active and now_ns - self._last_mic_rx_ns <= 250_000_000:
                    continue
                if now_ns - self._last_mic_rx_ns > 80_000_000:
                    self.engine.handle_frame(
                        AudioFrame(pcm=silence_pcm, sample_rate=16000, ts_ns=now_ns)
                    )
        except asyncio.CancelledError:
            return

    async def _meter_loop(self) -> None:
        """Stream smooth 25 Hz signal levels + overlap window timer to the browser canvas."""
        try:
            while not self._stopped:
                await asyncio.sleep(0.04)
                now_ns = self._clock.now_ns()
                user_lvl = (
                    self._latest_user_level
                    if (
                        self._simulating_user_text or (now_ns - self._last_mic_rx_ns <= 120_000_000)
                    )
                    else 0.0
                )
                overlap_ms = (
                    max(0, int((now_ns - self.engine._overlap_onset_ns) // 1_000_000))
                    if self.engine.state == FsmState.DUCKED and self.engine._overlap_onset_ns > 0
                    else 0
                )
                self._enqueue_json(
                    {
                        "type": "meter",
                        "state": self.engine.state.value,
                        "gain": round(float(self.engine.mixer.current_gain), 2),
                        "user_level": round(user_lvl, 3),
                        "agent_level": round(self.transport.last_agent_level, 3),
                        "vad_active": bool(self.engine._user_speaking),
                        "overlap_ms": overlap_ms,
                    }
                )
        except asyncio.CancelledError:
            return


# =============================================================================
# HTTP & WebSocket Server (`aiohttp.web`)
# =============================================================================


def load_static_index_html() -> str:
    """Read bundled single-page desktop frontend HTML."""
    local_path = Path(__file__).resolve().parent / "static" / "index.html"
    if local_path.is_file():
        return local_path.read_text(encoding="utf-8")
    return (
        importlib.resources.files("overlap.demo")
        .joinpath("static/index.html")
        .read_text(encoding="utf-8")
    )


def create_demo_app(
    *,
    lang: str = "en",
    voice_id: str = "Nalini",
    mode: DemoMode = "overlap",
    use_live_apis: bool = True,
) -> web.Application:
    """Create configured `aiohttp.web.Application` serving frontend and `/ws` endpoint."""
    app = web.Application()
    html_content = load_static_index_html()

    async def handle_index(_request: web.Request) -> web.Response:
        return web.Response(text=html_content, content_type="text/html", charset="utf-8")

    async def handle_health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "lang": lang, "voice": voice_id, "mode": mode})

    async def handle_ws(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024)
        await ws.prepare(request)

        async def send_json(data: dict[str, Any]) -> None:
            if not ws.closed:
                await ws.send_json(data)

        async def send_bytes(data: bytes) -> None:
            if not ws.closed:
                await ws.send_bytes(data)

        session = DemoSession(
            send_json=send_json,
            send_bytes=send_bytes,
            lang=lang,
            voice_id=voice_id,
            mode=mode,
            use_live_apis=use_live_apis,
        )
        await session.start()
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    session.handle_incoming_pcm(bytes(msg.data))
                elif msg.type == WSMsgType.TEXT:
                    try:
                        payload = msg.json()
                    except Exception:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    msg_type = payload.get("type")
                    if msg_type == "user_text":
                        txt = str(payload.get("text", ""))
                        asyncio.create_task(session.handle_user_text(txt), name="demo_user_text")
                    elif msg_type == "set_mode":
                        new_mode = "stock" if payload.get("mode") == "stock" else "overlap"
                        session.set_mode(cast(DemoMode, new_mode))
                    elif msg_type == "set_lang":
                        await session.set_language(str(payload.get("lang", "en")))
                    elif msg_type == "mic_state":
                        await session.set_mic_active(bool(payload.get("active", False)))
                    elif msg_type == "buffer_ms":
                        session.transport.update_client_buffer_ms(
                            int(payload.get("buffered_ms", 0))
                        )
                    elif msg_type == "reset":
                        await session.reset()
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            await session.stop()
            if not ws.closed:
                await ws.close()

        return ws

    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/ws", handle_ws)
    return app


async def run_demo_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    lang: str = "en",
    voice_id: str = "Nalini",
    open_browser: bool = True,
) -> None:
    """Run the OVERLAP desktop demo HTTP + WebSocket server until cancelled."""
    load_dotenv_if_present()
    app = create_demo_app(lang=lang, voice_id=voice_id)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()

    url = f"http://{host}:{port}"
    print(f"OVERLAP live demo running at {url}  (Ctrl+C to quit)")

    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()

    try:
        while True:
            await asyncio.sleep(3600.0)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await runner.cleanup()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m overlap.demo`."""
    parser = argparse.ArgumentParser(
        prog="python -m overlap.demo",
        description="Run the live OVERLAP full-duplex voice desktop demo.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8765")),
        help="Bind port (default: $PORT or 8765)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help="Initial conversation language (default: en)",
    )
    parser.add_argument("--voice", default="Nalini", help="Gnani Timbre voice ID (default: Nalini)")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the browser tab",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        asyncio.run(
            run_demo_server(
                host=args.host,
                port=args.port,
                lang=args.lang,
                voice_id=args.voice,
                open_browser=not args.no_browser,
            )
        )
    except KeyboardInterrupt:
        pass
    return 0
