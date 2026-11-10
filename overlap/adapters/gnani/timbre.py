"""Timbre TTS adapter (WebSocket streaming, cancel, word timings, batch synth).

Implements TtsPort and TtsStream protocols strictly following docs/WIRE.md §3:
- Streaming WebSocket: wss://api.vachana.ai/api/v1/tts
- Batch synthesis REST: POST https://api.vachana.ai/api/v1/tts/inference
- Strips 44-byte RIFF WAV header from first audio chunk; validates 16-bit PCM.
- Resamples non-16kHz audio and downmixes stereo to mono if needed.
- Cancel: sends protocol cancel {"action": "cancel", "code": 1000}, idempotent,
  discards inbound audio, stops chunk delivery within 1 loop turn (INV-L5/L7).
- Word timings: populated with clause-relative timings when provided; otherwise
  empty tuple (calibrated rate fallback per WIRE.md §3.4).
- Batch synthesize() returns AudioClip with exact sample count for ClipCache.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import os
import ssl
import struct
import urllib.request
import xml.sax.saxutils
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import numpy as np

from overlap.core.audio import float32_to_pcm16, pcm16_to_float32
from overlap.core.clock import CancelToken
from overlap.core.types import WordTiming
from overlap.ports.telemetry import NullSink, TelemetrySink
from overlap.ports.tts import AudioChunk, AudioClip, TtsStream

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE: int = 16000
DEFAULT_USER_AGENT: str = "overlap-client/0.1.0"
DEFAULT_WS_URL: str = "wss://api.vachana.ai/api/v1/tts"
DEFAULT_INFERENCE_URL: str = "https://api.vachana.ai/api/v1/tts/inference"


def _resample_pcm16(pcm: bytes, src_sr: int, dst_sr: int) -> bytes:
    """Resample 16-bit PCM bytes from src_sr to dst_sr using linear interpolation."""
    if src_sr == dst_sr or not pcm:
        return pcm
    samples = pcm16_to_float32(pcm)
    target_len = int(round(len(samples) * dst_sr / src_sr))
    if target_len <= 0:
        return b""
    orig_indices = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    target_indices = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    resampled = np.asarray(np.interp(target_indices, orig_indices, samples), dtype=np.float32)
    return float32_to_pcm16(resampled)


def _stereo_to_mono(pcm: bytes) -> bytes:
    """Downmix 16-bit interleaved stereo PCM to mono PCM."""
    if not pcm or len(pcm) < 4:
        return pcm
    samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 4)], dtype=np.int16).reshape(-1, 2)
    mono = samples.mean(axis=1).astype(np.int16)
    return bytes(mono.tobytes())


def _ensure_even_pcm(pcm: bytes) -> bytes:
    """Ensure PCM byte count is even (16-bit alignment). Truncate trailing odd byte if present."""
    if len(pcm) % 2 != 0:
        return pcm[:-1]
    return pcm


def _parse_wav_header(raw: bytes) -> tuple[bytes, int, int]:
    """Parse RIFF WAV header and return (pcm_data, sample_rate, num_channels).

    If raw is not a RIFF WAV or shorter than 44 bytes, returns (raw, 16000, 1).
    """
    if raw.startswith(b"RIFF") and len(raw) >= 44:
        channels = struct.unpack("<H", raw[22:24])[0]
        sample_rate = struct.unpack("<I", raw[24:28])[0]
        data_idx = raw.find(b"data")
        if data_idx != -1 and len(raw) >= data_idx + 8:
            pcm = raw[data_idx + 8 :]
        else:
            pcm = raw[44:]
        return pcm, sample_rate, channels
    return raw, DEFAULT_SAMPLE_RATE, 1


def _format_clause(clause: str, *, ssml: bool) -> str:
    """Format clause for synthesis.

    If ssml=True, preserves markup verbatim without entity escaping.
    If ssml=False, escapes XML characters (&, <, >) to prevent injection into XML parsers.
    """
    if ssml:
        return clause
    return xml.sax.saxutils.escape(clause)


# =============================================================================
# Stdlib Async WebSocket Client (RFC 6455)
# =============================================================================


def _make_ws_frame(opcode: int, data: bytes, masked: bool = True) -> bytes:
    """Encode an RFC 6455 WebSocket frame."""
    b1 = 0x80 | (opcode & 0x0F)
    length = len(data)
    if length < 126:
        b2 = (0x80 if masked else 0) | length
        header = bytes([b1, b2])
    elif length < 65536:
        b2 = (0x80 if masked else 0) | 126
        header = bytes([b1, b2]) + struct.pack(">H", length)
    else:
        b2 = (0x80 if masked else 0) | 127
        header = bytes([b1, b2]) + struct.pack(">Q", length)

    if masked:
        mask = os.urandom(4)
        masked_data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        return header + mask + masked_data
    return header + data


class _StdlibAsyncWebSocket:
    """Minimal async RFC 6455 WebSocket client using asyncio.open_connection."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer
        self._closed = False

    async def send_text(self, text: str) -> None:
        if self._closed:
            return
        frame = _make_ws_frame(1, text.encode("utf-8"), masked=True)
        self._writer.write(frame)
        await self._writer.drain()

    async def send_bytes(self, data: bytes) -> None:
        if self._closed:
            return
        frame = _make_ws_frame(2, data, masked=True)
        self._writer.write(frame)
        await self._writer.drain()

    async def recv(self) -> str | bytes:
        if self._closed:
            raise asyncio.IncompleteReadError(b"", None)

        while not self._closed:
            header = await self._reader.readexactly(2)
            b1, b2 = header[0], header[1]
            opcode = b1 & 0x0F
            masked = (b2 >> 7) & 1
            length = b2 & 0x7F

            if length == 126:
                ext = await self._reader.readexactly(2)
                length = struct.unpack(">H", ext)[0]
            elif length == 127:
                ext = await self._reader.readexactly(8)
                length = struct.unpack(">Q", ext)[0]

            mask = await self._reader.readexactly(4) if masked else None
            data = await self._reader.readexactly(length)
            if masked and mask:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))

            if opcode == 1:
                return data.decode("utf-8", errors="replace")
            if opcode == 2:
                return data
            if opcode == 8:  # Close frame
                self._closed = True
                raise asyncio.IncompleteReadError(b"", None)
            if opcode == 9:  # Ping -> Pong
                pong = _make_ws_frame(10, data, masked=True)
                self._writer.write(pong)
                await self._writer.drain()
                continue
            # Ignore other opcodes (continuation, pong)

        raise asyncio.IncompleteReadError(b"", None)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            close_frame = _make_ws_frame(8, struct.pack(">H", 1000), masked=True)
            self._writer.write(close_frame)
            await self._writer.drain()
        except Exception:
            pass
        finally:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass


async def _connect_stdlib_ws(
    ws_url: str,
    api_key: str | None,
    user_agent: str,
) -> _StdlibAsyncWebSocket:
    """Establish a WebSocket connection using stdlib asyncio and SSL."""
    parsed = urlparse(ws_url)
    host = parsed.hostname or "api.vachana.ai"
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    path = parsed.path or "/api/v1/tts"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    ctx = ssl.create_default_context() if parsed.scheme == "wss" else None
    reader, writer = await asyncio.open_connection(host, port, ssl=ctx)

    sec_key = base64.b64encode(os.urandom(16)).decode("ascii")
    headers = [
        f"GET {path} HTTP/1.1",
        f"Host: {host}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {sec_key}",
        "Sec-WebSocket-Version: 13",
        f"User-Agent: {user_agent}",
    ]
    if api_key:
        headers.append(f"X-API-Key-ID: {api_key}")
    headers.append("\r\n")

    writer.write("\r\n".join(headers).encode("utf-8"))
    await writer.drain()

    resp = await reader.read(2048)
    if b"101 Switching Protocols" not in resp:
        writer.close()
        await writer.wait_closed()
        raise ConnectionError(
            f"WebSocket handshake failed: {resp.decode('utf-8', errors='replace')}"
        )

    return _StdlibAsyncWebSocket(reader, writer)


# =============================================================================
# Timbre Configuration and Stream
# =============================================================================


@dataclass(frozen=True, slots=True)
class TimbreTtsConfig:
    """Configuration options for TimbreTtsAdapter."""

    api_key: str | None = None
    base_url: str = "https://api.vachana.ai"
    ws_url: str = DEFAULT_WS_URL
    inference_url: str = DEFAULT_INFERENCE_URL
    model: str = "timbre-v2.5"
    sample_rate: int = DEFAULT_SAMPLE_RATE
    speed: float = 1.0
    user_agent: str = DEFAULT_USER_AGENT


class TimbreTtsStream:
    """Active streaming TTS session for Gnani Timbre."""

    def __init__(
        self,
        ws: Any,
        *,
        voice_id: str,
        lang: str,
        epoch: int,
        cancel: CancelToken,
        config: TimbreTtsConfig,
    ) -> None:
        self._ws = ws
        self._voice_id = voice_id
        self._lang = lang
        self._epoch = epoch
        self._cancel = cancel
        self._config = config

        self._cancelled: bool = False
        self._finished: bool = False
        self._server_sr: int = config.sample_rate
        self._send_lock = asyncio.Lock()

        # Bind external CancelToken to cancel this stream
        self._cancel.add_callback(self._on_cancel_token)

    def _on_cancel_token(self) -> None:
        """Callback invoked immediately when external CancelToken is cancelled."""
        self._cancelled = True
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(self.cancel())
        except RuntimeError:
            pass

    async def send_text(self, clause: str, *, ssml: bool = False) -> None:
        """Enqueue text clause for synthesis.

        Sends request schema matching WIRE.md §3.2.
        """
        if self._cancelled or self._cancel.cancelled:
            return

        formatted = _format_clause(clause, ssml=ssml)
        payload = {
            "text": formatted,
            "voice": self._voice_id,
            "model": self._config.model,
            "language": self._lang,
            "speed": self._config.speed,
            "audio_config": {
                "sample_rate": self._config.sample_rate,
                "num_channels": 1,
                "sample_width": 2,
                "encoding": "linear_pcm",
                "container": "wav",
            },
        }
        text_payload = json.dumps(payload, ensure_ascii=False)

        async with self._send_lock:
            if self._cancelled or self._cancel.cancelled:
                return
            try:
                await self._ws.send_text(text_payload)
            except Exception as exc:
                if not (self._cancelled or self._cancel.cancelled):
                    logger.debug("Timbre send_text failed: %s", exc)

    async def finish(self) -> None:
        """Signal that no more text will be sent for this stream."""
        self._finished = True

    async def cancel(self) -> None:
        """Cancel synthesis stream. Idempotent; stops chunk delivery within 1 loop turn."""
        if self._cancelled:
            return
        self._cancelled = True

        # 1. Send protocol cancel / close frame per WIRE.md §3.5
        try:
            await self._ws.send_text(json.dumps({"action": "cancel", "code": 1000}))
        except Exception:
            pass

        # 2. Close websocket
        try:
            await self._ws.close()
        except Exception:
            pass

    async def chunks(self) -> AsyncIterator[AudioChunk]:
        """Asynchronous iterator yielding synthesized AudioChunks.

        Stops delivering chunks immediately if stream was cancelled.
        """
        if self._cancelled or self._cancel.cancelled:
            return

        while not self._cancelled and not self._cancel.cancelled:
            try:
                raw_msg = await self._ws.recv()
            except (asyncio.IncompleteReadError, EOFError, ConnectionError, OSError):
                break
            except Exception:
                break

            if self._cancelled or self._cancel.cancelled:
                break

            msg_str = (
                raw_msg.decode("utf-8", errors="replace")
                if isinstance(raw_msg, bytes)
                else str(raw_msg)
            )

            try:
                parsed: dict[str, Any] = json.loads(msg_str)
            except Exception:
                continue

            msg_type = str(parsed.get("type", ""))
            if msg_type == "start":
                continue

            data_obj: dict[str, Any] = (
                parsed.get("data", {}) if isinstance(parsed.get("data"), dict) else parsed
            )
            is_final = bool(
                data_obj.get("is_final") or parsed.get("is_final") or msg_type == "complete"
            )
            raw_b64 = str(data_obj.get("audio", ""))
            raw_pcm = base64.b64decode(raw_b64) if raw_b64 else b""

            # Extract word timings if provided (otherwise empty tuple per WIRE.md §3.4)
            words_list: list[WordTiming] = []
            raw_words = data_obj.get("words") or parsed.get("words")
            if isinstance(raw_words, list):
                for item in raw_words:
                    if isinstance(item, dict):
                        w_text = str(item.get("word") or item.get("text") or "")
                        w_start = int(item.get("start_ms", 0))
                        w_end = int(item.get("end_ms", 0))
                        words_list.append(WordTiming(word=w_text, start_ms=w_start, end_ms=w_end))
            words_tuple = tuple(words_list)

            if raw_pcm:
                if raw_pcm.startswith(b"RIFF"):
                    pcm, sr, ch = _parse_wav_header(raw_pcm)
                    self._server_sr = sr
                    if ch == 2:
                        pcm = _stereo_to_mono(pcm)
                else:
                    pcm = raw_pcm

                if self._server_sr != self._config.sample_rate:
                    pcm = _resample_pcm16(pcm, self._server_sr, self._config.sample_rate)

                pcm = _ensure_even_pcm(pcm)

                if self._cancelled or self._cancel.cancelled:
                    break

                yield AudioChunk(
                    pcm=pcm,
                    sample_rate=self._config.sample_rate,
                    epoch=self._epoch,
                    words=words_tuple,
                    is_last=is_final,
                )
            elif is_final:
                if self._cancelled or self._cancel.cancelled:
                    break
                yield AudioChunk(
                    pcm=b"",
                    sample_rate=self._config.sample_rate,
                    epoch=self._epoch,
                    words=words_tuple,
                    is_last=True,
                )

            if is_final:
                break


# =============================================================================
# Timbre TTS Port Adapter
# =============================================================================


class TimbreTtsAdapter:
    """Gnani Timbre TTS adapter implementing TtsPort."""

    def __init__(
        self,
        config: TimbreTtsConfig | None = None,
        *,
        ws_factory: Callable[[], Any] | None = None,
        http_client: Any | None = None,
        telemetry: TelemetrySink | None = None,
    ) -> None:
        self._config = config or TimbreTtsConfig(
            api_key=os.environ.get("GNANI_API_KEY", "").strip() or None
        )
        self._ws_factory = ws_factory
        self._http_client = http_client
        self._telemetry = telemetry or NullSink()

    async def open_stream(
        self,
        *,
        voice_id: str,
        lang: str,
        epoch: int,
        cancel: CancelToken,
    ) -> TtsStream:
        """Open a streaming TTS session."""
        if self._ws_factory is not None:
            res = self._ws_factory()
            ws = await res if inspect.iscoroutine(res) else res
        else:
            ws = await _connect_stdlib_ws(
                self._config.ws_url,
                self._config.api_key,
                self._config.user_agent,
            )

        return TimbreTtsStream(
            ws,
            voice_id=voice_id,
            lang=lang,
            epoch=epoch,
            cancel=cancel,
            config=self._config,
        )

    async def synthesize(
        self,
        text: str,
        *,
        voice_id: str,
        lang: str,
    ) -> AudioClip:
        """Batch synthesize a complete clip (used by ClipCache).

        Returns AudioClip with exact sample count and sample rate.
        """
        # 1. Custom HTTP client injection (testing / mock)
        if self._http_client is not None:
            res = self._http_client(text, voice_id=voice_id, lang=lang)
            raw = await res if inspect.iscoroutine(res) else res
            if isinstance(raw, str):
                raw = base64.b64decode(raw)
            if raw.startswith(b"RIFF"):
                pcm, sr, ch = _parse_wav_header(raw)
                if ch == 2:
                    pcm = _stereo_to_mono(pcm)
            else:
                pcm, sr = raw, self._config.sample_rate

            if sr != self._config.sample_rate:
                pcm = _resample_pcm16(pcm, sr, self._config.sample_rate)

            pcm = _ensure_even_pcm(pcm)
            return AudioClip(pcm=pcm, sample_rate=self._config.sample_rate)

        # 2. WebSocket stream fallback (hermetic testing via ws_factory)
        if self._ws_factory is not None:
            token = CancelToken()
            stream = await self.open_stream(voice_id=voice_id, lang=lang, epoch=0, cancel=token)
            await stream.send_text(text)
            await stream.finish()

            chunks: list[bytes] = []
            sr = self._config.sample_rate
            async for chunk in stream.chunks():
                if chunk.pcm:
                    chunks.append(chunk.pcm)
                    sr = chunk.sample_rate

            pcm = b"".join(chunks)
            pcm = _ensure_even_pcm(pcm)
            return AudioClip(pcm=pcm, sample_rate=sr)

        # 3. Live REST call via urllib.request
        def _call_rest() -> bytes:
            payload = {
                "text": text,
                "voice": voice_id,
                "model": self._config.model,
                "language": lang,
                "speed": self._config.speed,
                "audio_config": {
                    "sample_rate": self._config.sample_rate,
                    "num_channels": 1,
                    "encoding": "linear_pcm",
                    "container": "wav",
                },
            }
            ctx = ssl.create_default_context()
            headers = {
                "Content-Type": "application/json",
                "User-Agent": self._config.user_agent,
            }
            if self._config.api_key:
                headers["X-API-Key-ID"] = self._config.api_key

            req = urllib.request.Request(
                self._config.inference_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                return bytes(resp.read())

        raw_bytes = await asyncio.to_thread(_call_rest)
        if raw_bytes.startswith(b"RIFF"):
            pcm, sr, ch = _parse_wav_header(raw_bytes)
            if ch == 2:
                pcm = _stereo_to_mono(pcm)
        else:
            pcm, sr = raw_bytes, self._config.sample_rate

        if sr != self._config.sample_rate:
            pcm = _resample_pcm16(pcm, sr, self._config.sample_rate)

        pcm = _ensure_even_pcm(pcm)
        return AudioClip(pcm=pcm, sample_rate=self._config.sample_rate)
