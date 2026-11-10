"""Gnani wire protocol utilities, constants, headers, and RFC 6455 WebSocket client.

Verified against docs/WIRE.md and live production captures from T3.01.
"""

from __future__ import annotations

import asyncio
import base64
import os
import ssl
import struct
import urllib.parse
from collections.abc import AsyncIterator

# =============================================================================
# Constants & Endpoints (per docs/WIRE.md)
# =============================================================================

DEFAULT_GNANI_BASE_URL: str = "https://api.vachana.ai"
DEFAULT_GNANI_WS_URL: str = "wss://api.vachana.ai"

PRISMA_STT_WS_PATH: str = "/stt/v3/stream"
PRISMA_STT_WS_URL: str = f"{DEFAULT_GNANI_WS_URL}{PRISMA_STT_WS_PATH}"

TIMBRE_TTS_WS_PATH: str = "/api/v1/tts"
TIMBRE_TTS_WS_URL: str = f"{DEFAULT_GNANI_WS_URL}{TIMBRE_TTS_WS_PATH}"
TIMBRE_TTS_REST_PATH: str = "/api/v1/tts/inference"

# Explicit User-Agent required to bypass Cloudflare bot protection (error 1010)
DEFAULT_USER_AGENT: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

# Audio format: raw mono PCM 16-bit linear (int16), little-endian at 16,000 Hz
PRISMA_SAMPLE_RATE: int = 16000
PRISMA_FRAME_MS: int = 20
PRISMA_FRAME_BYTES: int = 640  # 320 samples * 2 bytes


# =============================================================================
# Wire Exceptions
# =============================================================================


class GnaniWireError(Exception):
    """Base exception for Gnani wire protocol errors."""


class GnaniAuthenticationError(GnaniWireError):
    """Authentication failure (e.g. invalid or expired API key, HTTP 403)."""


class GnaniRateLimitError(GnaniWireError):
    """Rate limit exceeded (HTTP 429)."""


# =============================================================================
# Header Builders
# =============================================================================


def build_prisma_headers(
    api_key: str,
    *,
    model: str = "prisma-v3.0",
    language_code: str = "indic-auto",
    x_partials: bool = True,
    format_style: str = "verbatim",
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict[str, str]:
    """Construct HTTP handshake headers for Prisma STT WebSocket upgrade."""
    return {
        "X-API-Key-ID": api_key,
        "model": model,
        "language_code": language_code,
        "x-partials": "true" if x_partials else "false",
        "format": format_style,
        "User-Agent": user_agent,
    }


# =============================================================================
# RFC 6455 WebSocket Framing
# =============================================================================


def make_ws_frame(opcode: int, data: bytes, masked: bool = True) -> bytes:
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


# =============================================================================
# Async WebSocket Client (Pure stdlib)
# =============================================================================


class AsyncWebSocket:
    """Minimal async RFC 6455 WebSocket client using asyncio.open_connection."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer
        self._closed = False

    @property
    def is_closed(self) -> bool:
        """Whether the WebSocket connection is closed."""
        return self._closed

    @classmethod
    async def connect(
        cls,
        url: str,
        headers: dict[str, str],
        ssl_context: ssl.SSLContext | None = None,
        timeout: float = 10.0,
    ) -> AsyncWebSocket:
        """Establish WebSocket connection with HTTP upgrade handshake."""
        parsed = urllib.parse.urlsplit(url)
        host = parsed.hostname or "api.vachana.ai"
        is_ssl = parsed.scheme in ("wss", "https")
        port = parsed.port or (443 if is_ssl else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        ssl_ctx: ssl.SSLContext | None = None
        if is_ssl:
            ssl_ctx = ssl_context or ssl.create_default_context()

        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=ssl_ctx),
            timeout=timeout,
        )

        sec_key = base64.b64encode(os.urandom(16)).decode("ascii")
        req_lines = [
            f"GET {path} HTTP/1.1",
            f"Host: {host}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {sec_key}",
            "Sec-WebSocket-Version: 13",
        ]
        for k, v in headers.items():
            req_lines.append(f"{k}: {v}")
        req_lines.append("\r\n")
        req_bytes = "\r\n".join(req_lines).encode("utf-8")

        writer.write(req_bytes)
        await writer.drain()

        # Read response status line
        status_line_bytes = await reader.readline()
        if not status_line_bytes:
            raise GnaniWireError("Connection closed by server during WebSocket handshake")

        status_line = status_line_bytes.decode("latin1").strip()
        parts = status_line.split(" ", 2)
        if len(parts) < 2:
            raise GnaniWireError(f"Malformed HTTP response line: {status_line}")

        try:
            status_code = int(parts[1])
        except ValueError as err:
            raise GnaniWireError(f"Invalid HTTP status code in response: {status_line}") from err

        if status_code == 403:
            raise GnaniAuthenticationError(f"HTTP 403 Forbidden: {status_line}")
        if status_code == 429:
            raise GnaniRateLimitError(f"HTTP 429 Too Many Requests: {status_line}")
        if status_code != 101:
            raise GnaniWireError(
                f"WebSocket upgrade rejected with status {status_code}: {status_line}"
            )

        # Drain response headers until empty line
        while True:
            header_line = await reader.readline()
            if not header_line or header_line in (b"\r\n", b"\n"):
                break

        return cls(reader, writer)

    async def send_text(self, text: str) -> None:
        """Send a text frame (opcode 0x1)."""
        if self._closed:
            raise RuntimeError("Cannot send on closed WebSocket")
        frame = make_ws_frame(0x1, text.encode("utf-8"), masked=True)
        self._writer.write(frame)
        await self._writer.drain()

    async def send_bytes(self, data: bytes) -> None:
        """Send a binary frame (opcode 0x2)."""
        if self._closed:
            raise RuntimeError("Cannot send on closed WebSocket")
        frame = make_ws_frame(0x2, data, masked=True)
        self._writer.write(frame)
        await self._writer.drain()

    async def send(self, data: str | bytes) -> None:
        """Send either a text or binary frame."""
        if isinstance(data, str):
            await self.send_text(data)
        else:
            await self.send_bytes(data)

    async def recv_frame(self) -> tuple[int, bytes] | None:
        """Read and decode a single RFC 6455 frame from the stream."""
        try:
            header = await self._reader.readexactly(2)
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            return None

        b1, b2 = header[0], header[1]
        opcode = b1 & 0x0F
        masked = (b2 >> 7) & 1
        length = b2 & 0x7F
        if length == 126:
            try:
                ext = await self._reader.readexactly(2)
            except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
                return None
            length = struct.unpack(">H", ext)[0]
        elif length == 127:
            try:
                ext = await self._reader.readexactly(8)
            except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
                return None
            length = struct.unpack(">Q", ext)[0]

        mask: bytes | None = None
        if masked:
            try:
                mask = await self._reader.readexactly(4)
            except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
                return None

        try:
            data = await self._reader.readexactly(length)
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            return None

        if masked and mask:
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))

        return opcode, data

    async def recv(self) -> str | bytes:
        """Receive the next message payload (handling ping/pong/close)."""
        while not self._closed:
            res = await self.recv_frame()
            if res is None:
                self._closed = True
                raise asyncio.IncompleteReadError(b"", None)

            opcode, data = res
            if opcode == 0x1:
                return data.decode("utf-8")
            if opcode == 0x2:
                return data
            if opcode == 0x9:
                # Ping -> reply Pong (0xA)
                pong = make_ws_frame(0xA, data, masked=True)
                self._writer.write(pong)
                await self._writer.drain()
                continue
            if opcode == 0xA:
                # Pong -> ignore
                continue
            if opcode == 0x8:
                # Close frame
                self._closed = True
                raise asyncio.IncompleteReadError(b"", None)

        raise asyncio.IncompleteReadError(b"", None)

    async def recv_text(self) -> str:
        """Receive next text message."""
        res = await self.recv()
        if isinstance(res, str):
            return res
        return res.decode("utf-8")

    async def recv_bytes(self) -> bytes:
        """Receive next binary message."""
        res = await self.recv()
        if isinstance(res, bytes):
            return res
        return res.encode("utf-8")

    async def __aiter__(self) -> AsyncIterator[str | bytes]:
        """Iterate over inbound messages until closed."""
        while not self._closed:
            try:
                yield await self.recv()
            except asyncio.IncompleteReadError:
                break

    async def close(self, code: int = 1000) -> None:
        """Close connection with optional RFC 6455 close status code."""
        if self._closed:
            return
        self._closed = True
        try:
            payload = struct.pack(">H", code)
            self._writer.write(make_ws_frame(0x8, payload, masked=True))
            await self._writer.drain()
        except Exception:
            pass
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:
            pass
