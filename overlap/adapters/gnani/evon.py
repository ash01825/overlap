"""Evon / Gemini LLM streaming adapter with abort and incremental UTF-8 decoding.

Implements LlmPort and LlmStream protocols per docs/WIRE.md §4 and ARCHITECTURE.md §13.
Supports OpenAI-compatible SSE streaming completions, incremental decoding of multi-byte
Indic text (Devanagari, Tamil), deterministic timeouts via injected Clock, and immediate abort.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import logging
import os
import ssl
import urllib.parse
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from overlap.core.clock import CancelToken, Clock, MonotonicClock
from overlap.ports.llm import LlmPort, LlmStream, Message
from overlap.ports.telemetry import NullSink, TelemetrySink

logger = logging.getLogger(__name__)

# Default configuration constants per docs/WIRE.md §1, §4, §6
DEFAULT_ENDPOINT_URL: str = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
DEFAULT_MODEL: str = "gemini-2.5-flash"
DEFAULT_USER_AGENT: str = "overlap-client/0.1.0"
DEFAULT_CONNECT_TIMEOUT_S: float = 2.0
DEFAULT_FIRST_TOKEN_TIMEOUT_S: float = 2.0
DEFAULT_TOKEN_TIMEOUT_S: float = 5.0


# =============================================================================
# Exception Hierarchy (§14)
# =============================================================================


class LlmError(Exception):
    """Base exception for all Evon LLM adapter failures."""


class LlmConnectionError(LlmError):
    """Raised when connection to the LLM upstream fails or terminates unexpectedly."""


class LlmAuthError(LlmError):
    """Raised on authentication or permission failure (e.g. HTTP 401 or 403)."""


class LlmRateLimitError(LlmError):
    """Raised when rate limits are exceeded (HTTP 429)."""


class LlmTimeoutError(LlmError):
    """Raised when connect, first-token, or intermediate token generation times out."""


class LlmStreamError(LlmError):
    """Raised when an error occurs mid-stream or stream payload is malformed."""


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass(frozen=True, slots=True)
class EvonConfig:
    """Configuration parameters for Evon / Gemini LLM streaming adapter."""

    api_key: str | None = None
    endpoint_url: str = DEFAULT_ENDPOINT_URL
    model: str = DEFAULT_MODEL
    user_agent: str = DEFAULT_USER_AGENT
    connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S
    first_token_timeout_s: float = DEFAULT_FIRST_TOKEN_TIMEOUT_S
    token_timeout_s: float = DEFAULT_TOKEN_TIMEOUT_S


# =============================================================================
# Transport Protocols & Implementations
# =============================================================================


@runtime_checkable
class EvonTransport(Protocol):
    """Transport protocol for sending an HTTP request and receiving streaming byte chunks."""

    async def connect_and_stream(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
        cancel: CancelToken,
        on_close: Callable[[], None] | None = None,
    ) -> AsyncIterator[bytes]:
        """Establish connection and stream response body as raw byte chunks."""
        ...

    async def abort(self) -> None:
        """Abort the active transport connection immediately."""
        ...


class HttpTransport(EvonTransport):
    """Live HTTP/1.1 SSE streaming transport using asyncio.open_connection."""

    def __init__(self) -> None:
        self._writer: asyncio.StreamWriter | None = None
        self._aborted = False

    async def connect_and_stream(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
        cancel: CancelToken,
        on_close: Callable[[], None] | None = None,
    ) -> AsyncIterator[bytes]:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"

        ssl_ctx = ssl.create_default_context() if parsed.scheme == "https" else None

        try:
            reader, writer = await asyncio.open_connection(host, port, ssl=ssl_ctx)
        except Exception as exc:
            raise LlmConnectionError(f"Failed to connect to {host}:{port}: {exc}") from exc

        self._writer = writer

        body_bytes = json.dumps(payload).encode("utf-8")
        req_lines = [
            f"POST {path} HTTP/1.1",
            f"Host: {host}",
            f"Content-Length: {len(body_bytes)}",
        ]
        for k, v in headers.items():
            req_lines.append(f"{k}: {v}")
        req_lines.append("Connection: close")
        req_lines.append("\r\n")

        raw_req = "\r\n".join(req_lines).encode("utf-8") + body_bytes
        writer.write(raw_req)
        await writer.drain()

        # Read HTTP status line
        status_line = await reader.readline()
        if not status_line:
            writer.close()
            raise LlmConnectionError("Server closed connection without HTTP response")

        status_parts = status_line.decode("utf-8", errors="replace").split()
        if len(status_parts) < 2:
            writer.close()
            raise LlmConnectionError(f"Malformed HTTP response: {status_line!r}")

        try:
            status_code = int(status_parts[1])
        except ValueError as exc:
            writer.close()
            raise LlmConnectionError(f"Invalid HTTP status code in: {status_line!r}") from exc

        # Check status codes
        if status_code in (401, 403):
            writer.close()
            raise LlmAuthError(f"HTTP {status_code}: Unauthorized or Forbidden")
        if status_code == 429:
            writer.close()
            raise LlmRateLimitError("HTTP 429: Rate limit exceeded")
        if status_code >= 400:
            writer.close()
            raise LlmConnectionError(f"HTTP {status_code}: Server error")

        # Read headers
        is_chunked = False
        while True:
            header_line = await reader.readline()
            if header_line in (b"\r\n", b"\n", b""):
                break
            if (
                header_line.lower().startswith(b"transfer-encoding:")
                and b"chunked" in header_line.lower()
            ):
                is_chunked = True

        async def _generator() -> AsyncIterator[bytes]:
            try:
                if is_chunked:
                    while not self._aborted and not cancel.cancelled:
                        chunk_header = await reader.readline()
                        if not chunk_header:
                            break
                        hex_len = chunk_header.strip().split(b";")[0]
                        try:
                            chunk_len = int(hex_len, 16)
                        except ValueError as exc:
                            raise LlmStreamError(f"Invalid chunk length: {chunk_header!r}") from exc
                        if chunk_len == 0:
                            await reader.readline()  # trailing CRLF
                            break
                        chunk_data = await reader.readexactly(chunk_len)
                        await reader.readline()  # CRLF
                        yield chunk_data
                else:
                    while not self._aborted and not cancel.cancelled:
                        chunk = await reader.read(4096)
                        if not chunk:
                            break
                        yield chunk
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

        return _generator()

    async def abort(self) -> None:
        self._aborted = True
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass


class RecordedFixtureTransport(EvonTransport):
    """Transport that replays recorded wire fixture (.jsonl) frames."""

    def __init__(
        self,
        fixture_path: str | Path,
        *,
        chunk_size: int | None = None,
    ) -> None:
        self._path = Path(fixture_path)
        self._chunk_size = chunk_size
        self._aborted = False
        self.closed = False

    async def connect_and_stream(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
        cancel: CancelToken,
        on_close: Callable[[], None] | None = None,
    ) -> AsyncIterator[bytes]:
        if not self._path.exists():
            raise LlmConnectionError(f"Fixture file not found: {self._path}")

        lines = self._path.read_text(encoding="utf-8").splitlines()

        async def _generator() -> AsyncIterator[bytes]:
            for line in lines:
                if self._aborted or cancel.cancelled:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("dir") == "in":
                    raw_payload = record.get("payload", "")
                    payload_bytes = (raw_payload + "\n").encode("utf-8")

                    if self._chunk_size is not None and self._chunk_size > 0:
                        for i in range(0, len(payload_bytes), self._chunk_size):
                            if self._aborted or cancel.cancelled:
                                break
                            yield payload_bytes[i : i + self._chunk_size]
                    else:
                        yield payload_bytes

        return _generator()

    async def abort(self) -> None:
        self._aborted = True
        self.closed = True


class ByteStreamTransport(EvonTransport):
    """Transport that yields byte chunks from an iterable, async iterable, or factory."""

    def __init__(
        self,
        chunks: Sequence[bytes]
        | AsyncIterator[bytes]
        | Callable[[], AsyncIterator[bytes] | Sequence[bytes]],
    ) -> None:
        self._chunks = chunks
        self._aborted = False
        self.closed = False

    async def connect_and_stream(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
        cancel: CancelToken,
        on_close: Callable[[], None] | None = None,
    ) -> AsyncIterator[bytes]:
        async def _generator() -> AsyncIterator[bytes]:
            chunks_source = self._chunks() if callable(self._chunks) else self._chunks

            if isinstance(chunks_source, AsyncIterator):
                async for chunk in chunks_source:
                    if self._aborted or cancel.cancelled:
                        break
                    yield chunk
            else:
                for chunk in chunks_source:
                    if self._aborted or cancel.cancelled:
                        break
                    yield chunk

        return _generator()

    async def abort(self) -> None:
        self._aborted = True
        self.closed = True


# =============================================================================
# EvonLlmStream Implementation
# =============================================================================


class EvonLlmStream(LlmStream):
    """Active streaming LLM inference session fulfilling LlmStream protocol."""

    def __init__(
        self,
        byte_stream: AsyncIterator[bytes],
        transport: EvonTransport,
        *,
        epoch: int,
        cancel: CancelToken,
        clock: Clock,
        config: EvonConfig,
        telemetry: TelemetrySink,
    ) -> None:
        self._byte_stream = byte_stream
        self._transport = transport
        self.epoch = epoch
        self._cancel = cancel
        self._clock = clock
        self._config = config
        self._telemetry = telemetry

        self._aborted = False
        self._abort_event = asyncio.Event()
        self._consumed = False

        if cancel.cancelled:
            self._aborted = True
            self._abort_event.set()

        self._unregister_cancel = cancel.add_callback(self._on_cancel)

    def _on_cancel(self) -> None:
        self._aborted = True
        self._abort_event.set()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._transport.abort())
        except RuntimeError:
            pass

    async def abort(self) -> None:
        """Abort inference stream. Idempotent; closes HTTP/WS stream."""
        if self._aborted:
            await self._transport.abort()
            return

        self._aborted = True
        self._abort_event.set()

        if self._unregister_cancel is not None:
            try:
                self._unregister_cancel()
            except Exception:
                pass

        self._telemetry.emit("llm.abort", ts_ns=self._clock.now_ns(), epoch=self.epoch)
        await self._transport.abort()

    async def _next_chunk_with_timeout(
        self,
        stream: AsyncIterator[bytes],
        timeout_s: float,
    ) -> bytes | None:
        if self._aborted or self._cancel.cancelled:
            return None

        loop = asyncio.get_running_loop()
        timeout_fut: asyncio.Future[None] = loop.create_future()

        def _on_token_timeout() -> None:
            if not timeout_fut.done():
                timeout_fut.set_exception(
                    LlmTimeoutError(f"LLM token timeout exceeded ({timeout_s}s)")
                )

        timeout_handle = self._clock.call_later(timeout_s * 1000.0, _on_token_timeout)

        async def _fetch_next() -> bytes:
            return await stream.__anext__()

        abort_task = asyncio.create_task(self._abort_event.wait())
        next_chunk_task: asyncio.Task[bytes] = asyncio.create_task(_fetch_next())
        wait_targets: list[asyncio.Future[Any]] = [next_chunk_task, timeout_fut, abort_task]

        try:
            done, _ = await asyncio.wait(
                wait_targets,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if abort_task in done:
                next_chunk_task.cancel()
                return None

            if timeout_fut in done:
                next_chunk_task.cancel()
                timeout_fut.result()

            try:
                result: bytes = next_chunk_task.result()
                return result
            except StopAsyncIteration:
                return None
            except Exception as exc:
                if isinstance(exc, LlmError):
                    raise
                raise LlmStreamError(f"Stream error: {exc}") from exc

        finally:
            self._clock.cancel(timeout_handle)
            if not abort_task.done():
                abort_task.cancel()
            if not next_chunk_task.done():
                next_chunk_task.cancel()

    def _parse_sse_line(self, line: str) -> str | None:
        line = line.strip()
        if not line or line.startswith(":"):
            return None

        if line.startswith("data:"):
            data_str = line[5:].strip()
        else:
            data_str = line

        if data_str == "[DONE]":
            return None

        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError as exc:
            if line.startswith("data:"):
                raise LlmStreamError(f"Malformed SSE JSON payload: {data_str}") from exc
            return None

        if isinstance(payload, dict):
            if "error" in payload:
                err_data = payload["error"]
                if isinstance(err_data, dict):
                    err_msg = err_data.get("message", json.dumps(err_data))
                else:
                    err_msg = str(err_data)
                raise LlmStreamError(f"Upstream stream error: {err_msg}")

            choices = payload.get("choices")
            if choices and isinstance(choices, list):
                choice = choices[0]
                if isinstance(choice, dict):
                    delta = choice.get("delta")
                    if isinstance(delta, dict):
                        content = delta.get("content")
                        if content:
                            return str(content)

        return None

    async def deltas(self) -> AsyncIterator[str]:
        """Asynchronous iterator yielding generated text tokens/deltas."""
        if self._consumed or self._aborted or self._cancel.cancelled:
            return
        self._consumed = True

        decoder = codecs.getincrementaldecoder("utf-8")()
        text_buffer = ""
        first_token_received = False
        start_ns = self._clock.now_ns()

        try:
            while not self._aborted and not self._cancel.cancelled:
                timeout_s = (
                    self._config.first_token_timeout_s
                    if not first_token_received
                    else self._config.token_timeout_s
                )

                chunk = await self._next_chunk_with_timeout(self._byte_stream, timeout_s)
                if chunk is None:
                    break

                try:
                    text_chunk = decoder.decode(chunk, final=False)
                except UnicodeDecodeError as exc:
                    raise LlmStreamError(f"UTF-8 decode error: {exc}") from exc

                text_buffer += text_chunk

                while "\n" in text_buffer:
                    line, text_buffer = text_buffer.split("\n", 1)
                    line = line.strip("\r")
                    if not line:
                        continue

                    delta = self._parse_sse_line(line)
                    if delta is not None:
                        if not first_token_received:
                            first_token_received = True
                            latency_ms = (self._clock.now_ns() - start_ns) / 1_000_000
                            self._telemetry.emit(
                                "llm.first_token",
                                ts_ns=self._clock.now_ns(),
                                epoch=self.epoch,
                                latency_ms=latency_ms,
                            )
                        self._telemetry.emit(
                            "llm.token",
                            ts_ns=self._clock.now_ns(),
                            epoch=self.epoch,
                            length=len(delta),
                        )
                        yield delta

            if not self._aborted and not self._cancel.cancelled:
                try:
                    remaining = decoder.decode(b"", final=True)
                except UnicodeDecodeError as exc:
                    raise LlmStreamError(f"UTF-8 decode error at stream end: {exc}") from exc
                text_buffer += remaining
                if text_buffer.strip():
                    delta = self._parse_sse_line(text_buffer.strip("\r"))
                    if delta is not None:
                        yield delta

        finally:
            await self.abort()
            self._telemetry.emit(
                "llm.complete",
                ts_ns=self._clock.now_ns(),
                epoch=self.epoch,
                aborted=self._aborted,
            )


# =============================================================================
# EvonLlmAdapter Port Implementation
# =============================================================================


class EvonLlmAdapter(LlmPort):
    """Evon / Gemini LLM streaming adapter fulfilling LlmPort protocol."""

    def __init__(
        self,
        config: EvonConfig | None = None,
        *,
        fixture_path: str | Path | None = None,
        transport: EvonTransport | None = None,
        clock: Clock | None = None,
        telemetry: TelemetrySink | None = None,
        api_key: str | None = None,
        endpoint_url: str | None = None,
        model: str | None = None,
        connect_timeout_s: float | None = None,
        first_token_timeout_s: float | None = None,
        token_timeout_s: float | None = None,
    ) -> None:
        cfg = config or EvonConfig()
        resolved_api_key = (
            api_key
            or cfg.api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GNANI_API_KEY")
        )

        self._config = EvonConfig(
            api_key=resolved_api_key,
            endpoint_url=endpoint_url or cfg.endpoint_url,
            model=model or cfg.model,
            user_agent=cfg.user_agent,
            connect_timeout_s=(
                connect_timeout_s if connect_timeout_s is not None else cfg.connect_timeout_s
            ),
            first_token_timeout_s=(
                first_token_timeout_s
                if first_token_timeout_s is not None
                else cfg.first_token_timeout_s
            ),
            token_timeout_s=(
                token_timeout_s if token_timeout_s is not None else cfg.token_timeout_s
            ),
        )

        self._clock: Clock = clock or MonotonicClock()
        self._telemetry: TelemetrySink = telemetry or NullSink()
        self._fixture_path = fixture_path
        self._custom_transport = transport

    @classmethod
    def from_fixture(
        cls,
        fixture_path: str | Path,
        config: EvonConfig | None = None,
        clock: Clock | None = None,
        telemetry: TelemetrySink | None = None,
    ) -> EvonLlmAdapter:
        """Create adapter configured to replay recorded wire fixture (.jsonl)."""
        return cls(
            config=config,
            fixture_path=fixture_path,
            clock=clock,
            telemetry=telemetry,
        )

    async def _connect_with_timeout(
        self,
        transport: EvonTransport,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
        cancel: CancelToken,
    ) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        timeout_fut: asyncio.Future[None] = loop.create_future()

        def _on_connect_timeout() -> None:
            if not timeout_fut.done():
                timeout_fut.set_exception(
                    LlmTimeoutError(f"LLM connect timeout exceeded ({timeout_s}s)")
                )

        handle = self._clock.call_later(timeout_s * 1000.0, _on_connect_timeout)

        connect_task: asyncio.Task[AsyncIterator[bytes]] = asyncio.create_task(
            transport.connect_and_stream(
                url=url,
                headers=headers,
                payload=payload,
                timeout_s=timeout_s,
                cancel=cancel,
            )
        )
        wait_targets: list[asyncio.Future[Any]] = [connect_task, timeout_fut]

        try:
            done, _ = await asyncio.wait(
                wait_targets,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if timeout_fut in done:
                connect_task.cancel()
                await transport.abort()
                timeout_fut.result()

            return connect_task.result()
        finally:
            self._clock.cancel(handle)
            if not connect_task.done():
                connect_task.cancel()

    async def stream(
        self, messages: Sequence[Message], *, epoch: int, cancel: CancelToken
    ) -> LlmStream:
        """Open an LLM completion stream for the given messages."""
        payload_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg.role not in ("system", "user", "assistant"):
                raise ValueError(
                    f"Unsupported role {msg.role!r}. Allowed roles: system, user, assistant"
                )
            payload_messages.append({"role": msg.role, "content": msg.content})

        payload = {
            "model": self._config.model,
            "messages": payload_messages,
            "stream": True,
        }

        # Select transport
        transport: EvonTransport
        if self._custom_transport is not None:
            transport = self._custom_transport
        elif self._fixture_path is not None:
            transport = RecordedFixtureTransport(self._fixture_path)
        else:
            if not self._config.api_key:
                raise LlmAuthError(
                    "API key required: set GEMINI_API_KEY / GNANI_API_KEY or supply via config"
                )
            transport = HttpTransport()

        headers = {
            "Content-Type": "application/json",
            "User-Agent": self._config.user_agent,
            "Accept": "text/event-stream",
        }
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        if cancel.cancelled:
            # Short-circuit if already cancelled
            async def _empty_iter() -> AsyncIterator[bytes]:
                if False:
                    yield b""

            return EvonLlmStream(
                byte_stream=_empty_iter(),
                transport=transport,
                epoch=epoch,
                cancel=cancel,
                clock=self._clock,
                config=self._config,
                telemetry=self._telemetry,
            )

        try:
            byte_stream = await self._connect_with_timeout(
                transport,
                url=self._config.endpoint_url,
                headers=headers,
                payload=payload,
                timeout_s=self._config.connect_timeout_s,
                cancel=cancel,
            )
        except Exception:
            await transport.abort()
            raise

        self._telemetry.emit("llm.connect", ts_ns=self._clock.now_ns(), epoch=epoch)

        return EvonLlmStream(
            byte_stream=byte_stream,
            transport=transport,
            epoch=epoch,
            cancel=cancel,
            clock=self._clock,
            config=self._config,
            telemetry=self._telemetry,
        )
