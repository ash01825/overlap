"""Unit and contract tests for EvonLlmAdapter (Evon / Gemini LLM adapter).

Covers all done_when and failure_modes specified in TASKS.md T3.04:
- LlmContract fulfillment via recorded wire fixture replay
- Incremental UTF-8 decoding with Devanagari and Tamil split at every byte offset
- Abort idempotence and immediate closure of underlying connection
- Virtual-time configurable timeouts for connect, first token, and inter-token
- Typed LlmError mapping (auth, rate limit, connection, stream)
- Assistant history preservation with truncation markers
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

_helpers_dir = str(Path(__file__).resolve().parent.parent.parent / "helpers")
if _helpers_dir not in sys.path:
    sys.path.insert(0, _helpers_dir)

_contracts_dir = str(Path(__file__).resolve().parent.parent.parent / "contracts")
if _contracts_dir not in sys.path:
    sys.path.insert(0, _contracts_dir)

_root_dir = Path(__file__).resolve().parent.parent.parent.parent
_overlap_dir = str(_root_dir)
if _overlap_dir not in sys.path:
    sys.path.insert(0, _overlap_dir)

import overlap.adapters  # noqa: E402

_overlap_adapters_dir = str(_root_dir / "overlap" / "adapters")
if _overlap_adapters_dir not in overlap.adapters.__path__:
    overlap.adapters.__path__.insert(0, _overlap_adapters_dir)

from suites import LlmContract  # noqa: E402
from virtual import advance_virtual, run_virtual  # noqa: E402

from overlap.adapters.gnani.evon import (  # noqa: E402
    ByteStreamTransport,
    EvonConfig,
    EvonLlmAdapter,
    EvonTransport,
    LlmAuthError,
    LlmConnectionError,
    LlmError,
    LlmRateLimitError,
    LlmStreamError,
    LlmTimeoutError,
)
from overlap.core.clock import CancelToken, VirtualClock  # noqa: E402
from overlap.ports.llm import LlmPort, LlmStream, Message  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "wire" / "evon"
STREAM_FIXTURE = FIXTURES_DIR / "stream.jsonl"
ABORT_FIXTURE = FIXTURES_DIR / "abort.jsonl"


# =============================================================================
# 1. Contract Tests (LlmContract via recorded wire fixture)
# =============================================================================


class TestEvonLlmContract:
    """Verifies that EvonLlmAdapter strictly fulfills the LlmPort contract."""

    def test_implements_protocols(self) -> None:
        adapter = EvonLlmAdapter(fixture_path=STREAM_FIXTURE)
        assert isinstance(adapter, LlmPort)

    def test_llm_contract_stream_fixture(self) -> None:
        adapter = EvonLlmAdapter(fixture_path=STREAM_FIXTURE)
        LlmContract.verify_sync(adapter)

    def test_llm_contract_abort_fixture(self) -> None:
        adapter = EvonLlmAdapter(fixture_path=ABORT_FIXTURE)
        LlmContract.verify_sync(adapter)

    def test_fixture_replay_yields_expected_tokens(self) -> None:
        async def _run() -> None:
            adapter = EvonLlmAdapter(fixture_path=STREAM_FIXTURE)
            token = CancelToken()
            stream = await adapter.stream(
                [Message(role="user", content="Say 'hello world' and one sentence.")],
                epoch=1,
                cancel=token,
            )
            assert isinstance(stream, LlmStream)
            deltas: list[str] = []
            async for delta in stream.deltas():
                deltas.append(delta)

            full_text = "".join(deltas)
            assert "Hello world. The sun is shining today." in full_text

        run_virtual(_run())


# =============================================================================
# 2. Incremental UTF-8 Decoding Tests (Devanagari & Tamil byte splits)
# =============================================================================


class TestIncrementalUtf8Decoding:
    """Verifies incremental decoding without mojibake when bytes are split across chunks."""

    DEVANAGARI_TEXT = "नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।"
    TAMIL_TEXT = "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்? இன்று வானிலை நன்றாக உள்ளது."

    def _create_sse_payload(self, text: str) -> bytes:
        payload = {
            "choices": [
                {
                    "delta": {"content": text, "role": "assistant"},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "model": "gemini-2.5-flash",
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode()

    def test_devanagari_and_tamil_split_at_every_byte_offset(self) -> None:
        combined_text = f"{self.DEVANAGARI_TEXT} — {self.TAMIL_TEXT}"
        raw_bytes = self._create_sse_payload(combined_text)

        async def _run_all() -> None:
            for split_idx in range(len(raw_bytes)):
                part1 = raw_bytes[:split_idx]
                part2 = raw_bytes[split_idx:]

                transport = ByteStreamTransport([part1, part2])
                adapter = EvonLlmAdapter(transport=transport)
                stream = await adapter.stream(
                    [Message(role="user", content="test")], epoch=1, cancel=CancelToken()
                )
                collected: list[str] = []
                async for d in stream.deltas():
                    collected.append(d)

                result = "".join(collected)
                assert "\ufffd" not in result, f"Mojibake detected at split {split_idx}"
                assert result == combined_text, (
                    f"Mismatch at split {split_idx}: {result!r} != {combined_text!r}"
                )

        run_virtual(_run_all())

    def test_single_byte_chunks_streaming(self) -> None:
        """Stream the entire Indic text 1 byte at a time."""
        combined_text = f"{self.DEVANAGARI_TEXT} — {self.TAMIL_TEXT}"
        raw_bytes = self._create_sse_payload(combined_text)
        single_byte_chunks = [raw_bytes[i : i + 1] for i in range(len(raw_bytes))]

        async def _run() -> None:
            transport = ByteStreamTransport(single_byte_chunks)
            adapter = EvonLlmAdapter(transport=transport)
            stream = await adapter.stream(
                [Message(role="user", content="test")], epoch=1, cancel=CancelToken()
            )
            collected: list[str] = []
            async for d in stream.deltas():
                collected.append(d)

            result = "".join(collected)
            assert "\ufffd" not in result
            assert result == combined_text

        run_virtual(_run())

    def test_multi_byte_chunk_sizes(self) -> None:
        """Stream chunks of 2, 3, 5, 7 bytes straddling Indic multi-byte character boundaries."""
        combined_text = f"{self.DEVANAGARI_TEXT} | {self.TAMIL_TEXT}"
        raw_bytes = self._create_sse_payload(combined_text)

        async def _run_all() -> None:
            for chunk_size in [2, 3, 5, 7, 11]:
                chunks = [
                    raw_bytes[i : i + chunk_size] for i in range(0, len(raw_bytes), chunk_size)
                ]
                transport = ByteStreamTransport(chunks)
                adapter = EvonLlmAdapter(transport=transport)
                stream = await adapter.stream(
                    [Message(role="user", content="test")], epoch=1, cancel=CancelToken()
                )
                collected = [d async for d in stream.deltas()]
                result = "".join(collected)
                assert "\ufffd" not in result
                assert result == combined_text

        run_virtual(_run_all())


# =============================================================================
# 3. Abort Semantics & Idempotence Tests
# =============================================================================


class TestAbortSemantics:
    """Verifies that abort() is idempotent, closes underlying stream, and stops delivery."""

    def test_abort_is_idempotent(self) -> None:
        async def _run() -> None:
            transport = ByteStreamTransport([b"data: [DONE]\n\n"])
            adapter = EvonLlmAdapter(transport=transport)
            stream = await adapter.stream(
                [Message(role="user", content="test")], epoch=1, cancel=CancelToken()
            )
            await stream.abort()
            await stream.abort()
            await stream.abort()
            assert transport.closed

        run_virtual(_run())

    def test_abort_stops_tokens_mid_stream(self) -> None:
        async def _run() -> None:
            chunks = [
                b'data: {"choices":[{"delta":{"content":"Token1 "}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"Token2 "}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"Token3 "}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
            transport = ByteStreamTransport(chunks)
            adapter = EvonLlmAdapter(transport=transport)
            stream = await adapter.stream(
                [Message(role="user", content="test")], epoch=1, cancel=CancelToken()
            )

            delivered: list[str] = []
            async for token in stream.deltas():
                delivered.append(token)
                if len(delivered) == 1:
                    await stream.abort()

            assert delivered == ["Token1 "]
            assert transport.closed

        run_virtual(_run())

    def test_cancel_token_stops_delivery(self) -> None:
        async def _run() -> None:
            token = CancelToken()
            chunks = [
                b'data: {"choices":[{"delta":{"content":"Token1 "}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"Token2 "}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
            transport = ByteStreamTransport(chunks)
            adapter = EvonLlmAdapter(transport=transport)
            stream = await adapter.stream(
                [Message(role="user", content="test")], epoch=1, cancel=token
            )

            delivered: list[str] = []
            async for tok in stream.deltas():
                delivered.append(tok)
                token.cancel()

            assert delivered == ["Token1 "]
            assert transport.closed

        run_virtual(_run())

    def test_already_cancelled_token_yields_no_tokens(self) -> None:
        async def _run() -> None:
            token = CancelToken()
            token.cancel()
            transport = ByteStreamTransport([b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'])
            adapter = EvonLlmAdapter(transport=transport)
            stream = await adapter.stream(
                [Message(role="user", content="test")], epoch=1, cancel=token
            )
            delivered = [tok async for tok in stream.deltas()]
            assert len(delivered) == 0

        run_virtual(_run())


# =============================================================================
# 4. Timeout Tests in Virtual Time
# =============================================================================


class HangingTransport(EvonTransport):
    """Transport that never completes connection or yields chunks."""

    def __init__(self, hang_on_connect: bool = True) -> None:
        self.hang_on_connect = hang_on_connect
        self.closed = False

    async def connect_and_stream(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
        cancel: CancelToken,
        on_close: Any = None,
    ) -> AsyncIterator[bytes]:
        if self.hang_on_connect:
            # Hang forever until cancelled
            await asyncio.Event().wait()

        async def _hang_iter() -> AsyncIterator[bytes]:
            await asyncio.Event().wait()
            yield b""  # pragma: no cover

        return _hang_iter()

    async def abort(self) -> None:
        self.closed = True


class TestVirtualTimeouts:
    """Verifies connect and token timeouts in deterministic virtual time."""

    def test_connect_timeout_virtual_time(self) -> None:
        async def _run() -> None:
            clock = VirtualClock()
            transport = HangingTransport(hang_on_connect=True)
            cfg = EvonConfig(connect_timeout_s=2.0)
            adapter = EvonLlmAdapter(config=cfg, clock=clock, transport=transport)

            stream_task = asyncio.create_task(
                adapter.stream([Message(role="user", content="hi")], epoch=1, cancel=CancelToken())
            )
            # Yield to event loop so stream_task enters _connect_with_timeout
            await asyncio.sleep(0)

            # Advance virtual time to trip the connect timeout
            await advance_virtual(clock, 2000)

            with pytest.raises(LlmTimeoutError, match="connect timeout"):
                await stream_task

            assert transport.closed

        run_virtual(_run())

    def test_first_token_timeout_virtual_time(self) -> None:
        async def _run() -> None:
            clock = VirtualClock()
            # Connect succeeds, but streaming body hangs
            transport = HangingTransport(hang_on_connect=False)
            cfg = EvonConfig(first_token_timeout_s=2.0)
            adapter = EvonLlmAdapter(config=cfg, clock=clock, transport=transport)

            stream = await adapter.stream(
                [Message(role="user", content="hi")], epoch=1, cancel=CancelToken()
            )

            async def _consume() -> None:
                async for _ in stream.deltas():
                    pass

            consume_task = asyncio.create_task(_consume())
            # Yield to loop so deltas() starts waiting for first chunk
            await asyncio.sleep(0)

            # Advance virtual clock past first-token timeout
            await advance_virtual(clock, 2000)

            with pytest.raises(LlmTimeoutError, match="token timeout"):
                await consume_task

            assert transport.closed

        run_virtual(_run())

    def test_token_interval_timeout_virtual_time(self) -> None:
        """First token arrives, but subsequent token hangs past token_timeout_s."""

        class FirstTokenThenHangTransport(EvonTransport):
            def __init__(self) -> None:
                self.closed = False

            async def connect_and_stream(self, **kwargs: Any) -> AsyncIterator[bytes]:
                async def _gen() -> AsyncIterator[bytes]:
                    yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
                    await asyncio.Event().wait()

                return _gen()

            async def abort(self) -> None:
                self.closed = True

        async def _run() -> None:
            clock = VirtualClock()
            transport = FirstTokenThenHangTransport()
            cfg = EvonConfig(first_token_timeout_s=5.0, token_timeout_s=2.0)
            adapter = EvonLlmAdapter(config=cfg, clock=clock, transport=transport)

            stream = await adapter.stream(
                [Message(role="user", content="hi")], epoch=1, cancel=CancelToken()
            )

            deltas: list[str] = []

            async def _consume() -> None:
                async for tok in stream.deltas():
                    deltas.append(tok)

            consume_task = asyncio.create_task(_consume())
            # Let loop deliver first token
            for _ in range(10):
                if deltas:
                    break
                await asyncio.sleep(0)
            assert deltas == ["first"]

            # Give one more turn so deltas() starts waiting for the second chunk
            for _ in range(5):
                await asyncio.sleep(0)

            # Advance clock past token_timeout_s (2000 ms)
            await advance_virtual(clock, 2000)

            with pytest.raises(LlmTimeoutError, match="token timeout"):
                await consume_task

            assert transport.closed

        run_virtual(_run())


# =============================================================================
# 5. Typed Error Mapping Tests
# =============================================================================


class ErrorTransport(EvonTransport):
    """Transport that simulates HTTP status or mid-stream errors."""

    def __init__(
        self,
        *,
        error_to_raise: Exception | None = None,
        raw_error_payload: bytes | None = None,
    ) -> None:
        self.error_to_raise = error_to_raise
        self.raw_error_payload = raw_error_payload
        self.closed = False

    async def connect_and_stream(self, **kwargs: Any) -> AsyncIterator[bytes]:
        if self.error_to_raise is not None:
            raise self.error_to_raise

        async def _gen() -> AsyncIterator[bytes]:
            if self.raw_error_payload is not None:
                yield self.raw_error_payload

        return _gen()

    async def abort(self) -> None:
        self.closed = True


class TestErrorMapping:
    """Verifies that wire/HTTP errors map to typed LlmError subclasses."""

    def test_all_errors_inherit_from_llm_error(self) -> None:
        for err_cls in (
            LlmConnectionError,
            LlmAuthError,
            LlmRateLimitError,
            LlmTimeoutError,
            LlmStreamError,
        ):
            assert issubclass(err_cls, LlmError)

    def test_http_401_maps_to_auth_error(self) -> None:
        async def _run() -> None:
            transport = ErrorTransport(error_to_raise=LlmAuthError("HTTP 401: Unauthorized"))
            adapter = EvonLlmAdapter(transport=transport)
            with pytest.raises(LlmAuthError, match="401"):
                await adapter.stream(
                    [Message(role="user", content="hi")], epoch=1, cancel=CancelToken()
                )

        run_virtual(_run())

    def test_http_403_maps_to_auth_error(self) -> None:
        async def _run() -> None:
            transport = ErrorTransport(error_to_raise=LlmAuthError("HTTP 403: Forbidden"))
            adapter = EvonLlmAdapter(transport=transport)
            with pytest.raises(LlmAuthError, match="403"):
                await adapter.stream(
                    [Message(role="user", content="hi")], epoch=1, cancel=CancelToken()
                )

        run_virtual(_run())

    def test_http_429_maps_to_rate_limit_error(self) -> None:
        async def _run() -> None:
            transport = ErrorTransport(
                error_to_raise=LlmRateLimitError("HTTP 429: Too Many Requests")
            )
            adapter = EvonLlmAdapter(transport=transport)
            with pytest.raises(LlmRateLimitError, match="429"):
                await adapter.stream(
                    [Message(role="user", content="hi")], epoch=1, cancel=CancelToken()
                )

        run_virtual(_run())

    def test_mid_stream_json_error_surfaced_no_hang(self) -> None:
        """Mid-stream JSON error payload must raise LlmStreamError, not hang."""
        err_payload = (
            b'data: {"choices":[{"delta":{"content":"Start"}}]}\n\n'
            b'data: {"error":{"message":"Resource has been exhausted","code":429}}\n\n'
        )

        async def _run() -> None:
            transport = ErrorTransport(raw_error_payload=err_payload)
            adapter = EvonLlmAdapter(transport=transport)
            stream = await adapter.stream(
                [Message(role="user", content="hi")], epoch=1, cancel=CancelToken()
            )

            deltas: list[str] = []
            with pytest.raises(LlmStreamError, match="Resource has been exhausted"):
                async for tok in stream.deltas():
                    deltas.append(tok)

            assert deltas == ["Start"]
            assert transport.closed

        run_virtual(_run())

    def test_malformed_sse_payload_raises_stream_error(self) -> None:
        bad_payload = b"data: {this is not valid json}\n\n"

        async def _run() -> None:
            transport = ErrorTransport(raw_error_payload=bad_payload)
            adapter = EvonLlmAdapter(transport=transport)
            stream = await adapter.stream(
                [Message(role="user", content="hi")], epoch=1, cancel=CancelToken()
            )
            with pytest.raises(LlmStreamError, match="Malformed"):
                async for _ in stream.deltas():
                    pass

        run_virtual(_run())


# =============================================================================
# 6. Assistant History & Truncation Marker Tests
# =============================================================================


class RecordingTransport(EvonTransport):
    """Captures the serialized HTTP request payload."""

    def __init__(self) -> None:
        self.recorded_payload: dict[str, Any] | None = None
        self.recorded_headers: dict[str, str] | None = None
        self.closed = False

    async def connect_and_stream(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
        cancel: CancelToken,
        on_close: Any = None,
    ) -> AsyncIterator[bytes]:
        self.recorded_payload = payload
        self.recorded_headers = headers

        async def _gen() -> AsyncIterator[bytes]:
            yield b"data: [DONE]\n\n"

        return _gen()

    async def abort(self) -> None:
        self.closed = True


class TestAssistantHistoryAndTruncation:
    """Verifies that assistant messages with truncation markers are preserved intact."""

    def test_assistant_history_with_truncation_marker_preserved(self) -> None:
        transport = RecordingTransport()
        adapter = EvonLlmAdapter(transport=transport)

        messages = [
            Message(role="system", content="You are a concise voice assistant."),
            Message(role="user", content="नमस्ते, आप कौन हैं?"),
            Message(
                role="assistant",
                content="नमस्ते! मैं आपका सहायक हूँ। मैं आपकी कैसे मदद... [cut]",
            ),
            Message(role="user", content="मौसम कैसा है?"),
        ]

        async def _run() -> None:
            stream = await adapter.stream(messages, epoch=1, cancel=CancelToken())
            async for _ in stream.deltas():
                pass

        run_virtual(_run())

        payload = transport.recorded_payload
        assert payload is not None
        assert payload["model"] == "gemini-2.5-flash"
        assert payload["stream"] is True

        payload_msgs = payload["messages"]
        assert len(payload_msgs) == 4

        # Verify assistant message preserved its role and exact truncated text
        assistant_msg = payload_msgs[2]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "नमस्ते! मैं आपका सहायक हूँ। मैं आपकी कैसे मदद... [cut]"

    def test_invalid_role_rejected(self) -> None:
        adapter = EvonLlmAdapter(fixture_path=STREAM_FIXTURE)
        bad_message: Any = Message(role="unknown", content="bad role")  # type: ignore[arg-type]

        async def _run() -> None:
            with pytest.raises(ValueError, match="Unsupported role"):
                await adapter.stream([bad_message], epoch=1, cancel=CancelToken())

        run_virtual(_run())


# =============================================================================
# 7. Configuration & Defaults Tests
# =============================================================================


class TestEvonConfigAndDefaults:
    """Verifies default parameters and configuration overrides per docs/WIRE.md."""

    def test_defaults(self) -> None:
        cfg = EvonConfig()
        assert cfg.model == "gemini-2.5-flash"
        assert cfg.user_agent == "overlap-client/0.1.0"
        assert (
            cfg.endpoint_url
            == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        )
        assert cfg.connect_timeout_s == 2.0
        assert cfg.first_token_timeout_s == 2.0
        assert cfg.token_timeout_s == 5.0

    def test_wire_headers_supplied(self) -> None:
        transport = RecordingTransport()
        adapter = EvonLlmAdapter(
            api_key="test-key-12345",
            transport=transport,
        )

        async def _run() -> None:
            await adapter.stream(
                [Message(role="user", content="hi")], epoch=1, cancel=CancelToken()
            )

        run_virtual(_run())

        headers = transport.recorded_headers
        assert headers is not None
        assert headers["Authorization"] == "Bearer test-key-12345"
        assert headers["User-Agent"] == "overlap-client/0.1.0"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "text/event-stream"
