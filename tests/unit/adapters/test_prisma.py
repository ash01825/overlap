"""Unit and contract tests for PrismaSttPort and PrismaSttStream using FakeWebSocket."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_contracts_dir = str(Path(__file__).resolve().parent.parent.parent / "contracts")
if _contracts_dir not in sys.path:
    sys.path.insert(0, _contracts_dir)

_fakes_dir = str(Path(__file__).resolve().parent.parent.parent / "fakes")
if _fakes_dir not in sys.path:
    sys.path.insert(0, _fakes_dir)

_helpers_dir = str(Path(__file__).resolve().parent.parent.parent / "helpers")
if _helpers_dir not in sys.path:
    sys.path.insert(0, _helpers_dir)

from sink import RecordingSink  # noqa: E402
from suites import SttContract  # noqa: E402
from virtual import run_virtual  # noqa: E402
from ws import FakeWebSocket  # noqa: E402

from overlap.adapters.gnani.prisma import PrismaSttPort  # noqa: E402
from overlap.core.clock import CancelToken, VirtualClock  # noqa: E402
from overlap.core.types import AudioFrame, SttEvent  # noqa: E402
from overlap.ports.stt import SttPort  # noqa: E402

_FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "wire" / "prisma"


def _make_fake_ws_factory(fixture_path: Path, clock: VirtualClock | None = None) -> Any:
    """Create a factory returning a FakeWebSocket pre-loaded with a JSONL fixture."""

    def factory(*args: Any, **kwargs: Any) -> FakeWebSocket:
        ws = FakeWebSocket(clock=clock)
        ws.load_jsonl(fixture_path)
        return ws

    return factory


# =============================================================================
# SttContract Test Suite Implementation
# =============================================================================


class TestPrismaContract(SttContract):
    """Reusable contract suite verifying PrismaSttPort against SttContract."""

    @pytest.fixture
    def stt_port(self) -> SttPort:
        fixture_path = _FIXTURE_DIR / "short_utterance.jsonl"
        factory = _make_fake_ws_factory(fixture_path)
        return PrismaSttPort(ws_factory=factory, api_key="test-key")


# =============================================================================
# Replay Tests on Real Recorded Fixtures
# =============================================================================


def test_no_assumed_fixtures_used() -> None:
    """Verify all Prisma recorded fixtures are production-captured (assumed: false)."""
    fixtures = ["short_utterance.jsonl", "code_switch.jsonl", "silence.jsonl"]
    for fname in fixtures:
        fpath = _FIXTURE_DIR / fname
        assert fpath.exists(), f"Fixture {fname} missing"
        ws = FakeWebSocket()
        ws.load_jsonl(fpath)
        assert ws.is_assumed is False, f"Fixture {fname} is flagged as assumed!"


def test_prisma_short_utterance_replay() -> None:
    """Replay short_utterance.jsonl and verify partials, final, and language."""

    async def _run() -> None:
        fixture_path = _FIXTURE_DIR / "short_utterance.jsonl"
        factory = _make_fake_ws_factory(fixture_path)
        port = PrismaSttPort(ws_factory=factory, api_key="test-key")
        token = CancelToken()
        stream = await port.open(cancel=token)

        events: list[SttEvent] = []
        async for evt in stream.events():
            events.append(evt)

        await stream.close()

        # From short_utterance.jsonl: 3 partials, 1 final
        partials = [e for e in events if e.kind == "partial"]
        finals = [e for e in events if e.kind == "final"]

        assert len(partials) == 3
        assert len(finals) == 1

        assert partials[0].text == "नमस्ते"
        assert partials[1].text == "नमस्ते आप"
        assert partials[2].text == "नमस्ते आप कैसे"

        assert finals[0].text == "नमस्ते, आप कैसे हैं?"
        assert finals[0].lang == "hi-IN"

        # Check monotonic timestamps
        for i in range(1, len(events)):
            assert events[i].ts_ns >= events[i - 1].ts_ns

    run_virtual(_run())


def test_prisma_code_switch_language_detection() -> None:
    """Replay code_switch.jsonl and verify multilingual code-switching detection."""

    async def _run() -> None:
        fixture_path = _FIXTURE_DIR / "code_switch.jsonl"
        factory = _make_fake_ws_factory(fixture_path)
        port = PrismaSttPort(ws_factory=factory, api_key="test-key", language_code="indic-auto")
        token = CancelToken()
        stream = await port.open(cancel=token)

        events: list[SttEvent] = []
        async for evt in stream.events():
            events.append(evt)

        await stream.close()

        partials = [e for e in events if e.kind == "partial"]
        finals = [e for e in events if e.kind == "final"]

        assert len(partials) == 5
        assert len(finals) == 1

        final = finals[0]
        assert final.text == "हैन, मीटिंग कैंसिल हो गई है।"
        assert final.lang == "hi-IN"

    run_virtual(_run())


def test_prisma_silence_session() -> None:
    """Replay silence.jsonl and verify no transcripts emitted for pure silence."""

    async def _run() -> None:
        fixture_path = _FIXTURE_DIR / "silence.jsonl"
        factory = _make_fake_ws_factory(fixture_path)
        port = PrismaSttPort(ws_factory=factory, api_key="test-key")
        token = CancelToken()
        stream = await port.open(cancel=token)

        events: list[SttEvent] = []
        async for evt in stream.events():
            events.append(evt)

        await stream.close()

        assert len(events) == 0

    run_virtual(_run())


# =============================================================================
# Failure Modes & Resilience Tests
# =============================================================================


def test_send_audio_bounded_queue_drop_oldest_telemetry() -> None:
    """Verify bounded send queue drops oldest on overflow and emits telemetry."""

    async def _run() -> None:
        telemetry = RecordingSink()
        ws = FakeWebSocket()
        port = PrismaSttPort(
            ws_factory=lambda *a, **kw: ws,
            api_key="test-key",
            telemetry=telemetry,
            max_queue_frames=5,
        )
        token = CancelToken()
        stream = await port.open(cancel=token)

        # Rapidly enqueue 15 frames without reading
        for i in range(15):
            frame = AudioFrame(pcm=bytes([i % 256]) * 640, sample_rate=16000, ts_ns=i * 20_000_000)
            await stream.send_audio(frame)

        # Allow send loop a turn
        await asyncio.sleep(0.01)
        await stream.close()

        # Total frames sent to stream = 15, queue capacity = 5
        # At least 10 frames must have been dropped
        assert stream.dropped_frames_total >= 10
        dropped_events = telemetry.events_named("stt.dropped_frames")
        assert len(dropped_events) > 0

    run_virtual(_run())


def test_out_of_order_and_late_partials_dropped() -> None:
    """Verify that partials delivered out of order or after final are dropped."""

    async def _run() -> None:
        ws = FakeWebSocket()
        port = PrismaSttPort(ws_factory=lambda *a, **kw: ws, api_key="test-key")
        token = CancelToken()
        stream = await port.open(cancel=token)

        # Simulate out-of-order partials followed by final and late partial
        messages = [
            # partial 2 arrives first
            {
                "t_ms": 100,
                "dir": "in",
                "kind": "text",
                "payload": json.dumps(
                    {
                        "type": "partial_transcript",
                        "utterance_id": "u1",
                        "seq": 2,
                        "text": "partial two",
                        "audio_ms": 200,
                    }
                ),
            },
            # partial 1 arrives late (out of order, seq 1 < 2) -> must be dropped
            {
                "t_ms": 150,
                "dir": "in",
                "kind": "text",
                "payload": json.dumps(
                    {
                        "type": "partial_transcript",
                        "utterance_id": "u1",
                        "seq": 1,
                        "text": "partial one",
                        "audio_ms": 100,
                    }
                ),
            },
            # final transcript arrives
            {
                "t_ms": 200,
                "dir": "in",
                "kind": "text",
                "payload": json.dumps(
                    {
                        "type": "final_transcript",
                        "utterance_id": "u1",
                        "text": "final one",
                        "detected_language_code": "en-IN",
                        "audio_duration_ms": 500,
                    }
                ),
            },
            # partial 3 arrives after final -> must be dropped
            {
                "t_ms": 250,
                "dir": "in",
                "kind": "text",
                "payload": json.dumps(
                    {
                        "type": "partial_transcript",
                        "utterance_id": "u1",
                        "seq": 3,
                        "text": "partial three late",
                        "audio_ms": 300,
                    }
                ),
            },
        ]
        ws.load_jsonl("\n".join(json.dumps(m) for m in messages))

        events: list[SttEvent] = []
        async for evt in stream.events():
            events.append(evt)

        await stream.close()

        # Only partial two and final one should be delivered
        assert len(events) == 2
        assert events[0].kind == "partial"
        assert events[0].text == "partial two"
        assert events[1].kind == "final"
        assert events[1].text == "final one"

    run_virtual(_run())


def test_malformed_and_unknown_frames_ignored_and_errors_surfaced() -> None:
    """Verify unknown frame types are ignored and error frames surface error events."""

    async def _run() -> None:
        telemetry = RecordingSink()
        ws = FakeWebSocket()
        port = PrismaSttPort(
            ws_factory=lambda *a, **kw: ws,
            api_key="test-key",
            telemetry=telemetry,
        )
        token = CancelToken()
        stream = await port.open(cancel=token)

        messages = [
            # 1. Malformed JSON
            {"t_ms": 10, "dir": "in", "kind": "text", "payload": "{malformed json"},
            # 2. Unknown frame type
            {
                "t_ms": 20,
                "dir": "in",
                "kind": "text",
                "payload": json.dumps(
                    {
                        "type": "unknown_future_message",
                        "foo": "bar",
                    }
                ),
            },
            # 3. Server error frame
            {
                "t_ms": 30,
                "dir": "in",
                "kind": "text",
                "payload": json.dumps(
                    {
                        "type": "error",
                        "message": "Rate limit exceeded",
                        "code": 429,
                    }
                ),
            },
            # 4. Valid final transcript
            {
                "t_ms": 40,
                "dir": "in",
                "kind": "text",
                "payload": json.dumps(
                    {
                        "type": "final_transcript",
                        "utterance_id": "u2",
                        "text": "recovered",
                        "audio_duration_ms": 100,
                    }
                ),
            },
        ]
        ws.load_jsonl("\n".join(json.dumps(m) for m in messages))

        events: list[SttEvent] = []
        async for evt in stream.events():
            events.append(evt)

        await stream.close()

        # Stream didn't crash and yielded the valid final transcript
        assert len(events) == 1
        assert events[0].text == "recovered"

        # Error frame was surfaced
        assert stream.last_error is not None
        assert stream.last_error["message"] == "Rate limit exceeded"
        err_metrics = telemetry.events_named("stt.error")
        assert len(err_metrics) == 1

    run_virtual(_run())


def test_reconnect_replays_audio_buffer_with_backoff() -> None:
    """Verify reconnect uses backoff and replays <= 2s audio buffer."""

    async def _run() -> None:
        telemetry = RecordingSink()
        ws1 = FakeWebSocket()
        ws2 = FakeWebSocket()

        # Preload ws1 with a partial
        ws1_msgs = [
            {
                "t_ms": 10,
                "dir": "in",
                "kind": "text",
                "payload": json.dumps(
                    {
                        "type": "partial_transcript",
                        "utterance_id": "u_rec",
                        "seq": 1,
                        "text": "first part",
                        "audio_ms": 100,
                    }
                ),
            },
        ]
        ws1.load_jsonl("\n".join(json.dumps(m) for m in ws1_msgs))

        # ws2 will receive the replayed frames and yield final
        ws2_msgs = [
            {
                "t_ms": 50,
                "dir": "in",
                "kind": "text",
                "payload": json.dumps(
                    {
                        "type": "final_transcript",
                        "utterance_id": "u_rec",
                        "text": "full text",
                        "audio_duration_ms": 500,
                    }
                ),
            },
        ]
        ws2.load_jsonl("\n".join(json.dumps(m) for m in ws2_msgs))

        sockets = [ws1, ws2]

        def ws_factory(*a: Any, **kw: Any) -> FakeWebSocket:
            if sockets:
                return sockets.pop(0)
            return FakeWebSocket()

        port = PrismaSttPort(
            ws_factory=ws_factory,
            api_key="test-key",
            telemetry=telemetry,
            max_reconnect_attempts=2,
            reconnect_backoff_initial_ms=10,
        )
        token = CancelToken()
        stream = await port.open(cancel=token)

        # Send 3 audio frames (recorded in replay buffer)
        for i in range(3):
            frame = AudioFrame(pcm=b"\x01\x02" * 320, sample_rate=16000, ts_ns=i * 20_000_000)
            await stream.send_audio(frame)

        # Allow send loop to send to ws1
        await asyncio.sleep(0.01)

        events: list[SttEvent] = []
        async for evt in stream.events():
            events.append(evt)
            if evt.kind == "partial":
                # Drop socket after receiving partial
                await ws1.close()

        await stream.close()

        # Verify ws2 received the 3 replayed frames
        assert len(ws2.sent_messages) == 3

        # Verify reconnect telemetry was emitted
        reconnect_events = telemetry.events_named("stt.reconnected")
        assert len(reconnect_events) == 1
        assert reconnect_events[0].fields["replayed_frames"] == 3

        # Verify both partial and final were yielded across the reconnect
        assert len(events) == 2
        assert events[0].text == "first part"
        assert events[1].text == "full text"

    run_virtual(_run())


def test_reconnect_max_attempts_cap_prevents_storm() -> None:
    """Verify stream halts after reaching max reconnect attempts without hanging."""

    async def _run() -> None:
        telemetry = RecordingSink()
        # Factory returns sockets that are closed
        attempts_seen = 0

        def failing_factory(*a: Any, **kw: Any) -> FakeWebSocket:
            nonlocal attempts_seen
            attempts_seen += 1
            ws = FakeWebSocket()
            ws._closed = True
            return ws

        port = PrismaSttPort(
            ws_factory=failing_factory,
            api_key="test-key",
            telemetry=telemetry,
            max_reconnect_attempts=3,
            reconnect_backoff_initial_ms=5,
        )
        token = CancelToken()
        stream = await port.open(cancel=token)

        events: list[SttEvent] = []
        async for evt in stream.events():
            events.append(evt)

        await stream.close()

        # Reconnect cap reached (1 initial + 3 retries = 4 socket creations)
        assert attempts_seen == 4
        assert stream.has_error is True
        exhausted_events = telemetry.events_named("stt.reconnect_exhausted")
        assert len(exhausted_events) == 1

    run_virtual(_run())
