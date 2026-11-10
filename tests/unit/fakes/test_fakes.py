# ruff: noqa: E402
"""Unit and contract tests for all port fakes."""

from __future__ import annotations

import sys
from pathlib import Path

_tests_dir = Path(__file__).resolve().parents[2]
_fakes_dir = str(_tests_dir / "fakes")
_contracts_dir = str(_tests_dir / "contracts")
_helpers_dir = str(_tests_dir / "helpers")

for _p in (_fakes_dir, _contracts_dir, _helpers_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
from llm import FakeLlm
from sink import RecordingSink
from stt import FakeStt
from suites import (
    LlmContract,
    SttContract,
    TelemetryContract,
    TransportContract,
    TtsContract,
    VadContract,
)
from transport import FakeTransport
from tts import FakeTts
from vad import FakeVad
from virtual import run_virtual
from ws import AssumedFixtureWarning, FakeWebSocket

from overlap.core.clock import CancelToken, VirtualClock
from overlap.core.types import AudioFrame, SttEvent, Tag
from overlap.ports.llm import Message
from overlap.ports.tts import AudioChunk

# --- Contract Suite Verification ---


def test_fake_stt_passes_contract() -> None:
    SttContract.verify_sync(FakeStt())


def test_fake_tts_passes_contract() -> None:
    TtsContract.verify_sync(FakeTts())


def test_fake_llm_passes_contract() -> None:
    LlmContract.verify_sync(FakeLlm())


def test_fake_transport_passes_contract() -> None:
    TransportContract.verify(FakeTransport())


def test_fake_vad_passes_contract() -> None:
    VadContract.verify(FakeVad())


def test_recording_sink_passes_contract() -> None:
    TelemetryContract.verify(RecordingSink())


# --- Specific Fake Behaviors & Failure Modes ---


def test_fake_stt_timeline_and_cancellation() -> None:
    async def _test() -> None:
        clock = VirtualClock()
        evt1 = SttEvent(kind="partial", text="namaste", conf=0.9, ts_ns=1000)
        evt2 = SttEvent(kind="final", text="namaste ji", conf=0.95, ts_ns=2000)
        fake_stt = FakeStt(clock=clock)
        fake_stt.add_event(10, evt1)
        fake_stt.add_event(20, evt2)

        token = CancelToken()
        stream = await fake_stt.open(cancel=token)

        # Enqueue audio frame - must not block
        frame = AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=0)
        await stream.send_audio(frame)

        # Advance clock to 10 ms: evt1 should arrive
        await clock.advance(10)
        stream_iter = stream.events()
        first_evt = await anext(stream_iter)
        assert first_evt.text == "namaste"

        # Advance to 20 ms: evt2 should arrive
        await clock.advance(10)
        second_evt = await anext(stream_iter)
        assert second_evt.text == "namaste ji"

        # Test cancellation stops further delivery
        token.cancel()
        delivered: list[SttEvent] = []
        async for evt in stream.events():
            delivered.append(evt)
        assert len(delivered) == 0

    run_virtual(_test())


def test_fake_tts_honors_cancel_and_stops_emitting() -> None:
    """Failure mode 2: FakeTts ignores cancel() and keeps emitting (hides leak bugs)."""

    async def _test() -> None:
        clock = VirtualClock()
        tts = FakeTts(clock=clock, cadence_ms=10.0)
        token = CancelToken()
        stream = await tts.open_stream(voice_id="v1", lang="hi", epoch=1, cancel=token)

        await stream.send_text("Yeh ek lamba sentence hai jisme kaafi saare words hain.")
        await stream.finish()

        # Advance virtual clock to receive first chunk
        await clock.advance(10)
        chunks_iter = stream.chunks()
        first_chunk = await anext(chunks_iter)
        assert isinstance(first_chunk, AudioChunk)
        assert first_chunk.epoch == 1

        # Cancel while more chunks are scheduled
        await stream.cancel()

        # No more chunks may be delivered after cancellation
        delivered: list[AudioChunk] = []
        async for chunk in stream.chunks():
            delivered.append(chunk)
        assert len(delivered) == 0

    run_virtual(_test())


def test_fake_llm_streaming_and_abort() -> None:
    async def _test() -> None:
        clock = VirtualClock()
        llm = FakeLlm(deltas=["Haan ", "main ", "sun ", "raha ", "hoon."], clock=clock)
        token = CancelToken()
        stream = await llm.stream([Message(role="user", content="kya?")], epoch=3, cancel=token)

        deltas_iter = stream.deltas()
        d1 = await anext(deltas_iter)
        assert d1 == "Haan "

        # Abort stream
        await stream.abort()

        delivered: list[str] = []
        async for d in stream.deltas():
            delivered.append(d)
        assert len(delivered) == 0

    run_virtual(_test())


def test_fake_transport_records_frames_rms_and_flush() -> None:
    """FakeTransport records every sent frame (ts, tags, gain-applied rms)
    and models configurable downstream latency + flush.
    """
    clock = VirtualClock()
    transport = FakeTransport(downstream_latency_ms=40, clock=clock)

    # Frame 1: silent
    f1 = AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=100_000, tags=Tag.NONE)
    transport.send_frame(f1)

    # Frame 2: tone
    import math

    tone_samples = bytearray()
    for i in range(320):
        val = int(16384 * math.sin(2 * math.pi * 440 * i / 16000))
        tone_samples.extend(val.to_bytes(2, byteorder="little", signed=True))
    f2 = AudioFrame(pcm=bytes(tone_samples), sample_rate=16000, ts_ns=120_000, tags=Tag.SELF_TTS)
    transport.send_frame(f2)

    assert len(transport.sent_frames) == 2
    rec1 = transport.sent_frames[0]
    rec2 = transport.sent_frames[1]

    assert rec1.ts == 100_000
    assert rec1.tags == Tag.NONE
    assert rec1.rms == 0.0

    assert rec2.ts == 120_000
    assert rec2.tags == Tag.SELF_TTS
    assert rec2.rms > 10000.0  # RMS around 11585

    # output_queue_ms models duration of frames + downstream latency
    # f1 = 20ms, f2 = 20ms, downstream = 40ms => 80ms
    assert transport.output_queue_ms() == 80

    # Flush clears output queue
    transport.flush_output()
    assert transport.output_queue_ms() == 0
    assert transport.flushes == 1


def test_recording_sink_records_events() -> None:
    sink = RecordingSink()
    sink.emit("vad.onset", ts_ns=1000, energy=0.8, dur_ms=120)
    sink.emit("tts.cancel", ts_ns=2000, epoch=4)

    assert sink.has_event("vad.onset")
    assert sink.has_event("tts.cancel")
    assert not sink.has_event("llm.start")

    events = sink.events_named("vad.onset")
    assert len(events) == 1
    assert events[0].ts_ns == 1000
    assert events[0].fields["energy"] == 0.8


def test_fake_ws_replay_and_assumed_warning(tmp_path: Path) -> None:
    """FakeWebSocket replays JSONL transcripts honoring t_ms in virtual time,
    and flags assumed fixtures.
    """
    import base64

    b64_pcm = base64.b64encode(b"\x01\x02\x03\x04").decode("ascii")

    # JSONL with assumed: true
    fixture_content = (
        '{"t_ms": 10, "dir": "in", "kind": "text", "payload": "hello", "assumed": true}\n'
        f'{{"t_ms": 30, "dir": "in", "kind": "binary", "b64": "{b64_pcm}"}}\n'
        '{"t_ms": 50, "dir": "out", "kind": "text", "payload": "ack"}\n'
    )
    fixture_path = tmp_path / "test_wire.jsonl"
    fixture_path.write_text(fixture_content, encoding="utf-8")

    async def _test() -> None:
        clock = VirtualClock()
        ws = FakeWebSocket(clock=clock)

        with pytest.warns(AssumedFixtureWarning, match="assumed"):
            ws.load_jsonl(fixture_path)

        assert ws.is_assumed is True

        # Send outbound frame
        await ws.send_text("client_init")
        assert len(ws.sent_messages) == 1
        assert ws.sent_messages[0].payload == "client_init"

        # Advance to 10 ms -> receive text "hello"
        await clock.advance(10)
        msg1 = await ws.recv()
        assert msg1 == "hello"

        # Advance to 30 ms -> receive binary b64
        await clock.advance(20)
        msg2 = await ws.recv()
        assert msg2 == b"\x01\x02\x03\x04"

    run_virtual(_test())
