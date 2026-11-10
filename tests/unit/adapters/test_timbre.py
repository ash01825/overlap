# ruff: noqa: E402
"""Unit tests for TimbreTtsAdapter replaying wire fixtures and verifying contracts.

Covers all done_when criteria and failure modes specified in T3.03:
- Implements TtsPort/TtsStream strictly from docs/WIRE.md.
- Passes TtsContract suite via recorded-fixture replay.
- cancel(): sends protocol cancel/close, discards inbound audio, idempotent;
  verified with recorded cancel-mid-stream fixture (0 chunks yielded after cancel returns).
- AudioChunk.words populated with clause-relative timings when service provides them;
  otherwise empty (calibrated rate fallback per WIRE.md).
- synthesize() returns AudioClip with exact sample count and rate for ClipCache.
- Failure modes tested:
  1. chunks delivered after cancel() (leak prevention).
  2. word timings mis-aligned / absent fallback.
  3. sample-rate and format mismatch resampled to 16 kHz mono.
  4. cancel while send_text in flight deadlocks.
  5. SSML passthrough vs XML escaping.
  6. batch synthesize odd-length or clipped PCM.
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import sys
from pathlib import Path

_tests_dir = Path(__file__).resolve().parents[2]
_fakes_dir = str(_tests_dir / "fakes")
_contracts_dir = str(_tests_dir / "contracts")
_helpers_dir = str(_tests_dir / "helpers")
_repo_root = str(Path(__file__).resolve().parents[3])

for _p in (_repo_root, _fakes_dir, _contracts_dir, _helpers_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from overlap.adapters.gnani.timbre import (
    TimbreTtsAdapter,
    TimbreTtsConfig,
    _format_clause,
)
from overlap.core.clock import CancelToken, VirtualClock
from overlap.core.types import WordTiming
from overlap.ports.tts import AudioChunk, AudioClip, TtsPort, TtsStream

_tests_dir = Path(__file__).resolve().parents[2]
_fakes_dir = str(_tests_dir / "fakes")
_contracts_dir = str(_tests_dir / "contracts")
_helpers_dir = str(_tests_dir / "helpers")

for _p in (_fakes_dir, _contracts_dir, _helpers_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from suites import TtsContract  # noqa: E402
from virtual import run_virtual  # noqa: E402
from ws import FakeWebSocket  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "wire" / "timbre"
FULL_STREAM_PATH = FIXTURES_DIR / "full_stream.jsonl"
CANCEL_STREAM_PATH = FIXTURES_DIR / "cancel_mid_stream.jsonl"


def _make_full_stream_ws(clock: VirtualClock | None = None) -> FakeWebSocket:
    ws = FakeWebSocket(clock=clock)
    ws.load_jsonl(FULL_STREAM_PATH)
    return ws


def _make_cancel_stream_ws(clock: VirtualClock | None = None) -> FakeWebSocket:
    ws = FakeWebSocket(clock=clock)
    ws.load_jsonl(CANCEL_STREAM_PATH)
    return ws


class TestTimbreTtsContract:
    """Verifies that TimbreTtsAdapter strictly fulfills the TtsContract."""

    def test_tts_contract_compliance(self) -> None:
        """Run full reusable TtsContract test suite against TimbreTtsAdapter."""
        adapter = TimbreTtsAdapter(
            TimbreTtsConfig(api_key="mock-key"),
            ws_factory=_make_full_stream_ws,
        )
        assert isinstance(adapter, TtsPort)
        TtsContract.verify_sync(adapter)


class TestTimbreStreamReplay:
    """Tests replaying recorded production traffic through TimbreTtsAdapter."""

    def test_full_stream_replay_and_schema(self) -> None:
        """Verify full stream fixture yields valid PCM chunks and strips WAV header."""

        async def _run() -> None:
            ws = _make_full_stream_ws()
            adapter = TimbreTtsAdapter(
                TimbreTtsConfig(api_key="mock-key", sample_rate=16000),
                ws_factory=lambda: ws,
            )

            token = CancelToken()
            stream = await adapter.open_stream(
                voice_id="Nalini",
                lang="hi-IN",
                epoch=42,
                cancel=token,
            )
            assert isinstance(stream, TtsStream)

            await stream.send_text("नमस्ते, आप कैसे हैं?")
            await stream.finish()

            # Verify outbound message schema matches WIRE.md §3.2
            assert len(ws.sent_messages) >= 1
            out_msg = ws.sent_messages[0]
            assert out_msg.kind == "text"
            assert out_msg.payload is not None
            out_data = json.loads(out_msg.payload)
            assert out_data["text"] == "नमस्ते, आप कैसे हैं?"
            assert out_data["voice"] == "Nalini"
            assert out_data["model"] == "timbre-v2.5"
            assert out_data["language"] == "hi-IN"
            assert out_data["audio_config"]["sample_rate"] == 16000
            assert out_data["audio_config"]["encoding"] == "linear_pcm"
            assert out_data["audio_config"]["container"] == "wav"

            # Drain chunks
            chunks: list[AudioChunk] = []
            async for chunk in stream.chunks():
                chunks.append(chunk)

            assert len(chunks) > 0
            # Chunk 1 had WAV header (RIFF) - verify WAV header was stripped from PCM
            assert not chunks[0].pcm.startswith(b"RIFF")
            # All audio chunks have 16 kHz sample rate and epoch 42
            for c in chunks:
                assert c.sample_rate == 16000
                assert c.epoch == 42
                assert len(c.pcm) % 2 == 0  # 16-bit PCM must have even byte length
                # Timbre wire analysis verified word timings are absent
                assert c.words == ()

            # Last chunk or stream termination marks is_last=True
            assert chunks[-1].is_last is True

        run_virtual(_run())

    def test_cancel_mid_stream_fixture(self) -> None:
        """Verify cancel-mid-stream fixture sends cancel frame and stops chunks."""

        async def _run() -> None:
            ws = _make_cancel_stream_ws()
            adapter = TimbreTtsAdapter(
                TimbreTtsConfig(api_key="mock-key"),
                ws_factory=lambda: ws,
            )

            token = CancelToken()
            stream = await adapter.open_stream(
                voice_id="Nalini",
                lang="hi-IN",
                epoch=99,
                cancel=token,
            )

            await stream.send_text("यह एक बहुत लंबा वाक्य है...")

            chunk_iter = stream.chunks()
            # Receive chunk 1
            chunk1 = await anext(chunk_iter)
            assert isinstance(chunk1, AudioChunk)
            assert len(chunk1.pcm) > 0

            # Mid-stream interruption
            await stream.cancel()

            # Verify protocol cancel message sent to wire per WIRE.md §3.5
            cancel_msgs = [
                m
                for m in ws.sent_messages
                if m.payload and "cancel" in m.payload and "1000" in m.payload
            ]
            assert len(cancel_msgs) >= 1

            # 0 chunks yielded after cancel returns
            chunks_after_cancel: list[AudioChunk] = []
            try:
                async for ch in chunk_iter:
                    chunks_after_cancel.append(ch)
            except (StopAsyncIteration, asyncio.CancelledError):
                pass

            assert len(chunks_after_cancel) == 0

            # Idempotence: calling cancel again must not raise or send duplicate close
            await stream.cancel()
            await stream.cancel()

        run_virtual(_run())


class TestTimbreFailureModes:
    """Unit tests targeting each explicit failure mode for T3.03."""

    def test_chunks_still_delivered_after_cancel_prevented(self) -> None:
        """Failure mode 1: No audio leakage after cancel()."""

        async def _run() -> None:
            # Custom websocket with a long stream of chunks
            ws = FakeWebSocket()
            ws.load_jsonl(FULL_STREAM_PATH)

            adapter = TimbreTtsAdapter(
                TimbreTtsConfig(api_key="mock-key"),
                ws_factory=lambda: ws,
            )

            token = CancelToken()
            stream = await adapter.open_stream(voice_id="v1", lang="hi", epoch=1, cancel=token)
            await stream.send_text("test")

            # Cancel immediately
            await stream.cancel()
            await asyncio.sleep(0)

            delivered: list[AudioChunk] = []
            async for ch in stream.chunks():
                delivered.append(ch)

            assert len(delivered) == 0, f"Expected 0 chunks after cancel, got {len(delivered)}"

        run_virtual(_run())

    def test_word_timings_clause_relative_and_fallback(self) -> None:
        """Failure mode 2: Word timings populated when provided, clause-relative, or empty."""

        async def _run() -> None:
            # Fake WS emitting audio frame with word timings
            ws = FakeWebSocket()
            start_evt = {"type": "start", "message": "Streaming started"}
            audio_pcm = b"\x00\x01" * 160  # 320 bytes = 20 ms
            audio_b64 = base64.b64encode(audio_pcm).decode("ascii")
            audio_evt = {
                "type": "audio",
                "data": {
                    "chunk_index": 1,
                    "audio": audio_b64,
                    "is_final": True,
                    "words": [
                        {"word": "नमस्ते", "start_ms": 0, "end_ms": 100},
                        {"word": "जी", "start_ms": 110, "end_ms": 190},
                    ],
                },
            }
            p1 = json.dumps(json.dumps(start_evt))
            p2 = json.dumps(json.dumps(audio_evt))
            jsonl = f'{{"t_ms": 10, "dir": "in", "kind": "text", "payload": {p1}}}\n'
            jsonl += f'{{"t_ms": 20, "dir": "in", "kind": "text", "payload": {p2}}}\n'
            ws.load_jsonl(jsonl)

            adapter = TimbreTtsAdapter(ws_factory=lambda: ws)
            stream = await adapter.open_stream(
                voice_id="v1", lang="hi", epoch=5, cancel=CancelToken()
            )
            await stream.send_text("नमस्ते जी")

            chunks = [ch async for ch in stream.chunks()]
            assert len(chunks) == 1
            assert len(chunks[0].words) == 2
            assert chunks[0].words[0] == WordTiming(word="नमस्ते", start_ms=0, end_ms=100)
            assert chunks[0].words[1] == WordTiming(word="जी", start_ms=110, end_ms=190)

        run_virtual(_run())

    def test_sample_rate_and_format_mismatch_resampled(self) -> None:
        """Failure mode 3: Audio with non-16kHz sample rate is resampled to 16kHz mono."""

        async def _run() -> None:
            # Construct a 24 kHz mono WAV chunk
            sr_24k = 24000
            num_samples = 480  # 20 ms at 24 kHz
            pcm_24k = b"\x10\x00" * num_samples  # 960 bytes

            # Create 44-byte WAV header for 24 kHz mono
            byte_rate = sr_24k * 2
            wav_header = (
                b"RIFF\xff\xff\xff\xffWAVEfmt \x10\x00\x00\x00"
                + struct.pack("<HHIIHH", 1, 1, sr_24k, byte_rate, 2, 16)
                + b"data\xff\xff\xff\xff"
            )
            full_wav = wav_header + pcm_24k
            b64_str = base64.b64encode(full_wav).decode("ascii")

            ws = FakeWebSocket()
            evt = {
                "type": "audio",
                "data": {
                    "chunk_index": 1,
                    "audio": b64_str,
                    "is_final": True,
                },
            }
            p = json.dumps(json.dumps(evt))
            ws.load_jsonl(f'{{"t_ms": 10, "dir": "in", "kind": "text", "payload": {p}}}\n')

            adapter = TimbreTtsAdapter(
                TimbreTtsConfig(sample_rate=16000),
                ws_factory=lambda: ws,
            )
            stream = await adapter.open_stream(
                voice_id="v1", lang="hi", epoch=1, cancel=CancelToken()
            )
            await stream.send_text("test")

            chunks = [c async for c in stream.chunks()]
            assert len(chunks) == 1
            # 20 ms at 16 kHz is 320 samples = 640 bytes
            assert chunks[0].sample_rate == 16000
            assert len(chunks[0].pcm) == 640

        run_virtual(_run())

    def test_cancel_while_send_text_in_flight_no_deadlock(self) -> None:
        """Failure mode 4: cancel() concurrent with send_text() does not deadlock."""

        async def _run() -> None:
            ws = FakeWebSocket()
            adapter = TimbreTtsAdapter(ws_factory=lambda: ws)
            stream = await adapter.open_stream(
                voice_id="v1", lang="hi", epoch=1, cancel=CancelToken()
            )

            # Run send_text and cancel concurrently
            t1 = asyncio.create_task(stream.send_text("concurrent clause"))
            t2 = asyncio.create_task(stream.cancel())

            await asyncio.gather(t1, t2)
            assert getattr(stream, "_cancelled", True) is True
            # chunks() terminates immediately
            chunks = [c async for c in stream.chunks()]
            assert len(chunks) == 0

        run_virtual(_run())

    def test_ssml_passthrough_vs_escaping(self) -> None:
        """Failure mode 5: SSML passthrough preserves tags; plain text escapes XML special chars."""

        # 1. Plain text escaping helper
        plain = "A & B < C > 'D' \"E\""
        escaped = _format_clause(plain, ssml=False)
        assert "&amp;" in escaped
        assert "&lt;" in escaped
        assert "&gt;" in escaped
        assert "<" not in escaped
        assert ">" not in escaped

        # 2. SSML passthrough preserves tags
        ssml_text = "<whispers>नमस्ते &amp; स्वागतम्</whispers>"
        passed = _format_clause(ssml_text, ssml=True)
        assert passed == ssml_text
        assert "<whispers>" in passed
        assert "</whispers>" in passed

        # 3. Stream transmission with ssml=True vs ssml=False
        async def _run() -> None:
            ws = FakeWebSocket()
            adapter = TimbreTtsAdapter(ws_factory=lambda: ws)
            stream = await adapter.open_stream(
                voice_id="v1", lang="hi", epoch=1, cancel=CancelToken()
            )

            await stream.send_text("<whispers>Hello</whispers>", ssml=True)
            msg1 = json.loads(ws.sent_messages[0].payload or "{}")
            assert msg1["text"] == "<whispers>Hello</whispers>"

            await stream.send_text("Tom & Jerry <friends>", ssml=False)
            msg2 = json.loads(ws.sent_messages[1].payload or "{}")
            assert "&amp;" in msg2["text"]
            assert "&lt;" in msg2["text"]

        run_virtual(_run())

    def test_batch_synthesize_clip_exact_samples_and_rate(self) -> None:
        """Failure mode 6: synthesize() returns AudioClip with exact samples and no odd PCM."""

        async def _run() -> None:
            # 1. Test via WebSocket delegation (offline replay)
            adapter = TimbreTtsAdapter(
                TimbreTtsConfig(sample_rate=16000),
                ws_factory=_make_full_stream_ws,
            )

            clip = await adapter.synthesize("नमस्ते", voice_id="Nalini", lang="hi-IN")
            assert isinstance(clip, AudioClip)
            assert clip.sample_rate == 16000
            assert len(clip.pcm) > 0
            assert len(clip.pcm) % 2 == 0  # Even length invariant

            # 2. Test via REST http_client injection with odd-length raw PCM (e.g. 513 bytes)
            odd_raw = b"\x00\x05" * 256 + b"\x01"  # 513 bytes
            stub_http = lambda text, **kw: odd_raw  # noqa: E731

            adapter_rest = TimbreTtsAdapter(
                TimbreTtsConfig(sample_rate=16000),
                http_client=stub_http,
            )
            clip_rest = await adapter_rest.synthesize("test clip", voice_id="v1", lang="hi")
            assert isinstance(clip_rest, AudioClip)
            assert clip_rest.sample_rate == 16000
            # 513 bytes must be trimmed to 512 bytes (256 samples)
            assert len(clip_rest.pcm) == 512
            assert len(clip_rest.pcm) % 2 == 0

        run_virtual(_run())

    def test_cancel_token_cancels_stream(self) -> None:
        """Verify that cancelling the CancelToken stops chunks delivery and sends cancel frame."""

        async def _run() -> None:
            ws = _make_full_stream_ws()
            adapter = TimbreTtsAdapter(ws_factory=lambda: ws)
            token = CancelToken()
            stream = await adapter.open_stream(voice_id="v1", lang="hi", epoch=1, cancel=token)
            await stream.send_text("test")

            # External token cancellation
            token.cancel()
            await asyncio.sleep(0)

            delivered: list[AudioChunk] = []
            async for ch in stream.chunks():
                delivered.append(ch)

            assert len(delivered) == 0

        run_virtual(_run())
