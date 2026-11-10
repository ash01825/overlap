# ruff: noqa: E402
"""Unit and golden scenario tests for TelemetryCollector, JsonlWriter, and report CLI (T4.03)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_fakes_dir = str(Path(__file__).resolve().parents[2] / "fakes")
if _fakes_dir not in sys.path:
    sys.path.insert(0, _fakes_dir)

_contracts_dir = str(Path(__file__).resolve().parents[2] / "contracts")
if _contracts_dir not in sys.path:
    sys.path.insert(0, _contracts_dir)

from llm import FakeLlm
from stt import FakeStt
from suites import TelemetryContract
from transport import FakeTransport
from tts import FakeTts
from vad import FakeVad

from overlap.core.audio import EchoGate, LoudnessTracker
from overlap.core.clock import VirtualClock
from overlap.core.fsm import FSM
from overlap.core.predictor import BackchannelPredictor
from overlap.core.types import (
    AudioFrame,
    EndOfTurn,
    FsmState,
    Label,
    SttEvent,
    Tuning,
    WordTiming,
    YieldCompleted,
)
from overlap.ports.tts import AudioChunk, AudioClip
from overlap.runtime.clip_cache import ClipCache
from overlap.runtime.engine import OverlapEngine
from overlap.runtime.mixer import PlayoutMixer
from overlap.telemetry.collector import TelemetryCollector
from overlap.telemetry.jsonl import JsonlWriter, read_jsonl
from overlap.telemetry.report import format_report_table
from overlap.telemetry.report import main as report_main


def test_collector_passes_telemetry_contract(tmp_path: Path) -> None:
    """Verify TelemetryCollector passes the T0.04 TelemetryContract suite."""
    jsonl_file = tmp_path / "contract.jsonl"
    collector = TelemetryCollector(jsonl_path=jsonl_file, max_queue_size=50)
    try:
        TelemetryContract.verify(collector)
    finally:
        collector.close()


def test_jsonl_writer_non_blocking_drop_oldest_and_never_raises(tmp_path: Path) -> None:
    """Failure mode 1: sink blocks or raises on the fast path (disk I/O inline)."""
    jsonl_file = tmp_path / "bounded.jsonl"
    writer = JsonlWriter(jsonl_file, max_queue_size=5)

    # Pause worker by holding condition lock while we flood the bounded queue
    with writer._cond:
        for i in range(12):
            writer._queue.append({"name": "ev", "ts_ns": i, "seq": i})
            if len(writer._queue) > writer.max_queue_size:
                writer._queue.popleft()
                writer._dropped_count += 1
        writer._cond.notify_all()

    writer.flush()
    writer.close()

    assert writer.dropped_count == 7
    records = read_jsonl(jsonl_file)
    assert len(records) == 5
    # Oldest 0..6 were dropped; 7..11 preserved in order
    assert [r["seq"] for r in records] == [7, 8, 9, 10, 11]

    # Also test enqueue overflow directly via public API
    jsonl_file2 = tmp_path / "overflow2.jsonl"
    writer2 = JsonlWriter(jsonl_file2, max_queue_size=3)
    with writer2._cond:
        # Fill queue while worker waits on lock
        pass
    for i in range(20):
        writer2.enqueue({"name": "fast", "ts_ns": i, "seq": i})
    writer2.flush()
    writer2.close()
    assert writer2.dropped_count + writer2.written_count == 20

    # Test that even an unwritable path (directory instead of file) never raises on emit()
    bad_collector = TelemetryCollector(jsonl_path=tmp_path, max_queue_size=10)
    bad_collector.emit("vad.onset", 1000, detail="must_not_raise")
    bad_collector.flush(timeout_s=0.2)
    bad_collector.close(timeout_s=0.2)


def test_redaction_replaces_transcript_pii_with_hash_and_length(tmp_path: Path) -> None:
    """Failure mode 4: transcript text/PII written to disk when redaction is on."""
    jsonl_file = tmp_path / "redacted.jsonl"
    secret_pii = "Mera phone number 9876543210 hai aur naam Rahul Sharma"

    collector = TelemetryCollector(jsonl_path=jsonl_file, redact=True)
    collector.emit(
        "verdict",
        100_000_000,
        label="INTERRUPT",
        partial_text=secret_pii,
        text=secret_pii,
        nested={"transcript": secret_pii, "safe_num": 42},
    )
    collector.emit(
        "yield.truncate",
        120_000_000,
        heard_words="Namaste main",
        truth_words="Namaste main aapki",
        text=secret_pii,
    )
    collector.flush()
    collector.close()

    raw_disk = jsonl_file.read_text(encoding="utf-8")
    assert "9876543210" not in raw_disk
    assert "Rahul Sharma" not in raw_disk
    assert "Namaste" not in raw_disk
    assert "<redacted sha256=" in raw_disk
    assert f"len={len(secret_pii)}" in raw_disk
    assert "words=9" in raw_disk

    # Verify truncation word error can still be derived from redacted record
    records = read_jsonl(jsonl_file)
    assert len(records) == 2
    m = collector.derive_metrics()
    assert m.truncation_error_words == (1,)


def test_golden_t204_scenario_event_log_produces_expected_metrics(tmp_path: Path) -> None:
    """Golden test: event log from a T2.04 scenario produces expected metric values."""
    jsonl_file = tmp_path / "golden_scenario.jsonl"
    collector = TelemetryCollector(jsonl_path=jsonl_file, redact=False)

    clock = VirtualClock(start_ns=0)
    tuning = Tuning()
    transport = FakeTransport(clock=clock)
    vad = FakeVad()
    stt = FakeStt(clock=clock)
    tts = FakeTts(clock=clock)
    llm = FakeLlm(clock=clock)
    echo_gate = EchoGate()
    loudness_tracker = LoudnessTracker()
    fsm = FSM(tuning=tuning, initial_epoch=1)
    predictor = BackchannelPredictor(tuning=tuning, default_lang="hi")
    clip_cache = ClipCache(tts=tts, voice_id="v1", tuning=tuning)
    mixer = PlayoutMixer(
        transport=transport,
        clock=clock,
        tuning=tuning,
        echo_gate=echo_gate,
        telemetry=collector,
    )

    engine = OverlapEngine(
        transport=transport,
        vad=vad,
        stt=stt,
        tts=tts,
        llm=llm,
        clock=clock,
        telemetry=collector,
        tuning=tuning,
        voice_id="v1",
        lang="hi",
        clip_cache=clip_cache,
        echo_gate=echo_gate,
        loudness_tracker=loudness_tracker,
        fsm=fsm,
        predictor=predictor,
        mixer=mixer,
    )

    # Pre-populate clip cache for agent backchannel
    clip_pcm = b"\x10\x27" * (320 * 10)
    clip_cache._memory_cache[("v1", "hi", "haan", False)] = AudioClip(
        pcm=clip_pcm, sample_rate=16000
    )
    clip_cache._by_lang["hi"] = [AudioClip(pcm=clip_pcm, sample_rate=16000)]

    # Part A: T2.04 Scenario 1 ('haan haan' user backchannel during agent speech)
    turn_id = "turn_1"
    engine.supervisor.start_turn(turn_id, 1)
    mixer.start_turn(turn_id, 1)
    engine._apply_event_fast_path(
        EndOfTurn(final=SttEvent(kind="final", text="hello")), clock.now_ns()
    )
    chunk1 = AudioChunk(
        pcm=b"\x08\x10" * (320 * 25),
        sample_rate=16000,
        epoch=1,
        words=(
            WordTiming(word="Namaste", start_ms=0, end_ms=200),
            WordTiming(word="dosto", start_ms=200, end_ms=500),
        ),
        is_last=True,
    )
    engine.turn_pipeline._handle_audio_chunk(chunk1)
    engine._handle_tts_audio_chunk(chunk1)

    clock.advance(20)
    clock.advance(20)  # t = 40 ms

    # User onset at t = 40 ms
    vad.set_speech(True, confidence=0.95)
    engine.handle_frame(AudioFrame(pcm=b"\x20\x30" * 320, sample_rate=16000, ts_ns=clock.now_ns()))
    for _ in range(9):
        clock.advance(20)
        engine.handle_frame(
            AudioFrame(pcm=b"\x20\x30" * 320, sample_rate=16000, ts_ns=clock.now_ns())
        )

    # User offset at t = 220 ms + STT partial "haan haan"
    vad.set_speech(False)
    engine.handle_frame(AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=clock.now_ns()))
    engine.handle_stt_event(
        SttEvent(kind="partial", text="haan haan", lang="hi", conf=0.95, ts_ns=clock.now_ns())
    )
    while mixer.is_playing:
        clock.advance(20)
    engine._apply_event_fast_path(
        EndOfTurn(final=SttEvent(kind="final", text="next")), clock.now_ns()
    )

    # Part B: T2.04 Scenario 2 ('ruko ek minute' user interruption)
    engine.supervisor.start_turn("turn_2", 1)
    mixer.start_turn("turn_2", 1)
    chunk2 = AudioChunk(
        pcm=b"\x08\x10" * (320 * 40),
        sample_rate=16000,
        epoch=1,
        words=(
            WordTiming(word="Namaste", start_ms=0, end_ms=100),
            WordTiming(word="main", start_ms=100, end_ms=200),
            WordTiming(word="aapki", start_ms=200, end_ms=300),
            WordTiming(word="madad", start_ms=300, end_ms=500),
        ),
        is_last=True,
    )
    engine.turn_pipeline._handle_audio_chunk(chunk2)
    engine._handle_tts_audio_chunk(chunk2)

    for _ in range(8):
        clock.advance(20)

    vad.set_speech(True)
    engine.handle_frame(AudioFrame(pcm=b"\x20\x30" * 320, sample_rate=16000, ts_ns=clock.now_ns()))
    clock.advance(20)
    engine.handle_stt_event(
        SttEvent(kind="partial", text="ruko ek minute", lang="hi", ts_ns=clock.now_ns())
    )
    vad.set_speech(False)
    engine.handle_frame(AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=clock.now_ns()))
    if engine.state == FsmState.YIELDED:
        engine._handle_supervisor_event(YieldCompleted())

    # Part C: T2.04 Scenario 7 (long user turn => agent backchannel)
    vad.set_speech(True)
    for _ in range(160):  # 3200 ms user speech
        engine.handle_frame(
            AudioFrame(pcm=b"\x20\x30" * 320, sample_rate=16000, ts_ns=clock.now_ns())
        )
        clock.advance(20)

    engine.handle_stt_event(
        SttEvent(
            kind="partial",
            text="Main kal subah aapse milna chahta hoon.",
            lang="hi",
            ts_ns=clock.now_ns(),
        )
    )
    vad.set_speech(False)
    for _ in range(12):  # 240 ms pause
        engine.handle_frame(AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=clock.now_ns()))
        if engine.state == FsmState.BACKCHANNELING:
            break
        clock.advance(20)

    clock.advance(1000)
    engine.supervisor.assert_quiescent()

    collector.flush()
    collector.close()

    # Verify derived metrics from the golden scenario event log
    records = read_jsonl(jsonl_file)
    assert len(records) >= 8

    # Tag ground truth on verdict records for FIR/missed rate check
    for r in records:
        if r.get("name") == "verdict":
            if r.get("label") == Label.BACKCHANNEL.value:
                r["ground_truth"] = "BACKCHANNEL"
            elif r.get("label") == Label.INTERRUPT.value:
                r["ground_truth"] = "INTERRUPT"

    jsonl_file.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )

    m = collector.derive_metrics()
    assert len(m.t_duck_ms) == 2
    assert m.t_duck_p95_ms is not None and m.t_duck_p95_ms <= tuning.duck_latency_cap_ms
    assert len(m.t_verdict_ms) == 2
    assert m.t_verdict_p95_ms is not None and m.t_verdict_p95_ms <= tuning.classify_cap_ms
    assert len(m.t_stop_ms) == 1
    assert m.t_stop_p50_ms == 20.0
    assert len(m.bc_timing_ms) == 1
    assert (
        tuning.bc_pause_min_ms
        <= m.bc_timing_ms[0]
        <= (tuning.bc_pause_max_ms + tuning.bc_clip_start_ms)
    )
    assert m.echo_false_duck == 0


def test_report_cli_prints_table_and_pass_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test `python -m overlap.telemetry.report <jsonl>` CLI output and exit codes."""
    jsonl_file = tmp_path / "cli_pass.jsonl"
    events: list[dict[str, Any]] = [
        {"name": "vad.onset", "ts_ns": 100_000_000, "epoch": 1},
        {"name": "duck.applied", "ts_ns": 120_000_000, "epoch": 1},
        {"name": "vad.offset", "ts_ns": 300_000_000, "epoch": 1},
        {
            "name": "verdict",
            "ts_ns": 300_000_000,
            "label": "BACKCHANNEL",
            "ground_truth": "BACKCHANNEL",
            "epoch": 1,
        },
        {"name": "vad.onset", "ts_ns": 1_000_000_000, "epoch": 1},
        {"name": "duck.applied", "ts_ns": 1_020_000_000, "epoch": 1},
        {
            "name": "verdict",
            "ts_ns": 1_100_000_000,
            "label": "INTERRUPT",
            "ground_truth": "INTERRUPT",
            "epoch": 1,
        },
        {"name": "yield.flush", "ts_ns": 1_120_000_000, "epoch": 1},
        {"name": "yield.truncate", "ts_ns": 1_120_000_000, "error_words": 0, "epoch": 1},
        {"name": "vad.offset", "ts_ns": 1_400_000_000, "epoch": 1},
        {"name": "vad.onset", "ts_ns": 2_000_000_000, "epoch": 2},
        {"name": "vad.offset", "ts_ns": 12_000_000_000, "epoch": 2},
        {
            "name": "bc.agent.fire",
            "ts_ns": 12_240_000_000,
            "pause_start_ns": 12_000_000_000,
            "epoch": 2,
        },
        {"name": "bc.agent.clip_start", "ts_ns": 12_260_000_000, "epoch": 2},
    ]
    jsonl_file.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    exit_code = report_main([str(jsonl_file)])
    captured = capsys.readouterr().out
    assert exit_code == 0
    for metric_name in (
        "t_duck",
        "t_verdict",
        "t_stop",
        "false_interruption_rate",
        "missed_interruption_rate",
        "bc_timing",
        "bc_rate",
        "truncation_error_words",
        "echo_false_duck",
    ):
        assert metric_name in captured
    assert "OVERALL: PASS" in captured

    # Failing log (echo_false_duck > 0) returns exit code 1 and prints FAIL
    fail_file = tmp_path / "cli_fail.jsonl"
    fail_events = [*events, {"name": "duck.applied", "ts_ns": 20_000_000_000, "is_echo": True}]
    fail_file.write_text("\n".join(json.dumps(e) for e in fail_events) + "\n", encoding="utf-8")

    fail_code = report_main([str(fail_file)])
    fail_out = capsys.readouterr().out
    assert fail_code == 1
    assert "OVERALL: FAIL" in fail_out

    # Format table helper works directly on DerivedMetrics too
    table_str, ok = format_report_table(events)
    assert ok is True
    assert "OVERALL: PASS" in table_str
