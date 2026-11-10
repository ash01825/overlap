# ruff: noqa: E402
"""Unit tests for StreamSupervisor: cancel tokens, timers, effect executor, quiescence check.

Tests all done_when criteria and failure modes specified in task T2.02 and ARCHITECTURE §3, §6, §10:
- Effect executor maps each Effect to port calls with isolation and telemetry logging.
- Timers via Clock with ArmTimer/CancelTimer semantics and replacement on re-arm.
- Yield sequence (flush -> cancel TTS -> abort LLM) within one loop turn (INV-L6).
- YieldCompleted emission on all-acked or 100 ms hard timeout when cancels hang.
- assert_quiescent() verification of 0 live tasks/streams/timers (INV-S1).
- Stale-epoch event filtering helper.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

_tests_dir = Path(__file__).resolve().parents[2]
_root_dir = str(_tests_dir.parent)
_fakes_dir = str(_tests_dir / "fakes")
_helpers_dir = str(_tests_dir / "helpers")

for _p in (_root_dir, _fakes_dir, _helpers_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import overlap
import overlap.runtime

_local_overlap = str(Path(_root_dir) / "overlap")
if _local_overlap not in overlap.__path__:
    overlap.__path__.insert(0, _local_overlap)

_local_runtime = str(Path(_root_dir) / "overlap" / "runtime")
if _local_runtime not in overlap.runtime.__path__:
    overlap.runtime.__path__.insert(0, _local_runtime)

import pytest
from llm import FakeLlmStream
from tts import FakeTtsStream
from virtual import advance_virtual, run_virtual

from overlap.core.clock import VirtualClock
from overlap.core.types import (
    AbortLlm,
    ArmTimer,
    CancelResponse,
    CancelTimer,
    CancelTts,
    CloseOverlapWindow,
    FadeOutClip,
    FlushPlayout,
    MergeUserTurn,
    OpenOverlapWindow,
    PlayClip,
    SetGain,
    StartPlayout,
    StartResponse,
    Tag,
    TelemetryEffect,
    TimerFired,
    TruncateHistory,
    YieldCompleted,
)
from overlap.ports.telemetry import TelemetrySink
from overlap.runtime.supervisor import (
    StreamSupervisor,
    filter_stale_epoch,
    is_stale_epoch,
)


class RecordingTelemetrySink(TelemetrySink):
    """Recording telemetry sink for verifying logged events."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int, dict[str, Any]]] = []

    def emit(self, name: str, ts_ns: int, **fields: Any) -> None:
        self.events.append((name, ts_ns, fields))

    def has_event(self, name: str) -> bool:
        return any(e[0] == name for e in self.events)


class MockMixer:
    """Mock playout mixer recording port calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def set_gain(self, gain: float, ramp_ms: int = 0) -> None:
        self.calls.append(("set_gain", {"gain": gain, "ramp_ms": ramp_ms}))

    def flush(self) -> None:
        self.calls.append(("flush", {}))

    def play_clip(self, clip: str, tag: Tag = Tag.SELF_BACKCHANNEL) -> None:
        self.calls.append(("play_clip", {"clip": clip, "tag": tag}))

    def fade_out_clip(self, dur_ms: int = 40) -> None:
        self.calls.append(("fade_out_clip", {"dur_ms": dur_ms}))

    def start_playout(self) -> None:
        self.calls.append(("start_playout", {}))


class MockTransport:
    """Mock transport recording flush calls."""

    def __init__(self) -> None:
        self.flushed = False

    def flush_output(self) -> None:
        self.flushed = True


# =============================================================================
# 1. Timers via Clock & ArmTimer / CancelTimer Semantics
# =============================================================================


def test_arm_timer_fires_at_deadline() -> None:
    clock = VirtualClock(start_ns=1_000_000_000)

    async def _test() -> None:
        emitted: list[Any] = []
        sup = StreamSupervisor(clock=clock, on_event=emitted.append)

        # Arm timer for +300 ms
        deadline_ns = clock.now_ns() + 300_000_000
        sup.arm_timer("classify_cap", deadline_ns)

        assert "classify_cap" in sup.active_timers
        assert not emitted

        # Advance 200 ms -> should not fire yet
        await advance_virtual(clock, 200)
        assert not emitted
        assert "classify_cap" in sup.active_timers

        # Advance 100 ms -> deadline reached -> fires TimerFired("classify_cap")
        await advance_virtual(clock, 100)
        assert len(emitted) == 1
        assert emitted[0] == TimerFired(name="classify_cap")
        assert "classify_cap" not in sup.active_timers

    run_virtual(_test(), clock=clock)


def test_cancel_timer_prevents_firing() -> None:
    clock = VirtualClock()

    async def _test() -> None:
        emitted: list[Any] = []
        sup = StreamSupervisor(clock=clock, on_event=emitted.append)

        sup.arm_timer("test_timer", clock.now_ns() + 200_000_000)
        assert "test_timer" in sup.active_timers

        # Cancel timer before deadline
        cancelled = sup.cancel_timer("test_timer")
        assert cancelled is True
        assert "test_timer" not in sup.active_timers

        # Advance past original deadline
        await advance_virtual(clock, 300)
        assert not emitted

    run_virtual(_test(), clock=clock)


def test_cancel_timer_idempotent() -> None:
    clock = VirtualClock()
    sup = StreamSupervisor(clock=clock)
    assert sup.cancel_timer("non_existent") is False
    assert sup.cancel_timer("non_existent") is False


def test_rearm_timer_replaces_previous() -> None:
    """Requirement: same-name re-arm replaces; old timer never fires."""
    clock = VirtualClock()

    async def _test() -> None:
        emitted: list[Any] = []
        sup = StreamSupervisor(clock=clock, on_event=emitted.append)

        # Arm initial timer at +100 ms
        sup.arm_timer("cap", clock.now_ns() + 100_000_000)

        # Advance 50 ms
        await advance_virtual(clock, 50)

        # Re-arm same name for +200 ms (deadline = now + 200 ms = 250 ms from start)
        sup.arm_timer("cap", clock.now_ns() + 200_000_000)

        # Advance to 120 ms from start (past old 100 ms deadline)
        await advance_virtual(clock, 70)
        # Old timer must NOT have fired!
        assert not emitted

        # Advance past new deadline (another 150 ms, reaching 270 ms)
        await advance_virtual(clock, 150)
        assert len(emitted) == 1
        assert emitted[0] == TimerFired(name="cap")

    run_virtual(_test(), clock=clock)


def test_cancel_all_timers() -> None:
    clock = VirtualClock()

    async def _test() -> None:
        emitted: list[Any] = []
        sup = StreamSupervisor(clock=clock, on_event=emitted.append)

        sup.arm_timer("t1", clock.now_ns() + 100_000_000)
        sup.arm_timer("t2", clock.now_ns() + 200_000_000)
        assert len(sup.active_timers) == 2

        sup.cancel_all_timers()
        assert len(sup.active_timers) == 0

        await advance_virtual(clock, 300)
        assert not emitted

    run_virtual(_test(), clock=clock)


# =============================================================================
# 2. Cancel Tokens & Task/Stream Management (no leaked tasks)
# =============================================================================


def test_turn_token_lifecycle_and_cascade() -> None:
    sup = StreamSupervisor()
    token1 = sup.start_turn("turn-1", epoch=1)
    assert sup.current_turn_id == "turn-1"
    assert sup.current_epoch == 1
    assert not token1.cancelled

    child1 = sup.create_child_token("tts")
    child2 = sup.create_child_token("llm")
    assert not child1.cancelled
    assert not child2.cancelled

    # Canceling turn token cascades to children
    sup.cancel_turn()
    assert token1.cancelled
    assert child1.cancelled
    assert child2.cancelled


def test_spawn_task_auto_cancellation_and_cleanup() -> None:
    clock = VirtualClock()

    async def _test() -> None:
        sup = StreamSupervisor(clock=clock)
        sup.start_turn("turn-1", epoch=1)

        task_cancelled = False

        async def worker() -> None:
            nonlocal task_cancelled
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                task_cancelled = True
                raise

        task = sup.spawn_task(worker(), name="test_worker")
        assert task in sup.active_tasks
        assert not task.done()

        # Let worker start and wait on the future
        await advance_virtual(clock, 0)

        # Canceling turn should cancel task
        sup.cancel_turn()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert task.done()
        assert task.cancelled()
        assert task_cancelled
        assert task not in sup.active_tasks

    run_virtual(_test(), clock=clock)


def test_spawn_task_normal_completion_cleans_up() -> None:
    async def _test() -> None:
        sup = StreamSupervisor()

        async def quick_worker() -> str:
            return "done"

        task = sup.spawn_task(quick_worker(), name="quick")
        assert task in sup.active_tasks
        await task
        assert task.done()
        assert task not in sup.active_tasks

    run_virtual(_test())


def test_stream_registration_and_cancellation() -> None:
    async def _test() -> None:
        sup = StreamSupervisor()
        sup.start_turn("turn-1", epoch=1)

        tts_token = sup.create_child_token("tts")
        tts_stream = FakeTtsStream(epoch=1, voice_id="v1", lang="hi", cancel=tts_token)
        sup.register_tts_stream(tts_stream, cancel=tts_token)

        llm_token = sup.create_child_token("llm")
        llm_stream = FakeLlmStream(deltas=["a", "b"], epoch=1, cancel=llm_token)
        sup.register_llm_stream(llm_stream, cancel=llm_token)

        assert tts_stream in sup.active_streams
        assert llm_stream in sup.active_streams

        # Cancel turn cascades to tokens
        sup.cancel_turn()
        assert tts_stream._cancelled
        assert llm_stream._aborted

        sup.unregister_stream(tts_stream)
        sup.unregister_stream(llm_stream)
        assert len(sup.active_streams) == 0

    run_virtual(_test())


# =============================================================================
# 3. Effect Executor & Error Isolation
# =============================================================================


def test_effects_executed_strictly_in_order() -> None:
    """Failure mode: effects executed out of order (flush before gain)."""
    clock = VirtualClock()
    mixer = MockMixer()
    telemetry = RecordingTelemetrySink()
    sup = StreamSupervisor(clock=clock, mixer=mixer, telemetry=telemetry)

    # In YIELDED: FlushPlayout before SetGain(1.0)
    effects = (
        FlushPlayout(),
        TruncateHistory(played_pos_ms=120),
        SetGain(gain=1.0, ramp_ms=0),
        CancelTimer(name="classify_cap"),
    )
    sup.execute_effects(effects)

    assert mixer.calls == [
        ("flush", {}),
        ("set_gain", {"gain": 1.0, "ramp_ms": 0}),
    ]


def test_effect_failure_isolated_and_logged_never_raises() -> None:
    """Done when: failures isolated per effect, logged to telemetry, never raised into frame."""
    clock = VirtualClock()
    telemetry = RecordingTelemetrySink()
    order: list[str] = []

    def bad_flush() -> None:
        order.append("flush")
        raise RuntimeError("Mixer flush failed")

    def good_gain(gain: float, ramp_ms: int) -> None:
        order.append(f"gain:{gain}")

    sup = StreamSupervisor(
        clock=clock,
        telemetry=telemetry,
        flush_handler=bad_flush,
        gain_handler=good_gain,
    )

    # Flush fails, but SetGain must still execute without raising
    sup.execute_effects((FlushPlayout(), SetGain(gain=1.0, ramp_ms=0)))

    assert order == ["flush", "gain:1.0"]
    assert telemetry.has_event("supervisor.effect_error")
    err_events = [e for e in telemetry.events if e[0] == "supervisor.effect_error"]
    assert err_events[0][2]["effect"] == "FlushPlayout"
    assert "Mixer flush failed" in err_events[0][2]["error"]


def test_all_effects_mapped_to_ports() -> None:
    """Every Effect type must be mapped to the appropriate port/handler."""
    clock = VirtualClock()
    mixer = MockMixer()
    transport = MockTransport()
    telemetry = RecordingTelemetrySink()

    truncated: list[int] = []
    overlap_windows: list[bool] = []
    merged_turns: list[str] = []

    sup = StreamSupervisor(
        clock=clock,
        mixer=mixer,
        transport=transport,
        telemetry=telemetry,
        history_truncator=truncated.append,
        overlap_window_handler=overlap_windows.append,
        merge_turn_handler=merged_turns.append,
    )

    sup.execute_effects(
        (
            StartResponse(turn_id="turn-99", epoch=5),
            SetGain(gain=0.3, ramp_ms=20),
            OpenOverlapWindow(),
            ArmTimer(name="test_timer", deadline_ns=clock.now_ns() + 100_000_000),
            PlayClip(clip="haan", tag=Tag.SELF_BACKCHANNEL),
            FadeOutClip(dur_ms=40),
            StartPlayout(),
            CloseOverlapWindow(),
            FlushPlayout(),
            TruncateHistory(played_pos_ms=450),
            MergeUserTurn(text="previous text"),
            CancelResponse(),
            CancelTimer(name="test_timer"),
            TelemetryEffect(name="custom.metric", payload=(("val", 42),)),
        )
    )

    assert sup.current_turn_id == "turn-99"
    assert sup.current_epoch == 5
    assert truncated == [450]
    assert overlap_windows == [True, False]
    assert merged_turns == ["previous text"]
    assert transport.flushed
    assert telemetry.has_event("custom.metric")
    assert any(c[0] == "play_clip" for c in mixer.calls)
    assert any(c[0] == "fade_out_clip" for c in mixer.calls)
    assert any(c[0] == "start_playout" for c in mixer.calls)


# =============================================================================
# 4. Yield Sequence & YieldCompleted Coordination (INV-L6)
# =============================================================================


def test_yield_sequence_issued_within_one_loop_turn() -> None:
    """Done when: yield sequence (flush -> cancel TTS -> abort LLM) issued in 1 turn (INV-L6)."""
    clock = VirtualClock()

    async def _test() -> None:
        mixer = MockMixer()
        emitted: list[Any] = []
        sup = StreamSupervisor(clock=clock, mixer=mixer, on_event=emitted.append)
        sup.start_turn("turn-1", epoch=1)

        tts_stream = FakeTtsStream(epoch=1, voice_id="v1", lang="hi")
        llm_stream = FakeLlmStream(deltas=["test"], epoch=1)
        sup.register_tts_stream(tts_stream)
        sup.register_llm_stream(llm_stream)

        # Execute yield sequence
        effects = (
            FlushPlayout(),
            CancelTts(),
            AbortLlm(),
            TruncateHistory(played_pos_ms=0),
            SetGain(gain=1.0, ramp_ms=0),
            CancelTimer(name="classify_cap"),
        )
        sup.execute_effects(effects)

        # Flush issued immediately
        assert any(c[0] == "flush" for c in mixer.calls)

        # TTS and LLM cancellations issued within one loop turn
        await advance_virtual(clock, 0)
        assert tts_stream._cancelled
        assert llm_stream._aborted

        # YieldCompleted emitted after all acked
        assert YieldCompleted() in emitted

    run_virtual(_test(), clock=clock)


def test_yield_completed_emitted_on_100ms_hard_timeout_when_cancel_hangs() -> None:
    """Failure mode: YieldCompleted never emitted when a cancel hangs (no hard timeout)."""

    class HangingTtsStream:
        def __init__(self) -> None:
            self.cancel_called = False

        async def cancel(self) -> None:
            self.cancel_called = True
            # Never completes
            await asyncio.Future()

    clock = VirtualClock()

    async def _test() -> None:
        telemetry = RecordingTelemetrySink()
        emitted: list[Any] = []
        sup = StreamSupervisor(
            clock=clock,
            telemetry=telemetry,
            on_event=emitted.append,
            yield_timeout_ms=100,
        )
        sup.start_turn("turn-1", epoch=1)

        hanging_tts = HangingTtsStream()
        sup.register_tts_stream(hanging_tts)

        # Trigger yield sequence
        sup.execute_effects((FlushPlayout(), CancelTts(), AbortLlm()))

        # Yield sequence issued
        await advance_virtual(clock, 0)
        assert hanging_tts.cancel_called
        # Not yet acked, so YieldCompleted not emitted yet
        assert YieldCompleted() not in emitted

        # Advance 50 ms -> still waiting
        await advance_virtual(clock, 50)
        assert YieldCompleted() not in emitted

        # Advance past 100 ms hard timeout
        await advance_virtual(clock, 50)
        # 100 ms reached -> YieldCompleted MUST be emitted!
        assert YieldCompleted() in emitted
        assert telemetry.has_event("supervisor.yield_timeout")

    run_virtual(_test(), clock=clock)


def test_yield_with_no_active_streams() -> None:
    clock = VirtualClock()

    async def _test() -> None:
        emitted: list[Any] = []
        sup = StreamSupervisor(clock=clock, on_event=emitted.append)
        sup.start_turn("turn-1", epoch=1)

        # Yield with 0 active streams
        sup.execute_effects((FlushPlayout(), CancelTts(), AbortLlm()))
        await advance_virtual(clock, 0)

        assert YieldCompleted() in emitted

    run_virtual(_test(), clock=clock)


# =============================================================================
# 5. Quiescence Check (INV-S1)
# =============================================================================


def test_assert_quiescent_passes_when_clean() -> None:
    sup = StreamSupervisor()
    # Initially quiescent
    sup.assert_quiescent()
    assert sup.is_quiescent()


def test_assert_quiescent_fails_with_live_timer() -> None:
    clock = VirtualClock()
    sup = StreamSupervisor(clock=clock)
    sup.arm_timer("t1", clock.now_ns() + 100_000_000)

    with pytest.raises(AssertionError, match="live timers:.*t1"):
        sup.assert_quiescent()

    sup.cancel_timer("t1")
    sup.assert_quiescent()


def test_assert_quiescent_fails_with_live_stream() -> None:
    sup = StreamSupervisor()
    stream = FakeTtsStream(epoch=1, voice_id="v", lang="hi")
    sup.register_stream(stream)

    with pytest.raises(AssertionError, match="live streams"):
        sup.assert_quiescent()

    sup.unregister_stream(stream)
    sup.assert_quiescent()


def test_assert_quiescent_fails_with_live_task() -> None:
    async def _test() -> None:
        sup = StreamSupervisor()

        async def loop_forever() -> None:
            await asyncio.Future()

        task = sup.spawn_task(loop_forever(), name="eternal")
        with pytest.raises(AssertionError, match="live tasks:.*eternal"):
            sup.assert_quiescent()

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)

        sup.assert_quiescent()

    run_virtual(_test())


def test_shutdown_cleans_up_to_quiescent() -> None:
    clock = VirtualClock()

    async def _test() -> None:
        sup = StreamSupervisor(clock=clock)
        sup.start_turn("turn-1", epoch=1)

        sup.arm_timer("t1", clock.now_ns() + 500_000_000)

        tts = FakeTtsStream(epoch=1, voice_id="v", lang="hi")
        sup.register_tts_stream(tts)

        async def worker() -> None:
            await asyncio.Future()

        sup.spawn_task(worker(), name="worker_task")

        assert not sup.is_quiescent()

        await sup.shutdown()
        sup.assert_quiescent()

    run_virtual(_test(), clock=clock)


# =============================================================================
# 6. Stale-epoch Filter Helper
# =============================================================================


class DummyEvent:
    def __init__(self, epoch: int | None = None) -> None:
        self.epoch = epoch


def test_is_stale_epoch_pure_helper() -> None:
    # Older epoch is stale
    assert is_stale_epoch(DummyEvent(epoch=1), current_epoch=2) is True
    assert filter_stale_epoch(DummyEvent(epoch=1), current_epoch=2) is False

    # Same epoch is NOT stale
    assert is_stale_epoch(DummyEvent(epoch=2), current_epoch=2) is False
    assert filter_stale_epoch(DummyEvent(epoch=2), current_epoch=2) is True

    # Newer epoch is NOT stale
    assert is_stale_epoch(DummyEvent(epoch=3), current_epoch=2) is False

    # Event without epoch attribute is NOT stale
    assert is_stale_epoch(object(), current_epoch=2) is False
    assert is_stale_epoch(DummyEvent(epoch=None), current_epoch=2) is False


def test_supervisor_is_stale_logs_telemetry() -> None:
    telemetry = RecordingTelemetrySink()
    sup = StreamSupervisor(telemetry=telemetry)
    sup.start_turn("turn-1", epoch=3)

    stale_event = DummyEvent(epoch=2)
    assert sup.is_stale(stale_event) is True
    assert sup.filter_event(stale_event) is False
    assert telemetry.has_event("supervisor.stale_epoch_dropped")

    fresh_event = DummyEvent(epoch=3)
    assert sup.is_stale(fresh_event) is False
    assert sup.filter_event(fresh_event) is True
