# ruff: noqa: E402
"""Unit tests for Scenario DSL and deterministic timeline compilation."""

from __future__ import annotations

import sys
from pathlib import Path

_fixtures_dir = str(Path(__file__).resolve().parents[2] / "fixtures")
if _fixtures_dir not in sys.path:
    sys.path.insert(0, _fixtures_dir)

import pytest
from scenario import Scenario, Trace

from overlap.core.types import AudioFrame, FsmState


def test_scenario_dsl_fluent_api_and_compilation() -> None:
    scenario = (
        Scenario("test_overlap")
        .agent_says("Namaste, main aapki kya madad kar sakta hoon?", dur_ms=1500)
        .user_overlap(at_ms=1200, text="haan", dur_ms=300)
        .user_turn(at_ms=2000, text="mera account balance batao", dur_ms=1000)
        .expect_state(FsmState.SPEAKING, at_ms=500)
        .expect_no_flush()
        .expect_flush_within(1200, 1600)
    )

    compiled = scenario.compile()
    timeline = compiled.timeline

    # Verify timeline is sorted monotonically by t_ms
    prev_t = -1
    for event in timeline:
        assert event.t_ms >= prev_t
        prev_t = event.t_ms

    assert len(timeline) >= 3


def test_scenario_dsl_timeline_deterministic_tie_breaking() -> None:
    """Failure mode 5: DSL timeline overlaps resolved non-deterministically."""
    # Create two events at the exact same millisecond
    scenario1 = (
        Scenario("s1")
        .agent_says("hello", dur_ms=500)
        .user_overlap(at_ms=500, text="haan", dur_ms=200)
        .user_turn(at_ms=500, text="arre suno", dur_ms=400)
    )
    scenario2 = (
        Scenario("s2")
        .agent_says("hello", dur_ms=500)
        .user_overlap(at_ms=500, text="haan", dur_ms=200)
        .user_turn(at_ms=500, text="arre suno", dur_ms=400)
    )

    c1 = scenario1.compile()
    c2 = scenario2.compile()

    # Identical order and actions
    assert len(c1.timeline) == len(c2.timeline)
    for e1, e2 in zip(c1.timeline, c2.timeline, strict=True):
        assert e1.t_ms == e2.t_ms
        assert e1.kind == e2.kind
        assert e1.priority == e2.priority
        assert e1.seq_id == e2.seq_id


def test_scenario_trace_assertions() -> None:
    trace = Trace(
        states=[(0, FsmState.LISTENING), (500, FsmState.SPEAKING), (1200, FsmState.DUCKED)],
        effects=[],
        frames_out=[
            AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=0),
            AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=20_000_000),
        ],
        telemetry=[],
        flush_timestamps_ms=[1300],
    )

    # State assertions
    trace.assert_state(FsmState.SPEAKING, at_ms=500)
    trace.assert_state(FsmState.DUCKED, at_ms=1200)
    with pytest.raises(AssertionError):
        trace.assert_state(FsmState.YIELDED, at_ms=500)

    # Flush assertions
    trace.assert_flush_within(1200, 1400)
    with pytest.raises(AssertionError):
        trace.assert_flush_within(100, 500)

    # expect_no_flush on a trace that had flushes should fail
    with pytest.raises(AssertionError):
        trace.assert_no_flush()

    trace_no_flush = Trace(
        states=[],
        effects=[],
        frames_out=[],
        telemetry=[],
        flush_timestamps_ms=[],
    )
    trace_no_flush.assert_no_flush()


def test_scenario_run_with_expectations() -> None:
    scenario = (
        Scenario("run_test")
        .agent_says("hello", dur_ms=500)
        .user_overlap(at_ms=200, text="haan", dur_ms=100)
        .expect_state(FsmState.SPEAKING, at_ms=0)
        .expect_state(FsmState.DUCKED, at_ms=200)
        .expect_no_flush()
    )
    trace = scenario.run()
    assert len(trace.states) == 2
    assert trace.states[0] == (0, FsmState.SPEAKING)
    assert trace.states[1] == (200, FsmState.DUCKED)
