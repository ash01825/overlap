# ruff: noqa: E402
"""Scenario DSL for composing full-duplex conversational test timelines."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, cast

_fakes_dir = str(Path(__file__).resolve().parent.parent / "fakes")
if _fakes_dir not in sys.path:
    sys.path.insert(0, _fakes_dir)

from sink import RecordingSink
from stt import FakeStt
from transport import FakeTransport
from tts import FakeTts
from vad import FakeVad

from overlap.core.clock import VirtualClock
from overlap.core.types import AudioFrame, Effect, FsmState


class EventPriority(IntEnum):
    """Deterministic priority order for timeline events occurring at the exact same millisecond."""

    TIMER = 10
    AGENT_TTS = 20
    USER_VAD = 30
    USER_STT = 40
    USER_TURN = 50
    INSPECTION = 90


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    """An event on the compiled scenario timeline."""

    t_ms: int
    priority: int
    seq_id: int
    kind: str
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Expectation:
    """An assertion expected to hold during or after scenario execution."""

    kind: str
    target: Any
    start_ms: int | None = None
    end_ms: int | None = None


@dataclass(frozen=True, slots=True)
class Trace:
    """The execution trace resulting from running a scenario."""

    states: list[tuple[int, FsmState]]
    effects: list[tuple[int, Effect]]
    frames_out: list[AudioFrame]
    telemetry: list[Any]
    flush_timestamps_ms: list[int] = field(default_factory=list)

    def assert_state(self, expected_state: FsmState | str, at_ms: int | None = None) -> None:
        """Assert FSM state at a specific time or final state."""
        expected_str = (
            expected_state.value if isinstance(expected_state, FsmState) else expected_state
        )
        if at_ms is None:
            assert self.states, "No states recorded in trace"
            last_state = self.states[-1][1]
            last_str = last_state.value if isinstance(last_state, FsmState) else str(last_state)
            assert last_str == expected_str, f"Expected final state {expected_str}, got {last_str}"
            return

        # Find closest state at or before at_ms
        matching = [s for (t, s) in self.states if t <= at_ms]
        assert matching, f"No state recorded at or before t={at_ms} ms"
        cur_state = matching[-1]
        cur_str = cur_state.value if isinstance(cur_state, FsmState) else str(cur_state)
        assert cur_str == expected_str, (
            f"Expected state {expected_str} at t={at_ms} ms, got {cur_str}"
        )

    def assert_no_flush(self) -> None:
        """Assert no audio flush occurred in trace."""
        assert len(self.flush_timestamps_ms) == 0, (
            f"Expected no audio flush, but found flushes at {self.flush_timestamps_ms} ms"
        )

    def assert_flush_within(self, start_ms: int, end_ms: int) -> None:
        """Assert at least one audio flush occurred between start_ms and end_ms."""
        matching = [t for t in self.flush_timestamps_ms if start_ms <= t <= end_ms]
        assert matching, (
            f"Expected flush between {start_ms} and {end_ms} ms, "
            f"flushes occurred at {self.flush_timestamps_ms}"
        )


@dataclass
class CompiledScenario:
    """Compiled scenario with deterministic timeline ready for execution."""

    name: str
    timeline: list[TimelineEvent]
    expectations: list[Expectation]
    fakes: dict[str, Any]


class Scenario:
    """Fluent Scenario DSL for authoring conversational test cases."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._raw_events: list[dict[str, Any]] = []
        self._expectations: list[Expectation] = []
        self._seq_counter = 0

    def _next_seq(self) -> int:
        self._seq_counter += 1
        return self._seq_counter

    def agent_says(self, text: str, dur_ms: int | None = None) -> Scenario:
        """Schedule agent TTS speech starting at current timeline end or 0."""
        self._raw_events.append(
            {
                "kind": "agent_says",
                "text": text,
                "dur_ms": dur_ms,
                "priority": EventPriority.AGENT_TTS,
                "seq_id": self._next_seq(),
            }
        )
        return self

    def user_overlap(self, at_ms: int, text: str = "", dur_ms: int = 300) -> Scenario:
        """Schedule user speech overlapping agent playout at at_ms."""
        self._raw_events.append(
            {
                "kind": "user_overlap",
                "at_ms": at_ms,
                "text": text,
                "dur_ms": dur_ms,
                "priority": EventPriority.USER_VAD,
                "seq_id": self._next_seq(),
            }
        )
        return self

    def user_turn(self, at_ms: int, text: str = "", dur_ms: int = 1000) -> Scenario:
        """Schedule user conversational turn at at_ms."""
        self._raw_events.append(
            {
                "kind": "user_turn",
                "at_ms": at_ms,
                "text": text,
                "dur_ms": dur_ms,
                "priority": EventPriority.USER_TURN,
                "seq_id": self._next_seq(),
            }
        )
        return self

    def expect_state(
        self,
        state: FsmState | str,
        at_ms: int | None = None,
        within_ms: int | None = None,
    ) -> Scenario:
        """Add expected state expectation."""
        self._expectations.append(
            Expectation(
                kind="state",
                target=state,
                start_ms=at_ms,
                end_ms=within_ms,
            )
        )
        return self

    def expect_no_flush(self) -> Scenario:
        """Add expectation that no audio flush occurs."""
        self._expectations.append(Expectation(kind="no_flush", target=None))
        return self

    def expect_flush_within(self, start_ms: int, end_ms: int) -> Scenario:
        """Add expectation of audio flush within time window."""
        self._expectations.append(
            Expectation(
                kind="flush_within",
                target=None,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
        return self

    def compile(self) -> CompiledScenario:
        """Compile scenario into a deterministically ordered timeline over fakes."""
        clock = VirtualClock()
        stt = FakeStt(clock=clock)
        tts = FakeTts(clock=clock)
        vad = FakeVad()
        transport = FakeTransport(clock=clock)
        sink = RecordingSink()

        timeline: list[TimelineEvent] = []
        cur_time_ms = 0

        for item in self._raw_events:
            kind = item["kind"]
            priority = int(item["priority"])
            seq_id = int(item["seq_id"])

            if kind == "agent_says":
                t_ms = cur_time_ms
                dur = item["dur_ms"] or 1000
                cur_time_ms += dur
                timeline.append(
                    TimelineEvent(
                        t_ms=t_ms,
                        priority=priority,
                        seq_id=seq_id,
                        kind=kind,
                        data=item,
                    )
                )
            elif kind in ("user_overlap", "user_turn"):
                t_ms = int(item["at_ms"])
                cur_time_ms = max(cur_time_ms, t_ms + int(item["dur_ms"]))
                timeline.append(
                    TimelineEvent(
                        t_ms=t_ms,
                        priority=priority,
                        seq_id=seq_id,
                        kind=kind,
                        data=item,
                    )
                )

        # Failure mode 5 prevention: deterministic sort key (t_ms, priority, seq_id)
        timeline.sort(key=lambda e: (e.t_ms, e.priority, e.seq_id))

        return CompiledScenario(
            name=self.name,
            timeline=timeline,
            expectations=list(self._expectations),
            fakes={
                "clock": clock,
                "stt": stt,
                "tts": tts,
                "vad": vad,
                "transport": transport,
                "sink": sink,
            },
        )

    def run(self, runtime: Any = None, clock: VirtualClock | None = None) -> Trace:
        """Compile and execute scenario, verifying expectations."""
        compiled = self.compile()
        if runtime is not None:
            # When runtime exists, execute through runtime
            trace = cast(Trace, runtime.execute(compiled))
        else:
            # In Phase 0 before runtime/FSM is integrated, synthesize basic trace
            states: list[tuple[int, FsmState]] = []
            flush_timestamps_ms: list[int] = []
            for ev in compiled.timeline:
                if ev.kind == "agent_says":
                    states.append((ev.t_ms, FsmState.SPEAKING))
                elif ev.kind == "user_overlap":
                    states.append((ev.t_ms, FsmState.DUCKED))
                elif ev.kind == "user_turn":
                    states.append((ev.t_ms, FsmState.LISTENING))

            trace = Trace(
                states=states,
                effects=[],
                frames_out=[],
                telemetry=[],
                flush_timestamps_ms=flush_timestamps_ms,
            )

        # Verify expectations
        for exp in compiled.expectations:
            if exp.kind == "state":
                trace.assert_state(exp.target, at_ms=exp.start_ms)
            elif exp.kind == "no_flush":
                trace.assert_no_flush()
            elif exp.kind == "flush_within":
                assert exp.start_ms is not None and exp.end_ms is not None
                trace.assert_flush_within(exp.start_ms, exp.end_ms)

        return trace
