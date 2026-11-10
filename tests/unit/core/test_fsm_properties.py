"""Hypothesis property tests for pure FSM invariants per ARCHITECTURE §6.

Validates all 7 invariants of ARCHITECTURE §6 using Hypothesis with
`derandomize=True` and `max_examples <= 50`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from overlap.core.fsm import (
    FsmContext,
    transition,
    transition_context,
)
from overlap.core.types import (
    AbortLlm,
    AgentAudioDrained,
    ArmTimer,
    BackchannelDue,
    CancelTimer,
    CancelTts,
    ClipDone,
    EndOfTurn,
    FirstAudioReady,
    FlushPlayout,
    FsmState,
    Label,
    PlayClip,
    SetGain,
    StartResponse,
    SttEvent,
    SttPartial,
    Tag,
    TelemetryEffect,
    TimerFired,
    TransportError,
    TruncateHistory,
    Tuning,
    VadOffset,
    VadOnset,
    Verdict,
    YieldCompleted,
)


@dataclass(frozen=True, slots=True)
class _EpochEvent:
    """Helper event with epoch field to test stale epoch handling."""

    epoch: int
    name: str = "stale_test"


@dataclass(frozen=True, slots=True)
class _UnknownEvent:
    """Arbitrary unexpected event."""

    payload: str = "unknown"


# -----------------------------------------------------------------------------
# Strategies
# -----------------------------------------------------------------------------

state_strategy = st.sampled_from(list(FsmState))

label_strategy = st.sampled_from(list(Label))

tag_strategy = st.sampled_from(
    [
        Tag.NONE,
        Tag.SELF_BACKCHANNEL,
        Tag.SELF_TTS,
        Tag.SELF_BACKCHANNEL | Tag.SELF_TTS,
    ]
)

event_strategy = st.one_of(
    st.builds(
        EndOfTurn,
        final=st.builds(
            SttEvent,
            kind=st.just("final"),
            text=st.text(min_size=1, max_size=20),
            ts_ns=st.integers(min_value=0, max_value=1_000_000_000),
        ),
    ),
    st.builds(BackchannelDue, clip=st.text(min_size=1, max_size=10)),
    st.builds(ClipDone),
    st.builds(FirstAudioReady),
    st.builds(
        VadOnset,
        ts_ns=st.integers(min_value=0, max_value=10_000_000_000),
        is_echo=st.booleans(),
        tag=tag_strategy,
    ),
    st.builds(VadOffset, ts_ns=st.integers(min_value=0, max_value=10_000_000_000)),
    st.builds(AgentAudioDrained),
    st.builds(SttPartial, text=st.text(max_size=20)),
    st.builds(
        Verdict,
        label=label_strategy,
        confidence=st.floats(min_value=0.0, max_value=1.0),
        reason=st.text(max_size=10),
        overlap_ms=st.integers(min_value=0, max_value=2000),
        decided_at_ns=st.integers(min_value=0, max_value=10_000_000_000),
    ),
    st.builds(
        TimerFired,
        name=st.sampled_from(["classify_cap", "bc_clip_end", "thinking_barge_in", "other_timer"]),
    ),
    st.builds(YieldCompleted),
    st.builds(TransportError, message=st.text(max_size=20)),
    st.builds(_EpochEvent, epoch=st.integers(min_value=0, max_value=10)),
    st.builds(_UnknownEvent, payload=st.text(max_size=10)),
)

context_strategy = st.builds(
    FsmContext,
    state=state_strategy,
    epoch=st.integers(min_value=0, max_value=10),
    turn_id=st.text(max_size=10),
    turn_text=st.text(max_size=20),
    user_vad_active=st.booleans(),
    thinking_vad_start_ns=st.one_of(st.none(), st.integers(min_value=0, max_value=10_000_000_000)),
    last_backchannel_ns=st.one_of(st.none(), st.integers(min_value=0, max_value=10_000_000_000)),
    reescalation_count=st.integers(min_value=0, max_value=5),
    played_pos_ms=st.integers(min_value=0, max_value=10_000),
)

now_strategy = st.integers(min_value=0, max_value=10_000_000_000)

tuning_strategy = st.builds(
    Tuning,
    duck_gain=st.floats(min_value=0.1, max_value=0.5),
    duck_attack_ms=st.integers(min_value=10, max_value=40),
    duck_release_ms=st.integers(min_value=50, max_value=300),
    classify_cap_ms=st.integers(min_value=200, max_value=600),
    recheck_window_ms=st.integers(min_value=500, max_value=3000),
    barge_in_thinking_ms=st.integers(min_value=50, max_value=200),
    failsafe_bias=st.sampled_from([Label.INTERRUPT, Label.BACKCHANNEL]),
)


# -----------------------------------------------------------------------------
# Invariant 1: Total and never raises; unknown events are no-ops
# -----------------------------------------------------------------------------


@settings(derandomize=True, max_examples=50)
@given(
    state=state_strategy,
    event=event_strategy,
    now_ns=now_strategy,
    tuning=tuning_strategy,
)
def test_inv1_transition_is_total_and_never_raises(
    state: FsmState,
    event: Any,
    now_ns: int,
    tuning: Tuning,
) -> None:
    """Inv 1: transition never raises on any (state, event, now, tuning) combination."""
    next_state, effects = transition(state, event, now_ns=now_ns, tuning=tuning)
    assert isinstance(next_state, FsmState)
    assert isinstance(effects, tuple)


@settings(derandomize=True, max_examples=50)
@given(
    state=state_strategy,
    payload=st.text(min_size=1, max_size=20),
    now_ns=now_strategy,
)
def test_inv1_unknown_events_are_noops(
    state: FsmState,
    payload: str,
    now_ns: int,
) -> None:
    """Inv 1: unknown events do not mutate state and emit fsm.ignored telemetry."""
    event = _UnknownEvent(payload=payload)
    next_state, effects = transition(state, event, now_ns=now_ns)
    assert next_state == state
    assert len(effects) == 1
    assert isinstance(effects[0], TelemetryEffect)
    assert effects[0].name == "fsm.ignored"


# -----------------------------------------------------------------------------
# Invariant 2: DUCKED => classify_cap timer armed; leaving DUCKED cancels timer
# -----------------------------------------------------------------------------


@settings(derandomize=True, max_examples=50)
@given(
    ctx=context_strategy,
    event=event_strategy,
    now_ns=now_strategy,
    tuning=tuning_strategy,
)
def test_inv2_ducked_arms_timer_and_leaving_cancels_timer(
    ctx: FsmContext,
    event: Any,
    now_ns: int,
    tuning: Tuning,
) -> None:
    """Inv 2: Entering DUCKED arms classify_cap; leaving DUCKED cancels classify_cap."""
    next_ctx, effects = transition_context(ctx, event, now_ns=now_ns, tuning=tuning)

    # If transitioning into DUCKED from another state: classify_cap MUST be armed
    if ctx.state != FsmState.DUCKED and next_ctx.state == FsmState.DUCKED:
        armed_timers = [e.name for e in effects if isinstance(e, ArmTimer)]
        assert "classify_cap" in armed_timers

    # If leaving DUCKED to another state: classify_cap MUST be cancelled
    if ctx.state == FsmState.DUCKED and next_ctx.state != FsmState.DUCKED:
        cancelled_timers = [e.name for e in effects if isinstance(e, CancelTimer)]
        assert "classify_cap" in cancelled_timers


# -----------------------------------------------------------------------------
# Invariant 3: Entering YIELDED emits FlushPlayout, CancelTts, AbortLlm,
#              TruncateHistory exactly once and bumps epoch
# -----------------------------------------------------------------------------


@settings(derandomize=True, max_examples=50)
@given(
    ctx=context_strategy,
    event=event_strategy,
    now_ns=now_strategy,
    tuning=tuning_strategy,
)
def test_inv3_yielded_emits_flush_cancel_abort_truncate_and_bumps_epoch(
    ctx: FsmContext,
    event: Any,
    now_ns: int,
    tuning: Tuning,
) -> None:
    """Inv 3: Entering YIELDED emits flush/cancel TTS/abort LLM/truncate and bumps epoch."""
    next_ctx, effects = transition_context(ctx, event, now_ns=now_ns, tuning=tuning)

    if ctx.state != FsmState.YIELDED and next_ctx.state == FsmState.YIELDED:
        flush_count = sum(1 for e in effects if isinstance(e, FlushPlayout))
        cancel_tts_count = sum(1 for e in effects if isinstance(e, CancelTts))
        abort_llm_count = sum(1 for e in effects if isinstance(e, AbortLlm))
        truncate_count = sum(1 for e in effects if isinstance(e, TruncateHistory))

        assert flush_count == 1
        assert cancel_tts_count == 1
        assert abort_llm_count == 1
        assert truncate_count == 1
        assert next_ctx.epoch == ctx.epoch + 1


# -----------------------------------------------------------------------------
# Invariant 4: After YIELDED -> LISTENING, no effect with old epoch is emitted
# -----------------------------------------------------------------------------


@settings(derandomize=True, max_examples=50)
@given(
    epoch=st.integers(min_value=0, max_value=20),
    now_ns=now_strategy,
    tuning=tuning_strategy,
)
def test_inv4_stale_epoch_events_dropped_after_yielded(
    epoch: int,
    now_ns: int,
    tuning: Tuning,
) -> None:
    """Inv 4: After YIELDED -> LISTENING, old epoch events are dropped."""
    # 1. Start in DUCKED with epoch E
    ctx = FsmContext(state=FsmState.DUCKED, epoch=epoch)

    # 2. Transition DUCKED -> YIELDED (epoch bumps to epoch + 1)
    yield_event = Verdict(
        label=Label.INTERRUPT,
        confidence=1.0,
        reason="interrupt",
        overlap_ms=300,
        decided_at_ns=now_ns,
    )
    yielded_ctx, _ = transition_context(ctx, yield_event, now_ns=now_ns, tuning=tuning)
    assert yielded_ctx.state == FsmState.YIELDED
    assert yielded_ctx.epoch == epoch + 1

    # 3. Complete yield -> LISTENING
    listening_ctx, _ = transition_context(
        yielded_ctx, YieldCompleted(), now_ns=now_ns + 10, tuning=tuning
    )
    assert listening_ctx.state == FsmState.LISTENING
    assert listening_ctx.epoch == epoch + 1

    # 4. Any event with the old epoch (<= epoch) must be dropped
    stale_event = _EpochEvent(epoch=epoch)
    after_stale_ctx, stale_effects = transition_context(
        listening_ctx, stale_event, now_ns=now_ns + 20, tuning=tuning
    )
    assert after_stale_ctx == listening_ctx
    assert len(stale_effects) == 1
    assert isinstance(stale_effects[0], TelemetryEffect)
    assert stale_effects[0].name == "fsm.ignored"

    # 5. Starting next response uses the new epoch (epoch + 1), never old epoch
    turn_event = EndOfTurn(final=SttEvent(kind="final", text="new turn"))
    thinking_ctx, turn_effects = transition_context(
        listening_ctx, turn_event, now_ns=now_ns + 30, tuning=tuning
    )
    assert thinking_ctx.state == FsmState.THINKING
    start_resp = [e for e in turn_effects if isinstance(e, StartResponse)][0]
    assert start_resp.epoch == epoch + 1


# -----------------------------------------------------------------------------
# Invariant 5: Gain is 1.0 in every state except DUCKED
# -----------------------------------------------------------------------------


@settings(derandomize=True, max_examples=50)
@given(
    ctx=context_strategy,
    event=event_strategy,
    now_ns=now_strategy,
    tuning=tuning_strategy,
)
def test_inv5_gain_is_one_in_every_state_except_ducked(
    ctx: FsmContext,
    event: Any,
    now_ns: int,
    tuning: Tuning,
) -> None:
    """Inv 5: Gain is 1.0 outside DUCKED; leaving DUCKED restores gain to 1.0."""
    next_ctx, effects = transition_context(ctx, event, now_ns=now_ns, tuning=tuning)

    # Leaving DUCKED must emit SetGain(1.0)
    if ctx.state == FsmState.DUCKED and next_ctx.state != FsmState.DUCKED:
        gain_effects = [e for e in effects if isinstance(e, SetGain)]
        assert len(gain_effects) >= 1
        assert gain_effects[0].gain == 1.0

    # Entering DUCKED must emit SetGain(duck_gain < 1.0)
    if ctx.state != FsmState.DUCKED and next_ctx.state == FsmState.DUCKED:
        gain_effects = [e for e in effects if isinstance(e, SetGain)]
        assert len(gain_effects) >= 1
        assert gain_effects[0].gain == tuning.duck_gain
        assert gain_effects[0].gain < 1.0


# -----------------------------------------------------------------------------
# Invariant 6: Backchannels only from LISTENING; self-audio never ducks
# -----------------------------------------------------------------------------


@settings(derandomize=True, max_examples=50)
@given(
    state=state_strategy,
    clip=st.text(min_size=1, max_size=10),
    now_ns=now_strategy,
)
def test_inv6_backchannels_only_from_listening(
    state: FsmState,
    clip: str,
    now_ns: int,
) -> None:
    """Inv 6a: Backchannel clip is only started from LISTENING state."""
    event = BackchannelDue(clip=clip)
    next_state, effects = transition(state, event, now_ns=now_ns)

    play_clips = [e for e in effects if isinstance(e, PlayClip)]
    if state == FsmState.LISTENING:
        assert next_state == FsmState.BACKCHANNELING
        assert len(play_clips) == 1
    else:
        assert next_state == state
        assert len(play_clips) == 0


@settings(derandomize=True, max_examples=50)
@given(
    is_echo=st.booleans(),
    has_self_tag=st.booleans(),
    now_ns=now_strategy,
    tuning=tuning_strategy,
)
def test_inv6_self_audio_and_echo_never_trigger_ducking(
    is_echo: bool,
    has_self_tag: bool,
    now_ns: int,
    tuning: Tuning,
) -> None:
    """Inv 6b: Echo and SELF_BACKCHANNEL/SELF_TTS audio never trigger ducking."""
    if not is_echo and not has_self_tag:
        return  # test only self-audio or echo conditions

    tag = Tag.SELF_BACKCHANNEL if has_self_tag else Tag.NONE
    event = VadOnset(ts_ns=now_ns, is_echo=is_echo, tag=tag)

    next_state, effects = transition(FsmState.SPEAKING, event, now_ns=now_ns, tuning=tuning)
    assert next_state == FsmState.SPEAKING
    assert not any(isinstance(e, SetGain) for e in effects)


# -----------------------------------------------------------------------------
# Invariant 7: State and effects are deterministic functions of (state, event, now, tuning)
# -----------------------------------------------------------------------------


@settings(derandomize=True, max_examples=50)
@given(
    ctx=context_strategy,
    event=event_strategy,
    now_ns=now_strategy,
    tuning=tuning_strategy,
)
def test_inv7_transition_is_deterministic(
    ctx: FsmContext,
    event: Any,
    now_ns: int,
    tuning: Tuning,
) -> None:
    """Inv 7: transition is pure and deterministic given identical inputs."""
    next_ctx1, effects1 = transition_context(ctx, event, now_ns=now_ns, tuning=tuning)
    next_ctx2, effects2 = transition_context(ctx, event, now_ns=now_ns, tuning=tuning)

    assert next_ctx1 == next_ctx2
    assert effects1 == effects2
