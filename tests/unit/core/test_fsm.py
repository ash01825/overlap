"""Unit tests for pure FSM transitions per ARCHITECTURE §6.

Covers every row of the transition table (>= 25 tests), edge cases,
failure modes, and inline fast-path contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

from overlap.core.fsm import (
    DEFAULT_MAX_REESCALATIONS_PER_TURN,
    FSM,
    FsmContext,
    transition,
    transition_context,
)
from overlap.core.types import (
    AbortLlm,
    AgentAudioDrained,
    ArmTimer,
    BackchannelDue,
    CancelResponse,
    CancelTimer,
    CancelTts,
    ClipDone,
    CloseOverlapWindow,
    EndOfTurn,
    FadeOutClip,
    FirstAudioReady,
    FlushPlayout,
    FsmState,
    Label,
    MergeUserTurn,
    OpenOverlapWindow,
    PlayClip,
    SetGain,
    StartPlayout,
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


def test_listening_end_of_turn_starts_response() -> None:
    """Row 1: LISTENING + EndOfTurn -> THINKING + StartResponse(turn_id, epoch)."""
    final = SttEvent(kind="final", text="Namaste", ts_ns=100)
    event = EndOfTurn(final=final)
    tuning = Tuning()
    next_state, effects = transition(FsmState.LISTENING, event, now_ns=1000, tuning=tuning)

    assert next_state == FsmState.THINKING
    assert len(effects) == 1
    assert isinstance(effects[0], StartResponse)
    assert effects[0].epoch == 0


def test_listening_backchannel_due_plays_clip() -> None:
    """Row 2: LISTENING + BackchannelDue(clip) -> BACKCHANNELING + PlayClip + ArmTimer."""
    event = BackchannelDue(clip="haan", lang="hi")
    tuning = Tuning()
    now_ns = 500_000_000
    next_state, effects = transition(FsmState.LISTENING, event, now_ns=now_ns, tuning=tuning)

    assert next_state == FsmState.BACKCHANNELING
    assert len(effects) == 2
    assert effects[0] == PlayClip(clip="haan", tag=Tag.SELF_BACKCHANNEL)
    assert isinstance(effects[1], ArmTimer)
    assert effects[1].name == "bc_clip_end"
    assert effects[1].deadline_ns > now_ns


def test_backchanneling_clip_done_returns_to_listening() -> None:
    """Row 3: BACKCHANNELING + ClipDone -> LISTENING with no effects."""
    event = ClipDone()
    next_state, effects = transition(FsmState.BACKCHANNELING, event, now_ns=1000)

    assert next_state == FsmState.LISTENING
    assert effects == ()


def test_backchanneling_clip_timer_fired_returns_to_listening() -> None:
    """Watchdog: BACKCHANNELING + TimerFired('bc_clip_end') -> LISTENING."""
    event = TimerFired(name="bc_clip_end")
    next_state, effects = transition(FsmState.BACKCHANNELING, event, now_ns=1000)

    assert next_state == FsmState.LISTENING
    assert effects == ()


def test_backchanneling_end_of_turn_fades_clip_and_starts_response() -> None:
    """Row 4: BACKCHANNELING + EndOfTurn -> THINKING + FadeOutClip(<=40ms) + StartResponse."""
    final = SttEvent(kind="final", text="Haan bolo", ts_ns=200)
    event = EndOfTurn(final=final)
    next_state, effects = transition(FsmState.BACKCHANNELING, event, now_ns=1000)

    assert next_state == FsmState.THINKING
    assert len(effects) == 2
    assert isinstance(effects[0], FadeOutClip)
    assert effects[0].dur_ms <= 40
    assert isinstance(effects[1], StartResponse)


def test_thinking_first_audio_ready_transitions_to_speaking() -> None:
    """Row 5: THINKING + FirstAudioReady -> SPEAKING + StartPlayout."""
    event = FirstAudioReady()
    next_state, effects = transition(FsmState.THINKING, event, now_ns=1000)

    assert next_state == FsmState.SPEAKING
    assert effects == (StartPlayout(),)


def test_thinking_vad_onset_sustained_barges_in() -> None:
    """Row 6: THINKING + sustained VadOnset >= barge_in_thinking_ms -> LISTENING."""
    tuning = Tuning(barge_in_thinking_ms=100)
    barge_ns = 100 * 1_000_000
    onset_ts = 1_000_000_000
    now_ns = onset_ts + barge_ns  # sustained for 100 ms

    # Pre-populate context with turn text from a previous final
    ctx = FsmContext(state=FsmState.THINKING, turn_text="Previous turn question")
    event = VadOnset(ts_ns=onset_ts)

    next_ctx, effects = transition_context(ctx, event, now_ns=now_ns, tuning=tuning)
    assert next_ctx.state == FsmState.LISTENING
    assert effects == (
        CancelResponse(),
        MergeUserTurn(text="Previous turn question"),
    )


def test_thinking_vad_onset_arms_timer_when_not_yet_sustained() -> None:
    """THINKING + VadOnset with delta < barge_in_thinking_ms arms deadline timer."""
    tuning = Tuning(barge_in_thinking_ms=100)
    onset_ts = 1_000_000_000
    now_ns = onset_ts + 20 * 1_000_000  # only 20 ms elapsed

    ctx = FsmContext(state=FsmState.THINKING)
    event = VadOnset(ts_ns=onset_ts)

    next_ctx, effects = transition_context(ctx, event, now_ns=now_ns, tuning=tuning)
    assert next_ctx.state == FsmState.THINKING
    assert len(effects) == 1
    assert isinstance(effects[0], ArmTimer)
    assert effects[0].name == "thinking_barge_in"
    assert effects[0].deadline_ns == onset_ts + 100 * 1_000_000


def test_thinking_barge_in_timer_fired() -> None:
    """THINKING + TimerFired('thinking_barge_in') executes barge-in transition."""
    ctx = FsmContext(state=FsmState.THINKING, turn_text="Held response")
    event = TimerFired(name="thinking_barge_in")

    next_ctx, effects = transition_context(ctx, event, now_ns=1000)
    assert next_ctx.state == FsmState.LISTENING
    assert effects == (
        CancelResponse(),
        MergeUserTurn(text="Held response"),
    )


def test_thinking_vad_offset_cancels_barge_in_timer() -> None:
    """THINKING + VadOffset cancels thinking_barge_in timer if user speech stops."""
    ctx = FsmContext(state=FsmState.THINKING, thinking_vad_start_ns=1000)
    event = VadOffset(ts_ns=1500)

    next_ctx, effects = transition_context(ctx, event, now_ns=1500)
    assert next_ctx.state == FsmState.THINKING
    assert effects == (CancelTimer(name="thinking_barge_in"),)
    assert next_ctx.thinking_vad_start_ns is None


def test_speaking_vad_onset_ducks_and_emits_set_gain_first() -> None:
    """Row 7: SPEAKING + VadOnset -> DUCKED with SetGain emitted first."""
    tuning = Tuning(duck_gain=0.3, duck_attack_ms=20, classify_cap_ms=400)
    now_ns = 2_000_000_000
    event = VadOnset(ts_ns=now_ns)

    next_state, effects = transition(FsmState.SPEAKING, event, now_ns=now_ns, tuning=tuning)

    assert next_state == FsmState.DUCKED
    assert len(effects) == 3
    # Fast path inline contract: SetGain MUST be index 0
    assert effects[0] == SetGain(gain=0.3, ramp_ms=20)
    assert effects[1] == OpenOverlapWindow()
    assert effects[2] == ArmTimer(
        name="classify_cap",
        deadline_ns=now_ns + 400 * 1_000_000,
    )


def test_speaking_echo_vad_does_not_duck() -> None:
    """VadOnset with is_echo=True is ignored during SPEAKING."""
    event = VadOnset(is_echo=True)
    next_state, effects = transition(FsmState.SPEAKING, event, now_ns=1000)

    assert next_state == FsmState.SPEAKING
    assert len(effects) == 1
    assert isinstance(effects[0], TelemetryEffect)
    assert effects[0].name == "fsm.ignored"


def test_speaking_self_backchannel_tag_does_not_duck() -> None:
    """VadOnset with SELF_BACKCHANNEL tag is ignored during SPEAKING."""
    event = VadOnset(tag=Tag.SELF_BACKCHANNEL)
    next_state, effects = transition(FsmState.SPEAKING, event, now_ns=1000)

    assert next_state == FsmState.SPEAKING
    assert len(effects) == 1
    assert isinstance(effects[0], TelemetryEffect)
    assert effects[0].name == "fsm.ignored"


def test_speaking_self_tts_tag_does_not_duck() -> None:
    """VadOnset with SELF_TTS tag is ignored during SPEAKING."""
    event = VadOnset(tag=Tag.SELF_TTS)
    next_state, effects = transition(FsmState.SPEAKING, event, now_ns=1000)

    assert next_state == FsmState.SPEAKING
    assert len(effects) == 1
    assert isinstance(effects[0], TelemetryEffect)
    assert effects[0].name == "fsm.ignored"


def test_speaking_agent_audio_drained_returns_to_listening() -> None:
    """Row 8: SPEAKING + AgentAudioDrained -> LISTENING."""
    event = AgentAudioDrained()
    next_state, effects = transition(FsmState.SPEAKING, event, now_ns=1000)

    assert next_state == FsmState.LISTENING
    assert effects == ()


def test_speaking_late_escalation_within_recheck_window() -> None:
    """Row 9: SPEAKING + SttPartial within recheck_window_ms -> DUCKED (skip attack)."""
    tuning = Tuning(recheck_window_ms=1500, duck_gain=0.3, classify_cap_ms=400)
    bc_ns = 1_000_000_000
    now_ns = bc_ns + 500 * 1_000_000  # 500 ms elapsed <= 1500 ms

    ctx = FsmContext(
        state=FsmState.SPEAKING,
        last_backchannel_ns=bc_ns,
        user_vad_active=True,
        reescalation_count=0,
    )
    event = SttPartial(text="ruk jao")

    next_ctx, effects = transition_context(ctx, event, now_ns=now_ns, tuning=tuning)

    assert next_ctx.state == FsmState.DUCKED
    assert next_ctx.reescalation_count == 1
    assert len(effects) == 3
    # Skip attack => ramp_ms = 0
    assert effects[0] == SetGain(gain=0.3, ramp_ms=0)
    assert effects[1] == OpenOverlapWindow()
    assert effects[2] == ArmTimer(
        name="classify_cap",
        deadline_ns=now_ns + 400 * 1_000_000,
    )


def test_speaking_late_escalation_outside_window_is_ignored() -> None:
    """SttPartial arriving after recheck_window_ms does NOT late-escalate."""
    tuning = Tuning(recheck_window_ms=1500)
    bc_ns = 1_000_000_000
    now_ns = bc_ns + 1600 * 1_000_000  # 1600 ms > 1500 ms

    ctx = FsmContext(
        state=FsmState.SPEAKING,
        last_backchannel_ns=bc_ns,
        user_vad_active=True,
    )
    event = SttPartial(text="stop")

    next_ctx, effects = transition_context(ctx, event, now_ns=now_ns, tuning=tuning)
    assert next_ctx.state == FsmState.SPEAKING
    assert len(effects) == 1
    assert isinstance(effects[0], TelemetryEffect)
    assert effects[0].name == "fsm.ignored"


def test_speaking_late_escalation_max_counter_blocks_infinite_loop() -> None:
    """Late escalation bounded by max_reescalations to prevent infinite loops."""
    tuning = Tuning(recheck_window_ms=1500)
    bc_ns = 1_000_000_000
    now_ns = bc_ns + 200 * 1_000_000

    ctx = FsmContext(
        state=FsmState.SPEAKING,
        last_backchannel_ns=bc_ns,
        user_vad_active=True,
        reescalation_count=DEFAULT_MAX_REESCALATIONS_PER_TURN,
    )
    event = SttPartial(text="ek minute")

    next_ctx, effects = transition_context(ctx, event, now_ns=now_ns, tuning=tuning)
    assert next_ctx.state == FsmState.SPEAKING
    assert len(effects) == 1
    assert isinstance(effects[0], TelemetryEffect)
    assert effects[0].name == "fsm.ignored"


def test_speaking_empty_stt_partial_is_ignored() -> None:
    """SttPartial with empty or whitespace-only text does not late-escalate."""
    ctx = FsmContext(
        state=FsmState.SPEAKING,
        last_backchannel_ns=1_000_000_000,
        user_vad_active=True,
    )
    event = SttPartial(text="   ")
    next_ctx, effects = transition_context(ctx, event, now_ns=1_100_000_000)

    assert next_ctx.state == FsmState.SPEAKING
    assert len(effects) == 1
    assert isinstance(effects[0], TelemetryEffect)
    assert effects[0].name == "fsm.ignored"


def test_ducked_verdict_backchannel_releases_gain() -> None:
    """Row 10: DUCKED + Verdict(BACKCHANNEL) -> SPEAKING + release gain."""
    tuning = Tuning(duck_release_ms=150)
    event = Verdict(
        label=Label.BACKCHANNEL,
        confidence=0.9,
        reason="lexicon_hit",
        overlap_ms=250,
        decided_at_ns=1000,
    )
    now_ns = 2_000_000_000
    next_state, effects = transition(FsmState.DUCKED, event, now_ns=now_ns, tuning=tuning)

    assert next_state == FsmState.SPEAKING
    assert effects == (
        SetGain(gain=1.0, ramp_ms=150),
        CancelTimer(name="classify_cap"),
        CloseOverlapWindow(),
    )


def test_ducked_verdict_side_talk_releases_gain() -> None:
    """Row 10: DUCKED + Verdict(SIDE_TALK) -> SPEAKING + release gain."""
    tuning = Tuning(duck_release_ms=150)
    event = Verdict(
        label=Label.SIDE_TALK,
        confidence=0.8,
        reason="low_volume",
        overlap_ms=300,
        decided_at_ns=1000,
    )
    next_state, effects = transition(FsmState.DUCKED, event, now_ns=1000, tuning=tuning)

    assert next_state == FsmState.SPEAKING
    assert effects == (
        SetGain(gain=1.0, ramp_ms=150),
        CancelTimer(name="classify_cap"),
        CloseOverlapWindow(),
    )


def test_ducked_verdict_pending_is_noop() -> None:
    """Verdict(PENDING) does not change state or emit effects."""
    event = Verdict(
        label=Label.PENDING,
        confidence=0.5,
        reason="need_more_frames",
        overlap_ms=100,
        decided_at_ns=1000,
    )
    next_state, effects = transition(FsmState.DUCKED, event, now_ns=1000)

    assert next_state == FsmState.DUCKED
    assert effects == ()


def test_ducked_verdict_interrupt_yields_and_bumps_epoch() -> None:
    """Row 11: DUCKED + Verdict(INTERRUPT) -> YIELDED + cancel/truncate/flush + bump epoch."""
    event = Verdict(
        label=Label.INTERRUPT,
        confidence=0.95,
        reason="loud_and_lexicon",
        overlap_ms=350,
        decided_at_ns=1000,
    )
    ctx = FsmContext(state=FsmState.DUCKED, epoch=1, played_pos_ms=420)
    next_ctx, effects = transition_context(ctx, event, now_ns=1000)

    assert next_ctx.state == FsmState.YIELDED
    assert next_ctx.epoch == 2  # bumped
    assert effects == (
        FlushPlayout(),
        CancelTts(),
        AbortLlm(),
        TruncateHistory(played_pos_ms=420),
        SetGain(gain=1.0, ramp_ms=0),
        CancelTimer(name="classify_cap"),
    )


def test_ducked_timer_fired_failsafe_bias_interrupt() -> None:
    """Row 12: DUCKED + TimerFired('classify_cap') with failsafe_bias=INTERRUPT."""
    tuning = Tuning(failsafe_bias=Label.INTERRUPT)
    event = TimerFired(name="classify_cap")
    ctx = FsmContext(state=FsmState.DUCKED, epoch=3, played_pos_ms=100)

    next_ctx, effects = transition_context(ctx, event, now_ns=2000, tuning=tuning)
    assert next_ctx.state == FsmState.YIELDED
    assert next_ctx.epoch == 4
    assert effects == (
        FlushPlayout(),
        CancelTts(),
        AbortLlm(),
        TruncateHistory(played_pos_ms=100),
        SetGain(gain=1.0, ramp_ms=0),
        CancelTimer(name="classify_cap"),
    )


def test_ducked_timer_fired_failsafe_bias_backchannel() -> None:
    """Row 12: DUCKED + TimerFired('classify_cap') with failsafe_bias=BACKCHANNEL."""
    tuning = Tuning(failsafe_bias=Label.BACKCHANNEL, duck_release_ms=150)
    event = TimerFired(name="classify_cap")
    ctx = FsmContext(state=FsmState.DUCKED, epoch=2)

    next_ctx, effects = transition_context(ctx, event, now_ns=2000, tuning=tuning)
    assert next_ctx.state == FsmState.SPEAKING
    assert next_ctx.epoch == 2  # not bumped
    assert effects == (
        SetGain(gain=1.0, ramp_ms=150),
        CancelTimer(name="classify_cap"),
        CloseOverlapWindow(),
    )


def test_ducked_agent_audio_drained_returns_to_listening() -> None:
    """Row 13: DUCKED + AgentAudioDrained -> LISTENING + SetGain(1.0)."""
    event = AgentAudioDrained()
    next_state, effects = transition(FsmState.DUCKED, event, now_ns=1000)

    assert next_state == FsmState.LISTENING
    assert effects == (
        SetGain(gain=1.0, ramp_ms=0),
        CancelTimer(name="classify_cap"),
        CloseOverlapWindow(),
    )


def test_yielded_yield_completed_returns_to_listening() -> None:
    """Row 14: YIELDED + YieldCompleted -> LISTENING."""
    event = YieldCompleted()
    next_state, effects = transition(FsmState.YIELDED, event, now_ns=1000)

    assert next_state == FsmState.LISTENING
    assert effects == ()


def test_transport_error_from_any_state() -> None:
    """Row 15: *any* + TransportError -> LISTENING + cancel everything."""
    event = TransportError(message="WebSocket connection closed unexpectedly")
    for st in FsmState:
        ctx = FsmContext(state=st, epoch=5)
        next_ctx, effects = transition_context(ctx, event, now_ns=1000)

        assert next_ctx.state == FsmState.LISTENING
        assert next_ctx.epoch == 6
        assert effects == (
            CancelTts(),
            AbortLlm(),
            FlushPlayout(),
            SetGain(gain=1.0, ramp_ms=0),
            CancelTimer(name="classify_cap"),
            CancelTimer(name="bc_clip_end"),
            CancelTimer(name="thinking_barge_in"),
            CloseOverlapWindow(),
            TelemetryEffect(
                name="transport.error",
                payload=(("message", "WebSocket connection closed unexpectedly"),),
            ),
        )


def test_stale_epoch_event_is_dropped() -> None:
    """Row 16: stale-epoch event is ignored and does not mutate state."""
    ctx = FsmContext(state=FsmState.SPEAKING, epoch=3)
    event = _EpochEvent(epoch=2)  # stale: 2 < 3

    next_ctx, effects = transition_context(ctx, event, now_ns=1000)
    assert next_ctx == ctx
    assert len(effects) == 1
    assert isinstance(effects[0], TelemetryEffect)
    assert effects[0].name == "fsm.ignored"


def test_current_epoch_event_is_not_dropped_as_stale() -> None:
    """Events matching current epoch are not dropped by the stale check."""
    ctx = FsmContext(state=FsmState.SPEAKING, epoch=2)
    event = AgentAudioDrained()

    next_ctx, effects = transition_context(ctx, event, now_ns=1000)
    assert next_ctx.state == FsmState.LISTENING


def test_unknown_event_never_raises() -> None:
    """Row 16: unknown event type returns same state and fsm.ignored telemetry."""

    @dataclass(frozen=True, slots=True)
    class UnexpectedEvent:
        data: str = "weird"

    for st in FsmState:
        next_state, effects = transition(st, UnexpectedEvent(), now_ns=1000)
        assert next_state == st
        assert len(effects) == 1
        assert isinstance(effects[0], TelemetryEffect)
        assert effects[0].name == "fsm.ignored"


def test_backchannel_due_rejected_when_speaking() -> None:
    """Failure mode: backchannel clip started while agent is speaking is forbidden."""
    event = BackchannelDue(clip="haan")
    next_state, effects = transition(FsmState.SPEAKING, event, now_ns=1000)

    assert next_state == FsmState.SPEAKING
    assert len(effects) == 1
    assert isinstance(effects[0], TelemetryEffect)
    assert effects[0].name == "fsm.ignored"


def test_fsm_stateful_wrapper_lifecycle() -> None:
    """FSM class manages sequential lifecycle with state, context, and epoch."""
    fsm = FSM()
    st0: FsmState = fsm.state
    assert st0 == FsmState.LISTENING
    assert fsm.epoch == 0

    # User finishes turn -> THINKING
    fsm.transition(EndOfTurn(final=SttEvent(kind="final", text="Hello")))
    st1: FsmState = fsm.state
    assert st1 == FsmState.THINKING

    # Audio ready -> SPEAKING
    fsm.transition(FirstAudioReady())
    st2: FsmState = fsm.state
    assert st2 == FsmState.SPEAKING

    # User speaks -> DUCKED
    fsm.transition(VadOnset(ts_ns=100))
    st3: FsmState = fsm.state
    assert st3 == FsmState.DUCKED

    # User interrupts -> YIELDED + epoch bumped
    fsm.transition(
        Verdict(
            label=Label.INTERRUPT,
            confidence=0.9,
            reason="interrupt",
            overlap_ms=300,
            decided_at_ns=200,
        )
    )
    st4: FsmState = fsm.state
    assert st4 == FsmState.YIELDED
    assert fsm.epoch == 1

    # Cleanup acknowledged -> LISTENING
    fsm.transition(YieldCompleted())
    st5: FsmState = fsm.state
    assert st5 == FsmState.LISTENING
    assert fsm.epoch == 1
