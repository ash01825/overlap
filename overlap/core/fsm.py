"""Pure Finite State Machine transition table, effects, timers-as-data, and epochs.

Per ARCHITECTURE §6:
Pure: `transition(state: FsmState, event: Event, now_ns: int, tuning: Tuning)
      -> tuple[FsmState, tuple[Effect, ...]]`

Invariants:
1. `transition` is total and never raises; unknown events are no-ops.
2. `DUCKED =>` a `classify_cap` timer is armed; leaving `DUCKED` always emits `CancelTimer`.
3. Entering `YIELDED` emits exactly one each of `FlushPlayout, CancelTts, AbortLlm, TruncateHistory`
   and bumps `epoch`.
4. After `YIELDED -> LISTENING`, no effect with the old epoch is ever emitted again.
5. Gain is 1.0 in every state except `DUCKED`.
6. Backchannel clips are only started from `LISTENING`; events tagged `SELF_BACKCHANNEL`
   never trigger `SPEAKING -> DUCKED`.
7. State and effects are deterministic functions of `(state, event, now, tuning)`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, overload

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
    Effect,
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

DEFAULT_MAX_REESCALATIONS_PER_TURN: int = 2


@dataclass(frozen=True, slots=True)
class FsmContext:
    """Immutable state context for FSM transitions.

    Holds the discrete FsmState plus turn, epoch, timer, and escalation bookkeeping
    required by ARCHITECTURE §3 and §6.
    """

    state: FsmState = FsmState.LISTENING
    epoch: int = 0
    turn_id: str = ""
    turn_text: str = ""
    user_vad_active: bool = False
    thinking_vad_start_ns: int | None = None
    last_backchannel_ns: int | None = None
    reescalation_count: int = 0
    max_reescalations: int = DEFAULT_MAX_REESCALATIONS_PER_TURN
    played_pos_ms: int = 0


def transition_context(
    ctx: FsmContext,
    event: Any,
    now_ns: int = 0,
    tuning: Tuning | None = None,
) -> tuple[FsmContext, tuple[Effect, ...]]:
    """Pure context transition function.

    Takes an FsmContext, event, virtual timestamp, and tuning configuration,
    and returns the new FsmContext and tuple of emitted effects.
    """
    cfg = tuning if tuning is not None else Tuning()

    # Drop stale-epoch events (ARCHITECTURE §3.2, §6 row 16)
    event_epoch = getattr(event, "epoch", None)
    if event_epoch is not None and isinstance(event_epoch, int) and event_epoch < ctx.epoch:
        return (
            ctx,
            (
                TelemetryEffect(
                    name="fsm.ignored",
                    payload=(
                        ("reason", "stale_epoch"),
                        ("event_epoch", event_epoch),
                        ("current_epoch", ctx.epoch),
                    ),
                ),
            ),
        )

    # Global handler: TransportError from any state (ARCHITECTURE §6 row 15)
    if isinstance(event, TransportError):
        next_ctx = replace(
            ctx,
            state=FsmState.LISTENING,
            epoch=ctx.epoch + 1,
            thinking_vad_start_ns=None,
            user_vad_active=False,
            last_backchannel_ns=None,
            reescalation_count=0,
        )
        effects: tuple[Effect, ...] = (
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
                payload=(("message", event.message),),
            ),
        )
        return (next_ctx, effects)

    current_state = ctx.state

    # -------------------------------------------------------------------------
    # LISTENING
    # -------------------------------------------------------------------------
    if current_state == FsmState.LISTENING:
        if isinstance(event, EndOfTurn):
            # Row 1: LISTENING + EndOfTurn -> THINKING + StartResponse(turn_id, epoch)
            turn_id = ctx.turn_id if ctx.turn_id else f"turn-{ctx.epoch}"
            next_ctx = replace(
                ctx,
                state=FsmState.THINKING,
                turn_id=turn_id,
                turn_text=event.final.text,
                reescalation_count=0,
                last_backchannel_ns=None,
            )
            return (
                next_ctx,
                (StartResponse(turn_id=turn_id, epoch=ctx.epoch),),
            )

        if isinstance(event, BackchannelDue):
            # Row 2: LISTENING + BackchannelDue(clip) -> BACKCHANNELING + PlayClip + ArmTimer
            next_ctx = replace(ctx, state=FsmState.BACKCHANNELING)
            deadline_ns = (
                now_ns + 1_000_000_000 if now_ns > 0 else 1_000_000_000
            )  # 1s watchdog timer
            return (
                next_ctx,
                (
                    PlayClip(clip=event.clip, tag=Tag.SELF_BACKCHANNEL),
                    ArmTimer(name="bc_clip_end", deadline_ns=deadline_ns),
                ),
            )

    # -------------------------------------------------------------------------
    # BACKCHANNELING
    # -------------------------------------------------------------------------
    elif current_state == FsmState.BACKCHANNELING:
        if isinstance(event, ClipDone):
            # Row 3: BACKCHANNELING + ClipDone -> LISTENING
            next_ctx = replace(ctx, state=FsmState.LISTENING)
            return (next_ctx, ())

        if isinstance(event, TimerFired) and event.name == "bc_clip_end":
            # Clip watchdog timer fired -> LISTENING
            next_ctx = replace(ctx, state=FsmState.LISTENING)
            return (next_ctx, ())

        if isinstance(event, EndOfTurn):
            # Row 4: BACKCHANNELING + EndOfTurn -> THINKING + FadeOutClip + StartResponse
            turn_id = ctx.turn_id if ctx.turn_id else f"turn-{ctx.epoch}"
            next_ctx = replace(
                ctx,
                state=FsmState.THINKING,
                turn_id=turn_id,
                turn_text=event.final.text,
                reescalation_count=0,
                last_backchannel_ns=None,
            )
            return (
                next_ctx,
                (
                    FadeOutClip(dur_ms=40),
                    StartResponse(turn_id=turn_id, epoch=ctx.epoch),
                ),
            )

    # -------------------------------------------------------------------------
    # THINKING
    # -------------------------------------------------------------------------
    elif current_state == FsmState.THINKING:
        if isinstance(event, FirstAudioReady):
            # Row 5: THINKING + FirstAudioReady -> SPEAKING + StartPlayout
            next_ctx = replace(
                ctx,
                state=FsmState.SPEAKING,
                reescalation_count=0,
                last_backchannel_ns=None,
            )
            return (next_ctx, (StartPlayout(),))

        if isinstance(event, VadOnset):
            # Self-tagged audio or echo is ignored
            if event.is_echo or bool(event.tag & (Tag.SELF_BACKCHANNEL | Tag.SELF_TTS)):
                return (
                    ctx,
                    (TelemetryEffect(name="fsm.ignored", payload=(("reason", "echo_or_self"),)),),
                )

            # Row 6: THINKING + VadOnset sustained >= barge_in_thinking_ms -> LISTENING
            barge_in_ns = cfg.barge_in_thinking_ms * 1_000_000
            elapsed_ns = (
                now_ns - event.ts_ns if event.ts_ns > 0 else (now_ns if now_ns > 0 else barge_in_ns)
            )

            if elapsed_ns >= barge_in_ns:
                # Sustained: immediate barge-in cancellation and merge
                next_ctx = replace(
                    ctx,
                    state=FsmState.LISTENING,
                    thinking_vad_start_ns=None,
                )
                return (
                    next_ctx,
                    (
                        CancelResponse(),
                        MergeUserTurn(text=ctx.turn_text),
                    ),
                )

            # Not yet sustained: arm watchdog deadline timer
            deadline_ns = event.ts_ns + barge_in_ns
            next_ctx = replace(ctx, thinking_vad_start_ns=event.ts_ns)
            return (
                next_ctx,
                (ArmTimer(name="thinking_barge_in", deadline_ns=deadline_ns),),
            )

        if isinstance(event, TimerFired) and event.name == "thinking_barge_in":
            # Barge-in timer fired: sustained user speech confirmed
            next_ctx = replace(
                ctx,
                state=FsmState.LISTENING,
                thinking_vad_start_ns=None,
            )
            return (
                next_ctx,
                (
                    CancelResponse(),
                    MergeUserTurn(text=ctx.turn_text),
                ),
            )

        if isinstance(event, VadOffset):
            # Speech stopped before sustained threshold
            next_ctx = replace(ctx, thinking_vad_start_ns=None)
            return (next_ctx, (CancelTimer(name="thinking_barge_in"),))

    # -------------------------------------------------------------------------
    # SPEAKING
    # -------------------------------------------------------------------------
    elif current_state == FsmState.SPEAKING:
        if isinstance(event, VadOnset):
            # Invariant 6: Echo and SELF_BACKCHANNEL/SELF_TTS never trigger ducking
            if event.is_echo or bool(event.tag & (Tag.SELF_BACKCHANNEL | Tag.SELF_TTS)):
                return (
                    ctx,
                    (TelemetryEffect(name="fsm.ignored", payload=(("reason", "echo_or_self"),)),),
                )

            # Row 7: SPEAKING + VadOnset -> DUCKED
            # Inline contract: SetGain MUST be emitted first in effects
            next_ctx = replace(ctx, state=FsmState.DUCKED, user_vad_active=True)
            classify_deadline = now_ns + cfg.classify_cap_ms * 1_000_000
            effects = (
                SetGain(gain=cfg.duck_gain, ramp_ms=cfg.duck_attack_ms),
                OpenOverlapWindow(),
                ArmTimer(name="classify_cap", deadline_ns=classify_deadline),
            )
            return (next_ctx, effects)

        if isinstance(event, AgentAudioDrained):
            # Row 8: SPEAKING + AgentAudioDrained -> LISTENING
            next_ctx = replace(ctx, state=FsmState.LISTENING)
            return (next_ctx, ())

        if isinstance(event, SttPartial):
            # Row 9: Late escalation within recheck_window_ms
            text = event.text.strip()
            if not text:
                return (
                    ctx,
                    (TelemetryEffect(name="fsm.ignored", payload=(("reason", "empty_partial"),)),),
                )

            # Guarded by recheck_window_ms
            recheck_window_ns = cfg.recheck_window_ms * 1_000_000
            last_bc = ctx.last_backchannel_ns
            if last_bc is not None and (now_ns - last_bc) > recheck_window_ns:
                return (
                    ctx,
                    (
                        TelemetryEffect(
                            name="fsm.ignored",
                            payload=(("reason", "recheck_window_expired"),),
                        ),
                    ),
                )

            # Guarded by max re-escalations per turn to prevent infinite loops
            if ctx.reescalation_count >= ctx.max_reescalations:
                return (
                    ctx,
                    (
                        TelemetryEffect(
                            name="fsm.ignored",
                            payload=(("reason", "max_reescalations_reached"),),
                        ),
                    ),
                )

            # Escalate: DUCKED (skip attack => ramp_ms=0)
            next_ctx = replace(
                ctx,
                state=FsmState.DUCKED,
                reescalation_count=ctx.reescalation_count + 1,
            )
            classify_deadline = now_ns + cfg.classify_cap_ms * 1_000_000
            effects = (
                SetGain(gain=cfg.duck_gain, ramp_ms=0),
                OpenOverlapWindow(),
                ArmTimer(name="classify_cap", deadline_ns=classify_deadline),
            )
            return (next_ctx, effects)

        if isinstance(event, VadOffset):
            next_ctx = replace(ctx, user_vad_active=False)
            return (next_ctx, ())

    # -------------------------------------------------------------------------
    # DUCKED
    # -------------------------------------------------------------------------
    elif current_state == FsmState.DUCKED:
        if isinstance(event, Verdict):
            if event.label in (Label.BACKCHANNEL, Label.SIDE_TALK):
                # Row 10: DUCKED + Verdict(BACKCHANNEL | SIDE_TALK) -> SPEAKING
                next_ctx = replace(ctx, state=FsmState.SPEAKING, last_backchannel_ns=now_ns)
                effects = (
                    SetGain(gain=1.0, ramp_ms=cfg.duck_release_ms),
                    CancelTimer(name="classify_cap"),
                    CloseOverlapWindow(),
                )
                return (next_ctx, effects)

            if event.label == Label.INTERRUPT:
                # Row 11: DUCKED + Verdict(INTERRUPT) -> YIELDED + bump epoch
                next_ctx = replace(ctx, state=FsmState.YIELDED, epoch=ctx.epoch + 1)
                effects = (
                    FlushPlayout(),
                    CancelTts(),
                    AbortLlm(),
                    TruncateHistory(played_pos_ms=ctx.played_pos_ms),
                    SetGain(gain=1.0, ramp_ms=0),
                    CancelTimer(name="classify_cap"),
                )
                return (next_ctx, effects)

            if event.label == Label.PENDING:
                # Remains in DUCKED
                return (ctx, ())

        if isinstance(event, TimerFired) and event.name == "classify_cap":
            # Row 12: DUCKED + TimerFired(classify_cap) per fail-safe policy (§7.4)
            if cfg.failsafe_bias == Label.INTERRUPT:
                next_ctx = replace(ctx, state=FsmState.YIELDED, epoch=ctx.epoch + 1)
                effects = (
                    FlushPlayout(),
                    CancelTts(),
                    AbortLlm(),
                    TruncateHistory(played_pos_ms=ctx.played_pos_ms),
                    SetGain(gain=1.0, ramp_ms=0),
                    CancelTimer(name="classify_cap"),
                )
                return (next_ctx, effects)

            # Bias towards BACKCHANNEL / SIDE_TALK
            next_ctx = replace(ctx, state=FsmState.SPEAKING, last_backchannel_ns=now_ns)
            effects = (
                SetGain(gain=1.0, ramp_ms=cfg.duck_release_ms),
                CancelTimer(name="classify_cap"),
                CloseOverlapWindow(),
            )
            return (next_ctx, effects)

        if isinstance(event, AgentAudioDrained):
            # Row 13: DUCKED + AgentAudioDrained -> LISTENING + SetGain(1.0)
            next_ctx = replace(ctx, state=FsmState.LISTENING)
            effects = (
                SetGain(gain=1.0, ramp_ms=0),
                CancelTimer(name="classify_cap"),
                CloseOverlapWindow(),
            )
            return (next_ctx, effects)

        if isinstance(event, VadOffset):
            next_ctx = replace(ctx, user_vad_active=False)
            return (next_ctx, ())

        if isinstance(event, VadOnset):
            next_ctx = replace(ctx, user_vad_active=True)
            return (next_ctx, ())

    # -------------------------------------------------------------------------
    # YIELDED
    # -------------------------------------------------------------------------
    elif current_state == FsmState.YIELDED:
        if isinstance(event, YieldCompleted):
            # Row 14: YIELDED + YieldCompleted -> LISTENING
            next_ctx = replace(ctx, state=FsmState.LISTENING)
            return (next_ctx, ())

    # -------------------------------------------------------------------------
    # Unknown / unexpected event for current state -> no-op
    # -------------------------------------------------------------------------
    return (
        ctx,
        (
            TelemetryEffect(
                name="fsm.ignored",
                payload=(
                    ("reason", "unexpected_event"),
                    ("state", current_state.value),
                    ("event", type(event).__name__),
                ),
            ),
        ),
    )


@overload
def transition(
    state: FsmState,
    event: Any,
    now_ns: int = 0,
    tuning: Tuning | None = None,
) -> tuple[FsmState, tuple[Effect, ...]]: ...


@overload
def transition(
    state: FsmContext,
    event: Any,
    now_ns: int = 0,
    tuning: Tuning | None = None,
) -> tuple[FsmContext, tuple[Effect, ...]]: ...


def transition(
    state: FsmState | FsmContext,
    event: Any,
    now_ns: int = 0,
    tuning: Tuning | None = None,
) -> tuple[FsmState | FsmContext, tuple[Effect, ...]]:
    """Pure transition function per ARCHITECTURE §6.

    Accepts either an `FsmState` enum value or an `FsmContext` data object.
    When given an `FsmState`, returns `(new_state: FsmState, effects)`.
    When given an `FsmContext`, returns `(new_context: FsmContext, effects)`.
    """
    if isinstance(state, FsmState):
        ctx = FsmContext(state=state)
        next_ctx, effects = transition_context(ctx, event, now_ns, tuning)
        return (next_ctx.state, effects)

    return transition_context(state, event, now_ns, tuning)


class FSM:
    """Stateful wrapper managing FsmContext across transitions."""

    def __init__(
        self,
        tuning: Tuning | None = None,
        initial_state: FsmState = FsmState.LISTENING,
        initial_epoch: int = 0,
    ) -> None:
        self._ctx = FsmContext(state=initial_state, epoch=initial_epoch)
        self._tuning = tuning if tuning is not None else Tuning()

    @property
    def state(self) -> FsmState:
        """Current discrete FSM state."""
        return self._ctx.state

    @property
    def epoch(self) -> int:
        """Current agent turn epoch."""
        return self._ctx.epoch

    @property
    def context(self) -> FsmContext:
        """Current full context."""
        return self._ctx

    def transition(self, event: Any, now_ns: int = 0) -> tuple[Effect, ...]:
        """Apply an event to the state machine and return emitted effects."""
        self._ctx, effects = transition_context(self._ctx, event, now_ns, self._tuning)
        return effects
