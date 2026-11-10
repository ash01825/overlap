"""StreamSupervisor: cancel tokens, timers, effect executor, quiescence check.

Per ARCHITECTURE §3, §6, and §10:
- Coordinates in-flight streams (LLM, TTS, STT, clips) with hierarchical CancelTokens.
- Realizes FSM timer effects (ArmTimer, CancelTimer) via an injected Clock.
- Executes FSM effects sequentially with per-effect error isolation and telemetry logging.
- Manages the yield sequence: flush -> cancel TTS -> abort LLM issued within one turn (INV-L6),
  and emits YieldCompleted when all cancels are acknowledged or on 100 ms hard timeout.
- Provides assert_quiescent() to verify 0 live tasks, streams, and timers (INV-S1).
- Provides pure and supervisor-level stale-epoch event filtering.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from typing import Any, TypeVar

from overlap.core.clock import CancelToken, Clock, MonotonicClock, TimerHandle
from overlap.core.types import (
    AbortLlm,
    ArmTimer,
    CancelResponse,
    CancelTimer,
    CancelTts,
    CloseOverlapWindow,
    Effect,
    Event,
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

T = TypeVar("T")

DEFAULT_YIELD_TIMEOUT_MS: int = 100
YIELD_TIMEOUT_TIMER_NAME: str = "_yield_hard_timeout"


def is_stale_epoch(event: Any, current_epoch: int) -> bool:
    """Return True if event carries an epoch strictly older than current_epoch.

    Events without an integer epoch attribute are never considered stale.
    """
    event_epoch = getattr(event, "epoch", None)
    if event_epoch is not None and isinstance(event_epoch, int):
        return event_epoch < current_epoch
    return False


def filter_stale_epoch(event: Any, current_epoch: int) -> bool:
    """Return True if event should be processed (not stale), False if stale."""
    return not is_stale_epoch(event, current_epoch)


class _YieldSession:
    """Internal tracker for an active yield sequence."""

    def __init__(self, epoch: int) -> None:
        self.epoch = epoch
        self.pending_tasks: set[asyncio.Task[Any]] = set()
        self.completed: bool = False


class StreamSupervisor:
    """Supervisor for stream tasks, timers, effect execution, and quiescence."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        telemetry: TelemetrySink | None = None,
        on_event: Callable[[Event], Any] | None = None,
        mixer: Any = None,
        transport: Any = None,
        gain_handler: Callable[[float, int], Any] | None = None,
        flush_handler: Callable[[], Any] | None = None,
        clip_handler: Callable[[str, Tag], Any] | None = None,
        fade_handler: Callable[[int], Any] | None = None,
        playout_handler: Callable[[], Any] | None = None,
        history_truncator: Callable[[int], Any] | None = None,
        overlap_window_handler: Callable[[bool], Any] | None = None,
        merge_turn_handler: Callable[[str], Any] | None = None,
        yield_timeout_ms: int = DEFAULT_YIELD_TIMEOUT_MS,
    ) -> None:
        self._clock: Clock = clock if clock is not None else MonotonicClock()
        self._telemetry: TelemetrySink | None = telemetry
        self._on_event: Callable[[Event], Any] | None = on_event
        self._mixer: Any = mixer
        self._transport: Any = transport
        self._gain_handler = gain_handler
        self._flush_handler = flush_handler
        self._clip_handler = clip_handler
        self._fade_handler = fade_handler
        self._playout_handler = playout_handler
        self._history_truncator = history_truncator
        self._overlap_window_handler = overlap_window_handler
        self._merge_turn_handler = merge_turn_handler
        self._yield_timeout_ms: int = yield_timeout_ms

        self._current_turn_id: str = ""
        self._current_epoch: int = 0
        self._turn_token: CancelToken = CancelToken()

        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._active_streams: set[Any] = set()
        self._tts_streams: set[Any] = set()
        self._llm_streams: set[Any] = set()
        self._stt_streams: set[Any] = set()

        self._timer_seq: int = 0
        self._timers: dict[str, tuple[int, TimerHandle]] = {}

        self._current_yield_session: _YieldSession | None = None
        self._last_errors: list[tuple[Effect, Exception]] = []
        self._emitted_events: list[Event] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def clock(self) -> Clock:
        """Configured time source."""
        return self._clock

    @property
    def current_turn_id(self) -> str:
        """Active agent turn ID."""
        return self._current_turn_id

    @property
    def current_epoch(self) -> int:
        """Active turn epoch."""
        return self._current_epoch

    @property
    def turn_token(self) -> CancelToken:
        """Root cancel token for the active turn."""
        return self._turn_token

    @property
    def active_tasks(self) -> set[asyncio.Task[Any]]:
        """Set of live tracked asyncio tasks."""
        return set(self._active_tasks)

    @property
    def active_streams(self) -> set[Any]:
        """Set of live tracked streams."""
        return set(self._active_streams)

    @property
    def active_timers(self) -> set[str]:
        """Set of currently armed timer names."""
        return {name for name, (_, handle) in self._timers.items() if not handle.cancelled}

    @property
    def emitted_events(self) -> list[Event]:
        """List of events emitted by this supervisor."""
        return list(self._emitted_events)

    # -------------------------------------------------------------------------
    # Turn & Token Management
    # -------------------------------------------------------------------------

    def start_turn(self, turn_id: str, epoch: int) -> CancelToken:
        """Begin a new turn, setting epoch and creating a new root CancelToken."""
        if not self._turn_token.cancelled:
            self.cancel_turn()

        self._current_turn_id = turn_id
        self._current_epoch = epoch
        self._turn_token = CancelToken()
        return self._turn_token

    def create_child_token(self, name: str = "") -> CancelToken:
        """Create a child CancelToken tied to the active turn token."""
        return self._turn_token.child()

    def cancel_turn(self) -> None:
        """Cancel the active turn token, cascading to all child tokens."""
        if not self._turn_token.cancelled:
            try:
                self._turn_token.cancel()
            except Exception as exc:
                self._log_error("cancel_turn", exc)

    # -------------------------------------------------------------------------
    # Task & Stream Tracking (Leak Prevention)
    # -------------------------------------------------------------------------

    def spawn_task(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        name: str = "",
        cancel: CancelToken | None = None,
    ) -> asyncio.Task[T]:
        """Spawn and track an asyncio Task bound to cancellation and lifecycle cleanup."""
        token = cancel if cancel is not None else self.create_child_token(name=name)
        task = asyncio.create_task(coro, name=name or None)
        self._active_tasks.add(task)

        def _cleanup(t: asyncio.Task[Any]) -> None:
            self._active_tasks.discard(t)

        task.add_done_callback(_cleanup)

        if token.cancelled:
            task.cancel()
        else:
            token.on_cancel(task.cancel)

        return task

    def register_stream(
        self,
        stream: Any,
        *,
        cancel: CancelToken | None = None,
        kind: str = "",
    ) -> None:
        """Register an active stream with optional cancellation token."""
        self._active_streams.add(stream)
        if cancel is not None:
            cancel.on_cancel(lambda: self._cancel_stream_sync_or_async(stream))

    def unregister_stream(self, stream: Any) -> None:
        """Unregister an active stream upon completion or cancellation."""
        self._active_streams.discard(stream)
        self._tts_streams.discard(stream)
        self._llm_streams.discard(stream)
        self._stt_streams.discard(stream)

    def register_tts_stream(
        self,
        stream: Any,
        *,
        cancel: CancelToken | None = None,
    ) -> None:
        """Register an active TTS stream."""
        self._tts_streams.add(stream)
        self.register_stream(stream, cancel=cancel, kind="tts")

    def register_llm_stream(
        self,
        stream: Any,
        *,
        cancel: CancelToken | None = None,
    ) -> None:
        """Register an active LLM stream."""
        self._llm_streams.add(stream)
        self.register_stream(stream, cancel=cancel, kind="llm")

    def register_stt_stream(
        self,
        stream: Any,
        *,
        cancel: CancelToken | None = None,
    ) -> None:
        """Register an active STT stream."""
        self._stt_streams.add(stream)
        self.register_stream(stream, cancel=cancel, kind="stt")

    def _cancel_stream_sync_or_async(self, stream: Any) -> None:
        """Trigger cancellation or abort on a stream object."""
        try:
            cancel_fn = getattr(stream, "cancel", None) or getattr(stream, "abort", None)
            if cancel_fn is not None and callable(cancel_fn):
                res = cancel_fn()
                if asyncio.iscoroutine(res):
                    try:
                        asyncio.get_running_loop()
                        self.spawn_task(res, name="cancel_stream")
                    except RuntimeError:
                        # No running event loop
                        pass
        except Exception as exc:
            self._log_error("cancel_stream", exc)

    # -------------------------------------------------------------------------
    # Timers via Clock
    # -------------------------------------------------------------------------

    def arm_timer(self, name: str, deadline_ns: int) -> TimerHandle:
        """Arm a named deadline timer via clock.call_at.

        Same-name re-arm replaces any previous timer of that name.
        """
        # If already armed, cancel old handle first
        old = self._timers.get(name)
        if old is not None:
            old[1].cancel()

        self._timer_seq += 1
        seq = self._timer_seq

        def _on_fired(timer_name: str, timer_seq: int) -> None:
            current = self._timers.get(timer_name)
            if current is None or current[0] != timer_seq:
                # Cancelled, replaced, or already fired; do nothing
                return
            del self._timers[timer_name]

            if timer_name == YIELD_TIMEOUT_TIMER_NAME:
                self._handle_yield_timeout()
            else:
                self._emit_event(TimerFired(name=timer_name))

        handle = self._clock.call_at(deadline_ns, _on_fired, name, seq)
        self._timers[name] = (seq, handle)
        return handle

    def cancel_timer(self, name: str) -> bool:
        """Disarm a named timer. Safe and idempotent; returns True if active timer was disarmed."""
        entry = self._timers.pop(name, None)
        if entry is not None:
            entry[1].cancel()
            return True
        return False

    def cancel_all_timers(self) -> None:
        """Disarm all currently active timers."""
        timers = list(self._timers.values())
        self._timers.clear()
        for _, handle in timers:
            handle.cancel()

    # -------------------------------------------------------------------------
    # Yield Sequence Coordination (INV-L6)
    # -------------------------------------------------------------------------

    def _start_yield_session(self) -> _YieldSession:
        """Initialize tracking for a yield sequence."""
        if self._current_yield_session is not None and not self._current_yield_session.completed:
            self.cancel_timer(YIELD_TIMEOUT_TIMER_NAME)

        session = _YieldSession(epoch=self._current_epoch)
        self._current_yield_session = session

        # Arm hard 100 ms timeout timer
        timeout_deadline = self._clock.now_ns() + self._yield_timeout_ms * 1_000_000
        self.arm_timer(YIELD_TIMEOUT_TIMER_NAME, timeout_deadline)
        return session

    def _finish_yield_session_setup(self) -> None:
        """Check if all cancellations in yield session were completed immediately."""
        session = self._current_yield_session
        if session is None or session.completed:
            return

        if not session.pending_tasks:
            # All cancels acked immediately!
            session.completed = True
            self.cancel_timer(YIELD_TIMEOUT_TIMER_NAME)
            self._emit_event(YieldCompleted())

    def _on_yield_task_done(self, task: asyncio.Task[Any]) -> None:
        """Callback invoked when a cancel task completes."""
        session = self._current_yield_session
        if session is None or session.completed:
            return

        session.pending_tasks.discard(task)
        if not session.pending_tasks and not session.completed:
            session.completed = True
            self.cancel_timer(YIELD_TIMEOUT_TIMER_NAME)
            self._emit_event(YieldCompleted())

    def _handle_yield_timeout(self) -> None:
        """Handle 100 ms hard timeout when cancels hang."""
        session = self._current_yield_session
        if session is None or session.completed:
            return

        session.completed = True
        pending_names = tuple(t.get_name() for t in session.pending_tasks)
        if self._telemetry is not None:
            self._telemetry.emit(
                "supervisor.yield_timeout",
                self._clock.now_ns(),
                pending_count=len(session.pending_tasks),
                pending_tasks=pending_names,
            )

        self._emit_event(YieldCompleted())

    # -------------------------------------------------------------------------
    # Effect Executor (Sequential with Error Isolation)
    # -------------------------------------------------------------------------

    def execute_effects(self, effects: Sequence[Effect]) -> None:
        """Execute effects strictly in order with error isolation per effect.

        Failures are logged to telemetry and never raised into the frame path.
        """
        is_yield_sequence = any(isinstance(e, FlushPlayout) for e in effects) and (
            any(isinstance(e, CancelTts) for e in effects)
            or any(isinstance(e, AbortLlm) for e in effects)
        )

        if is_yield_sequence:
            self._start_yield_session()

        for effect in effects:
            try:
                self._execute_single_effect(effect)
            except Exception as exc:
                self._last_errors.append((effect, exc))
                self._log_error(type(effect).__name__, exc, effect=type(effect).__name__)

        if is_yield_sequence:
            self._finish_yield_session_setup()

    def _execute_single_effect(self, effect: Effect) -> None:
        """Dispatch a single effect to its port or handler."""
        if isinstance(effect, SetGain):
            if self._mixer is not None and hasattr(self._mixer, "set_gain"):
                self._mixer.set_gain(gain=effect.gain, ramp_ms=effect.ramp_ms)
            elif self._gain_handler is not None:
                self._gain_handler(effect.gain, effect.ramp_ms)

        elif isinstance(effect, FlushPlayout):
            if self._mixer is not None and hasattr(self._mixer, "flush"):
                self._mixer.flush()
            if self._transport is not None and hasattr(self._transport, "flush_output"):
                self._transport.flush_output()
            if self._flush_handler is not None:
                self._flush_handler()

        elif isinstance(effect, CancelTts):
            self._dispatch_cancel_tts()

        elif isinstance(effect, AbortLlm):
            self._dispatch_abort_llm()

        elif isinstance(effect, TruncateHistory):
            if self._history_truncator is not None:
                self._history_truncator(effect.played_pos_ms)

        elif isinstance(effect, ArmTimer):
            self.arm_timer(name=effect.name, deadline_ns=effect.deadline_ns)

        elif isinstance(effect, CancelTimer):
            self.cancel_timer(name=effect.name)

        elif isinstance(effect, PlayClip):
            if self._mixer is not None and hasattr(self._mixer, "play_clip"):
                self._mixer.play_clip(clip=effect.clip, tag=effect.tag)
            elif self._clip_handler is not None:
                self._clip_handler(effect.clip, effect.tag)

        elif isinstance(effect, FadeOutClip):
            if self._mixer is not None and hasattr(self._mixer, "fade_out_clip"):
                self._mixer.fade_out_clip(dur_ms=effect.dur_ms)
            elif self._fade_handler is not None:
                self._fade_handler(effect.dur_ms)

        elif isinstance(effect, StartPlayout):
            if self._mixer is not None and hasattr(self._mixer, "start_playout"):
                self._mixer.start_playout()
            elif self._playout_handler is not None:
                self._playout_handler()

        elif isinstance(effect, StartResponse):
            self.start_turn(turn_id=effect.turn_id, epoch=effect.epoch)

        elif isinstance(effect, CancelResponse):
            self.cancel_turn()

        elif isinstance(effect, OpenOverlapWindow):
            if self._overlap_window_handler is not None:
                self._overlap_window_handler(True)

        elif isinstance(effect, CloseOverlapWindow):
            if self._overlap_window_handler is not None:
                self._overlap_window_handler(False)

        elif isinstance(effect, MergeUserTurn):
            if self._merge_turn_handler is not None:
                self._merge_turn_handler(effect.text)

        elif isinstance(effect, TelemetryEffect):
            if self._telemetry is not None:
                self._telemetry.emit(
                    effect.name,
                    self._clock.now_ns(),
                    **dict(effect.payload),
                )

    def _dispatch_cancel_tts(self) -> None:
        """Cancel all in-flight TTS streams within one loop turn."""
        streams = list(self._tts_streams)
        for s in streams:
            try:
                res = getattr(s, "cancel", None)
                if res is not None and callable(res):
                    cancel_coro = res()
                    if asyncio.iscoroutine(cancel_coro):
                        # Clean-up tasks use independent CancelToken so they are
                        # not pre-cancelled by turn token
                        task = self.spawn_task(cancel_coro, name="cancel_tts", cancel=CancelToken())
                        session = self._current_yield_session
                        if session is not None and not session.completed:
                            session.pending_tasks.add(task)
                            task.add_done_callback(self._on_yield_task_done)
            except Exception as exc:
                self._log_error("cancel_tts", exc)
            finally:
                self.unregister_stream(s)

    def _dispatch_abort_llm(self) -> None:
        """Abort all in-flight LLM streams within one loop turn."""
        streams = list(self._llm_streams)
        for s in streams:
            try:
                res = getattr(s, "abort", None)
                if res is not None and callable(res):
                    abort_coro = res()
                    if asyncio.iscoroutine(abort_coro):
                        # Clean-up tasks use independent CancelToken so they are
                        # not pre-cancelled by turn token
                        task = self.spawn_task(abort_coro, name="abort_llm", cancel=CancelToken())
                        session = self._current_yield_session
                        if session is not None and not session.completed:
                            session.pending_tasks.add(task)
                            task.add_done_callback(self._on_yield_task_done)
            except Exception as exc:
                self._log_error("abort_llm", exc)
            finally:
                self.unregister_stream(s)

    # -------------------------------------------------------------------------
    # Stale-Epoch Helper
    # -------------------------------------------------------------------------

    def is_stale(self, event: Any) -> bool:
        """Check if an event carries an epoch older than current_epoch."""
        stale = is_stale_epoch(event, self._current_epoch)
        if stale and self._telemetry is not None:
            self._telemetry.emit(
                "supervisor.stale_epoch_dropped",
                self._clock.now_ns(),
                event=type(event).__name__,
                event_epoch=getattr(event, "epoch", None),
                current_epoch=self._current_epoch,
            )
        return stale

    def filter_event(self, event: Any) -> bool:
        """Return True if event is not stale, False if stale."""
        return not self.is_stale(event)

    # -------------------------------------------------------------------------
    # Quiescence Verification (INV-S1)
    # -------------------------------------------------------------------------

    def is_quiescent(self) -> bool:
        """Return True if 0 live tasks, streams, and timers."""
        live_tasks = any(not t.done() for t in self._active_tasks)
        live_streams = bool(self._active_streams)
        live_timers = any(not handle.cancelled for _, handle in self._timers.values())
        return not (live_tasks or live_streams or live_timers)

    def assert_quiescent(self) -> None:
        """Verify 0 live tasks/streams/timers per INV-S1.

        Raises AssertionError with diagnostic details if not quiescent.
        """
        live_tasks = [t for t in self._active_tasks if not t.done()]
        live_streams = list(self._active_streams)
        live_timers = [name for name, (_, handle) in self._timers.items() if not handle.cancelled]

        if live_tasks or live_streams or live_timers:
            details: list[str] = []
            if live_tasks:
                task_names = [t.get_name() for t in live_tasks]
                details.append(f"{len(live_tasks)} live tasks: {task_names}")
            if live_streams:
                details.append(f"{len(live_streams)} live streams: {live_streams}")
            if live_timers:
                details.append(f"{len(live_timers)} live timers: {live_timers}")

            raise AssertionError(f"StreamSupervisor is not quiescent: {'; '.join(details)}")

    async def shutdown(self) -> None:
        """Gracefully cancel and drain all active timers, streams, and tasks."""
        self.cancel_all_timers()
        self.cancel_turn()

        # Cancel any remaining active streams
        for stream in list(self._active_streams):
            self._cancel_stream_sync_or_async(stream)
            self.unregister_stream(stream)

        # Cancel any active tasks
        pending = [t for t in self._active_tasks if not t.done()]
        for t in pending:
            t.cancel()

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        self._active_tasks.clear()
        self._active_streams.clear()

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _emit_event(self, event: Event) -> None:
        """Deliver an event to the registered on_event callback and internal record."""
        self._emitted_events.append(event)
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception as exc:
                self._log_error("on_event", exc)

    def _log_error(self, operation: str, exc: Exception, **extra: Any) -> None:
        """Log an error to telemetry."""
        if self._telemetry is not None:
            self._telemetry.emit(
                "supervisor.effect_error",
                self._clock.now_ns(),
                operation=operation,
                error=str(exc),
                **extra,
            )
