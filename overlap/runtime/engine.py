"""OverlapEngine: Full-duplex conversational audio runtime (§3, §6, §7, §8, §9, §10, §14).

Glues AudioTransport, VAD, EchoGate, STT, FSM, OverlapClassifier, BackchannelPredictor,
PlayoutMixer, StreamSupervisor, and TurnPipeline using ports only.

Key architectural properties:
- Fast path (frame callback) is strictly synchronous: no await anywhere in handle_frame (INV-L13).
- Delivers <= 40 ms virtual ducking latency on speech onset (INV-L2).
- Zero post-cancel audio leakage to transport (INV-L7).
- Graceful degradation: classifier exception degrades to stock policy; never crashes the call (§14).
- Watchdog timer safety: STT failures mid-overlap fall back to acoustic classification
  and classify_cap.
- Quiescence hygiene: 0 live tasks, streams, or timers after shutdown (INV-S1).
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from typing import Any, cast

from overlap.core.audio import EchoGate, LoudnessTracker
from overlap.core.classifier import classify
from overlap.core.clock import Clock, MonotonicClock, VirtualClock
from overlap.core.fsm import FSM
from overlap.core.predictor import BackchannelPredictor
from overlap.core.types import (
    AgentAudioDrained,
    AudioFrame,
    ClipDone,
    EndOfTurn,
    FirstAudioReady,
    FsmState,
    Label,
    OverlapFeatures,
    SetGain,
    SttEvent,
    SttPartial,
    Tag,
    TimerFired,
    Tuning,
    VadOffset,
    VadOnset,
    Verdict,
    YieldCompleted,
)
from overlap.ports.llm import LlmPort, Message
from overlap.ports.stt import SttPort, SttStream
from overlap.ports.telemetry import TelemetrySink
from overlap.ports.transport import AudioTransport
from overlap.ports.tts import AudioChunk, AudioClip, TtsPort
from overlap.ports.vad import Vad
from overlap.runtime.clip_cache import ClipCache
from overlap.runtime.mixer import PlayoutMixer
from overlap.runtime.supervisor import StreamSupervisor
from overlap.runtime.turn import TurnPipeline


class OverlapEngine:
    """Production runtime orchestrator for full-duplex conversational voice agents."""

    def __init__(
        self,
        transport: AudioTransport,
        vad: Vad,
        stt: SttPort,
        tts: TtsPort,
        llm: LlmPort,
        *,
        clock: Clock | None = None,
        telemetry: TelemetrySink | None = None,
        tuning: Tuning | None = None,
        voice_id: str = "v1",
        lang: str = "hi",
        clip_cache: ClipCache | None = None,
        echo_gate: EchoGate | None = None,
        loudness_tracker: LoudnessTracker | None = None,
        fsm: FSM | None = None,
        classifier_fn: Callable[..., Verdict] | None = None,
        predictor: BackchannelPredictor | None = None,
        mixer: PlayoutMixer | None = None,
        supervisor: StreamSupervisor | None = None,
        turn_pipeline: TurnPipeline | None = None,
    ) -> None:
        self._transport: AudioTransport = transport
        self._vad: Vad = vad
        self._stt: SttPort = stt
        self._tts: TtsPort = tts
        self._llm: LlmPort = llm

        self._clock: Clock = clock if clock is not None else MonotonicClock()
        self._telemetry: TelemetrySink | None = telemetry
        self._tuning: Tuning = tuning if tuning is not None else Tuning()
        self._voice_id: str = voice_id
        self._lang: str = lang

        # Core pure DSP and state primitives
        self._echo_gate: EchoGate = echo_gate if echo_gate is not None else EchoGate()
        self._loudness_tracker: LoudnessTracker = (
            loudness_tracker if loudness_tracker is not None else LoudnessTracker()
        )
        self._fsm: FSM = fsm if fsm is not None else FSM(tuning=self._tuning, initial_epoch=1)
        self._classifier_fn: Callable[..., Verdict] = (
            classifier_fn if classifier_fn is not None else classify
        )
        self._predictor: BackchannelPredictor = (
            predictor
            if predictor is not None
            else BackchannelPredictor(tuning=self._tuning, default_lang=self._lang)
        )

        # Runtime orchestrators
        self._clip_cache: ClipCache = (
            clip_cache
            if clip_cache is not None
            else ClipCache(tts=self._tts, voice_id=self._voice_id, tuning=self._tuning)
        )
        self._mixer: PlayoutMixer = (
            mixer
            if mixer is not None
            else PlayoutMixer(
                transport=self._transport,
                clock=self._clock,
                tuning=self._tuning,
                echo_gate=self._echo_gate,
            )
        )
        self._supervisor: StreamSupervisor = (
            supervisor
            if supervisor is not None
            else StreamSupervisor(
                clock=self._clock,
                telemetry=self._telemetry,
                mixer=self._mixer,
                transport=self._transport,
                clip_handler=self._resolve_clip,
                fade_handler=self._handle_fade_out_clip,
                on_event=self._handle_supervisor_event,
            )
        )
        self._turn_pipeline: TurnPipeline = (
            turn_pipeline
            if turn_pipeline is not None
            else TurnPipeline(
                llm=self._llm,
                tts=self._tts,
                supervisor=self._supervisor,
                voice_id=self._voice_id,
                lang=self._lang,
                tuning=self._tuning,
                audio_chunk_handler=self._handle_tts_audio_chunk,
            )
        )

        # Wire turn pipeline to supervisor if newly created
        if turn_pipeline is None:
            self._turn_pipeline.attach_supervisor(self._supervisor)

        # Tracked runtime state
        self._user_speaking: bool = False
        self._last_user_frame_pcm: bytes = b""
        self._overlap_onset_ns: int = 0
        self._overlap_offset_ns: int = 0
        self._current_partial_text: str | None = None
        self._current_partial_ts_ns: int = 0
        self._current_partial_conf: float | None = None
        self._current_partial_lang: str | None = None
        self._soft_tick_evaluated: bool = False
        self._last_verdict: Verdict | None = None
        self._last_verdict_ts_ns: int = 0
        self._active_agent_turn_task: asyncio.Task[Any] | None = None

        # STT audio buffer & background tasks
        self._stt_stream: SttStream | None = None
        self._stt_frame_queue: deque[AudioFrame] = deque(maxlen=100)
        self._stt_frame_event: asyncio.Event = asyncio.Event()
        self._stt_send_task: asyncio.Task[None] | None = None
        self._stt_recv_task: asyncio.Task[None] | None = None
        self._transport_recv_task: asyncio.Task[None] | None = None
        self._drain_check_handle: Any = None
        self._stopped: bool = False

        # Diagnostic counters & metrics
        self._duck_latencies_ms: list[float] = []
        self._verdict_latencies_ms: list[float] = []
        self._stop_latencies_ms: list[float] = []
        self._clip_latencies_ms: list[float] = []

    # =========================================================================
    # Public Properties
    # =========================================================================

    @property
    def state(self) -> FsmState:
        """Current discrete FSM state."""
        return self._fsm.state

    @property
    def fsm(self) -> FSM:
        """Finite State Machine instance."""
        return self._fsm

    @property
    def supervisor(self) -> StreamSupervisor:
        """StreamSupervisor managing timers, tasks, and cancellation."""
        return self._supervisor

    @property
    def mixer(self) -> PlayoutMixer:
        """PlayoutMixer driving audio framing and transport emissions."""
        return self._mixer

    @property
    def turn_pipeline(self) -> TurnPipeline:
        """TurnPipeline managing LLM and TTS streaming."""
        return self._turn_pipeline

    @property
    def clock(self) -> Clock:
        """Injected or system clock."""
        return self._clock

    @property
    def transport(self) -> AudioTransport:
        """Configured AudioTransport port."""
        return self._transport

    @property
    def vad(self) -> Vad:
        """Voice Activity Detector port."""
        return self._vad

    @property
    def stt(self) -> SttPort:
        """STT port."""
        return self._stt

    @property
    def tts(self) -> TtsPort:
        """TTS port."""
        return self._tts

    @property
    def llm(self) -> LlmPort:
        """LLM port."""
        return self._llm

    @property
    def clip_cache(self) -> ClipCache:
        """ClipCache for pre-synthesized backchannels."""
        return self._clip_cache

    @property
    def predictor(self) -> BackchannelPredictor:
        """Agent backchannel predictor."""
        return self._predictor

    @property
    def echo_gate(self) -> EchoGate:
        """EchoGate adaptive echo suppression."""
        return self._echo_gate

    @property
    def loudness_tracker(self) -> LoudnessTracker:
        """LoudnessTracker maintaining agent RMS reference."""
        return self._loudness_tracker

    @property
    def tuning(self) -> Tuning:
        """Active tuning constants."""
        return self._tuning

    @property
    def history(self) -> list[Message]:
        """Committed conversational message history."""
        return self._turn_pipeline.history

    @property
    def last_verdict(self) -> Verdict | None:
        """Most recent overlap classification verdict."""
        return self._last_verdict

    # =========================================================================
    # Fast Path (Synchronous Frame Ingestion - INV-L13, INV-L2)
    # =========================================================================

    def handle_frame(self, frame: AudioFrame) -> None:
        """Synchronously process an incoming 20 ms mic audio frame.

        Architecture invariant INV-L13: Fast path never awaits network or I/O.
        Delivers <= 40 ms gain-ducked frame leaving mixer (INV-L2).
        """
        now_ns = frame.ts_ns if frame.ts_ns > 0 else self._clock.now_ns()

        # 1. Update EchoGate and LoudnessTracker reference RMS from agent playout
        ref_rms = self._loudness_tracker._agent_rms

        # 2. Check for self-audio / echo suppression
        is_echo = False
        if bool(frame.tags & (Tag.SELF_BACKCHANNEL | Tag.SELF_TTS)):
            is_echo = True
        elif self._echo_gate is not None:
            is_echo = self._echo_gate.is_echo(frame, ref_rms=ref_rms, now_ns=now_ns)

        # 3. Process frame with VAD
        vad_res = self._vad.process(frame)
        is_speech = bool(vad_res.is_speech and not is_echo)

        # 4. Update LoudnessTracker on genuine user speech
        if is_speech:
            self._last_user_frame_pcm = frame.pcm
            self._loudness_tracker.rel_loudness_db(frame.pcm)

        # 5. Non-blocking enqueue to STT stream (drop oldest on full)
        if self._stt_frame_queue is not None:
            if len(self._stt_frame_queue) >= 100:
                self._stt_frame_queue.popleft()
            self._stt_frame_queue.append(frame)
            self._stt_frame_event.set()

        # 6. Fast path FSM speech onset / offset detection
        prev_user_speaking = self._user_speaking
        self._user_speaking = is_speech

        if is_speech and not prev_user_speaking:
            # User speech onset detected!
            self._overlap_onset_ns = now_ns
            self._soft_tick_evaluated = False
            onset_event = VadOnset(ts_ns=now_ns, tag=frame.tags, is_echo=is_echo)
            self._apply_event_fast_path(onset_event, now_ns)

        elif not is_speech and prev_user_speaking:
            # User speech offset detected!
            self._overlap_offset_ns = now_ns
            offset_event = VadOffset(ts_ns=now_ns)
            self._apply_event_fast_path(offset_event, now_ns)

            # Re-evaluate classifier on speech end if DUCKED
            if self._fsm.state == FsmState.DUCKED:
                self._reclassify_overlap(now_ns)

        elif is_speech and prev_user_speaking:
            # Continuing overlap in DUCKED: check soft tick at classify_min_ms
            if self._fsm.state == FsmState.DUCKED and not self._soft_tick_evaluated:
                min_ns = self._tuning.classify_min_ms * 1_000_000
                if (now_ns - self._overlap_onset_ns) >= min_ns:
                    self._soft_tick_evaluated = True
                    self._reclassify_overlap(now_ns)

        # 7. Agent backchannel prediction (active when agent is LISTENING)
        if self._predictor is not None and self._fsm.state == FsmState.LISTENING:
            bc_due = self._predictor.update(
                now_ns=now_ns,
                vad_state=is_speech,
                energy_ema=vad_res.energy_db,
                fsm_state=self._fsm.state,
            )
            if bc_due is not None:
                self._apply_event_fast_path(bc_due, now_ns)

    def _apply_event_fast_path(self, event: Any, now_ns: int) -> None:
        """Apply an event to FSM and immediately dispatch SetGain inline to PlayoutMixer."""
        effects = self._fsm.transition(event, now_ns=now_ns)

        # Inline gain adjustment: delivers 20-40 ms ducking on speech onset (INV-L2)
        for eff in effects:
            if isinstance(eff, SetGain):
                self._mixer.set_gain(eff.gain, eff.ramp_ms)

        # realizations of other effects are queued to supervisor
        if effects:
            self._supervisor.execute_effects(effects)

        # State-specific reactions
        self._on_fsm_state_changed(now_ns)

    def _on_fsm_state_changed(self, now_ns: int) -> None:
        """Respond to FSM state updates."""
        current_state = self._fsm.state
        if self._predictor is not None:
            self._predictor.set_fsm_state(current_state)

        if current_state == FsmState.LISTENING:
            self._current_partial_text = None
            self._current_partial_conf = None
            self._current_partial_lang = None

    # =========================================================================
    # STT Event Ingestion & Overlap Classification
    # =========================================================================

    def handle_stt_event(self, event: SttEvent) -> None:
        """Process incoming partial or final transcription event."""
        now_ns = self._clock.now_ns()

        if event.kind == "partial":
            self._current_partial_text = event.text
            self._current_partial_ts_ns = now_ns
            self._current_partial_conf = event.conf
            self._current_partial_lang = event.lang

            # Update predictor with latest text
            if self._predictor is not None:
                self._predictor.update(
                    now_ns=now_ns,
                    vad_state=self._user_speaking,
                    partial=event,
                    fsm_state=self._fsm.state,
                )

            # If DUCKED: re-classify overlap with new partial
            if self._fsm.state == FsmState.DUCKED:
                self._reclassify_overlap(now_ns, is_final=False)

            # If SPEAKING: check for late escalation
            elif self._fsm.state == FsmState.SPEAKING:
                partial_ev = SttPartial(
                    text=event.text,
                    lang=event.lang,
                    conf=event.conf,
                    ts_ns=now_ns,
                )
                effects = self._fsm.transition(partial_ev, now_ns=now_ns)
                for eff in effects:
                    if isinstance(eff, SetGain):
                        self._mixer.set_gain(eff.gain, eff.ramp_ms)
                if effects:
                    self._supervisor.execute_effects(effects)
                if cast(FsmState, self._fsm.state) == FsmState.DUCKED:
                    self._reclassify_overlap(now_ns, is_final=False)

        elif event.kind == "final":
            self._current_partial_text = event.text
            if self._fsm.state == FsmState.DUCKED:
                self._reclassify_overlap(now_ns, is_final=True)
            elif self._fsm.state in (FsmState.LISTENING, FsmState.BACKCHANNELING):
                # Conversational turn concluded: trigger EndOfTurn
                self._apply_event_fast_path(EndOfTurn(final=event), now_ns)

    def _reclassify_overlap(self, now_ns: int, is_final: bool = False) -> None:
        """Evaluate overlap classifier on current acoustic and lexical features."""
        if self._fsm.state != FsmState.DUCKED:
            return

        overlap_ms = max(0, int((now_ns - self._overlap_onset_ns) // 1_000_000))
        partial_age_ms = (
            max(0, int((now_ns - self._current_partial_ts_ns) // 1_000_000))
            if self._current_partial_ts_ns > 0
            else None
        )

        rel_loudness = (
            self._loudness_tracker.rel_loudness_db(self._last_user_frame_pcm)
            if self._last_user_frame_pcm
            else 0.0
        )

        features = OverlapFeatures(
            overlap_ms=overlap_ms,
            user_speaking=self._user_speaking,
            rel_loudness_db=rel_loudness,
            partial_text=self._current_partial_text,
            partial_age_ms=partial_age_ms,
            stt_conf=self._current_partial_conf,
            lang=self._current_partial_lang,
            is_final=is_final,
        )

        try:
            verdict = self._classifier_fn(features, self._tuning, decided_at_ns=now_ns)
        except Exception as exc:
            # Failure mode 5: classifier exception crashes call instead of stock-policy degrade
            if self._telemetry is not None:
                self._telemetry.emit("classifier.exception", now_ns, error=str(exc))

            stock_stop = getattr(self._tuning, "stock_stop_ms", 250)
            if features.user_speaking and overlap_ms >= stock_stop:
                verdict = Verdict(
                    label=Label.INTERRUPT,
                    confidence=1.0,
                    reason="stock_policy_degrade",
                    overlap_ms=overlap_ms,
                    decided_at_ns=now_ns,
                )
            else:
                verdict = Verdict(
                    label=Label.PENDING,
                    confidence=0.0,
                    reason="stock_policy_pending",
                    overlap_ms=overlap_ms,
                    decided_at_ns=now_ns,
                )

        if verdict.label != Label.PENDING:
            self._last_verdict = verdict
            self._last_verdict_ts_ns = now_ns
            if self._overlap_onset_ns > 0:
                v_latency = (now_ns - self._overlap_onset_ns) / 1_000_000.0
                self._verdict_latencies_ms.append(v_latency)

            effects = self._fsm.transition(verdict, now_ns=now_ns)
            for eff in effects:
                if isinstance(eff, SetGain):
                    self._mixer.set_gain(eff.gain, eff.ramp_ms)
            if effects:
                self._supervisor.execute_effects(effects)
            self._on_fsm_state_changed(now_ns)

    # =========================================================================
    # Supervisor & Turn Callbacks
    # =========================================================================

    def _handle_supervisor_event(self, event: Any) -> None:
        """Handle events emitted by StreamSupervisor."""
        now_ns = self._clock.now_ns()

        if isinstance(event, TimerFired):
            if event.name == "classify_cap" and self._fsm.state == FsmState.DUCKED:
                # Watchdog cap fired: FSM forces fail-safe policy
                effects = self._fsm.transition(event, now_ns=now_ns)
                for eff in effects:
                    if isinstance(eff, SetGain):
                        self._mixer.set_gain(eff.gain, eff.ramp_ms)
                if effects:
                    self._supervisor.execute_effects(effects)
                self._on_fsm_state_changed(now_ns)

            elif event.name == "thinking_barge_in" and self._fsm.state == FsmState.THINKING:
                # Sustained user barge-in confirmed in THINKING
                effects = self._fsm.transition(event, now_ns=now_ns)
                if effects:
                    self._supervisor.execute_effects(effects)
                self._on_fsm_state_changed(now_ns)

            elif event.name == "bc_clip_end" and self._fsm.state == FsmState.BACKCHANNELING:
                if self._predictor is not None:
                    self._predictor.notify_clip_done()
                effects = self._fsm.transition(ClipDone(), now_ns=now_ns)
                if effects:
                    self._supervisor.execute_effects(effects)
                self._on_fsm_state_changed(now_ns)

        elif isinstance(event, YieldCompleted):
            # Interruption yield sequence completed: return to LISTENING
            if self._last_verdict_ts_ns > 0:
                stop_lat = (now_ns - self._last_verdict_ts_ns) / 1_000_000.0
                self._stop_latencies_ms.append(stop_lat)

            effects = self._fsm.transition(event, now_ns=now_ns)
            if effects:
                self._supervisor.execute_effects(effects)
            self._on_fsm_state_changed(now_ns)

    def _handle_tts_audio_chunk(self, chunk: AudioChunk) -> None:
        """Callback from TurnPipeline as synthesized audio chunks are received."""
        # Drop stale-epoch audio chunks immediately (INV-L7)
        if self._supervisor.is_stale(chunk):
            return

        # If FSM is THINKING, the arrival of the first audio chunk triggers FirstAudioReady
        if self._fsm.state == FsmState.THINKING:
            now_ns = self._clock.now_ns()
            ready_ev = FirstAudioReady()
            effects = self._fsm.transition(ready_ev, now_ns=now_ns)
            if effects:
                self._supervisor.execute_effects(effects)
            self._on_fsm_state_changed(now_ns)

        # Enqueue chunk into playout mixer
        self._mixer.enqueue_chunk(chunk)

    def _resolve_clip(
        self, clip_name: str | AudioClip | bytes, tag: Tag
    ) -> AudioClip | bytes | None:
        """Resolve a backchannel clip name to an AudioClip or bytes via ClipCache."""
        if isinstance(clip_name, (AudioClip, bytes)):
            return clip_name

        text = str(clip_name)
        clip = self._clip_cache.get(self._lang, text=text)
        if clip is None:
            clip = self._clip_cache.get(self._lang)

        if clip is not None and self._predictor is not None:
            self._predictor.notify_clip_started()

        return clip

    def _handle_fade_out_clip(self, dur_ms: int) -> None:
        """Notify predictor when backchannel clip is faded out."""
        if self._predictor is not None:
            self._predictor.notify_clip_done()

    # =========================================================================
    # Lifecycle: Start, Stop, and Drain
    # =========================================================================

    async def start(self) -> None:
        """Start background workers for STT, transport, and engine orchestration."""
        self._stopped = False
        await self._clip_cache.prepare([self._lang])

        # Open active STT stream session
        cancel_token = self._supervisor.create_child_token("engine_stt")
        self._stt_stream = await self._stt.open(lang_hint=self._lang, cancel=cancel_token)
        self._supervisor.register_stt_stream(self._stt_stream, cancel=cancel_token)

        # Spawn background workers
        self._stt_send_task = self._supervisor.spawn_task(
            self._stt_send_worker(), name="engine_stt_send"
        )
        self._stt_recv_task = self._supervisor.spawn_task(
            self._stt_recv_worker(), name="engine_stt_recv"
        )
        self._transport_recv_task = self._supervisor.spawn_task(
            self._transport_recv_worker(), name="engine_transport_recv"
        )

    async def stop(self) -> None:
        """Gracefully stop engine and drain active tasks and streams."""
        self._stopped = True
        self._stt_frame_event.set()

        if self._drain_check_handle is not None:
            self._clock.cancel(self._drain_check_handle)
            self._drain_check_handle = None

        if self._stt_send_task is not None:
            self._stt_send_task.cancel()
        if self._stt_recv_task is not None:
            self._stt_recv_task.cancel()
        if self._transport_recv_task is not None:
            self._transport_recv_task.cancel()

        if self._stt_stream is not None:
            try:
                await self._stt_stream.close()
            except Exception:
                pass
            self._supervisor.unregister_stream(self._stt_stream)
            self._stt_stream = None

        self._mixer.close()
        await self._supervisor.shutdown()

    def assert_quiescent(self) -> None:
        """Verify 0 live tasks, streams, and timers per INV-S1."""
        self._supervisor.assert_quiescent()

    # =========================================================================
    # Background Workers
    # =========================================================================

    async def _stt_send_worker(self) -> None:
        """Drain audio frame queue to STT stream without blocking fast path."""
        try:
            while not self._stopped:
                while self._stt_frame_queue:
                    frame = self._stt_frame_queue.popleft()
                    if self._stt_stream is not None:
                        try:
                            await self._stt_stream.send_audio(frame)
                        except Exception:
                            pass
                self._stt_frame_event.clear()
                try:
                    await self._stt_frame_event.wait()
                except asyncio.CancelledError:
                    break
        except asyncio.CancelledError:
            pass

    async def _stt_recv_worker(self) -> None:
        """Consume transcription events from active STT stream."""
        try:
            if self._stt_stream is not None:
                async for event in self._stt_stream.events():
                    if self._stopped:
                        break
                    self.handle_stt_event(event)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if self._telemetry is not None:
                self._telemetry.emit("engine.stt_error", self._clock.now_ns(), error=str(exc))

    async def _transport_recv_worker(self) -> None:
        """Consume incoming user frames from AudioTransport."""
        try:
            async for frame in self._transport.user_frames():
                if self._stopped:
                    break
                self.handle_frame(frame)
        except asyncio.CancelledError:
            pass

    # =========================================================================
    # Scenario DSL Execution (Hermetic Virtual Time)
    # =========================================================================

    def execute(self, compiled: Any) -> Any:
        """Synchronously execute a CompiledScenario within a hermetic virtual event loop."""
        import importlib

        scenario_mod = importlib.import_module("tests.fixtures.scenario")
        trace_cls: Any = scenario_mod.Trace
        virtual_mod = importlib.import_module("tests.helpers.virtual")
        run_virtual: Any = virtual_mod.run_virtual

        async def _run_scenario_async() -> Any:
            clock: VirtualClock = compiled.fakes["clock"]
            transport = compiled.fakes["transport"]
            vad = compiled.fakes["vad"]

            recorded_states: list[tuple[int, FsmState]] = [
                (clock.now_ns() // 1_000_000, self._fsm.state)
            ]
            recorded_effects: list[tuple[int, Any]] = []

            # Intercept FSM transitions to record states and effects
            prev_transition = self._fsm.transition

            def _intercept_transition(event: Any, now_ns: int = 0) -> tuple[Any, ...]:
                effs = prev_transition(event, now_ns=now_ns)
                t_ms = clock.now_ns() // 1_000_000
                recorded_states.append((t_ms, self._fsm.state))
                for e in effs:
                    recorded_effects.append((t_ms, e))
                return effs

            self._fsm.transition = _intercept_transition  # type: ignore[method-assign]

            # Determine maximum scenario duration
            max_t_ms = 0
            for ev in compiled.timeline:
                end_t = ev.t_ms
                if "dur_ms" in ev.data and ev.data["dur_ms"]:
                    end_t += int(ev.data["dur_ms"])
                max_t_ms = max(max_t_ms, end_t)

            max_t_ms += 1000  # generous tail window for drain and resolution

            # Timeline event index pointers
            timeline_events = list(compiled.timeline)
            active_overlaps: list[dict[str, Any]] = []

            # Step forward in uniform 20 ms virtual frames
            cur_ms = 0
            frame_ms = self._tuning.frame_ms

            while cur_ms <= max_t_ms:
                # 1. Trigger scheduled timeline events at cur_ms
                while timeline_events and timeline_events[0].t_ms <= cur_ms:
                    ev = timeline_events.pop(0)
                    kind = ev.kind
                    data = ev.data

                    if kind == "agent_says":
                        dur_ms = data.get("dur_ms") or 1000
                        turn_id = f"turn_{self._supervisor.current_epoch + 1}"
                        epoch = self._supervisor.current_epoch + 1
                        self._supervisor.start_turn(turn_id, epoch)
                        self._mixer.start_turn(turn_id, epoch)

                        # Enqueue deterministic tone chunks into mixer
                        num_chunks = max(1, dur_ms // frame_ms)
                        bytes_per_chunk = int(16000 * 2 * (frame_ms / 1000.0))
                        for i in range(num_chunks):
                            chunk_pcm = bytes([(i * 7 + j) % 256 for j in range(bytes_per_chunk)])
                            is_last = i == (num_chunks - 1)
                            self._mixer.enqueue_chunk(
                                AudioChunk(
                                    pcm=chunk_pcm,
                                    sample_rate=16000,
                                    epoch=epoch,
                                    is_last=is_last,
                                )
                            )

                        # Transition FSM to SPEAKING
                        self._apply_event_fast_path(FirstAudioReady(), clock.now_ns())

                    elif kind == "user_overlap":
                        active_overlaps.append(
                            {
                                "start_ms": ev.t_ms,
                                "end_ms": ev.t_ms + int(data.get("dur_ms", 300)),
                                "text": data.get("text", ""),
                                "stt_sent": False,
                            }
                        )

                    elif kind == "user_turn":
                        active_overlaps.append(
                            {
                                "start_ms": ev.t_ms,
                                "end_ms": ev.t_ms + int(data.get("dur_ms", 1000)),
                                "text": data.get("text", ""),
                                "is_final_turn": True,
                                "stt_sent": False,
                            }
                        )

                # 2. Generate and process audio frames for active user speech
                is_user_speaking = False
                for ov in active_overlaps:
                    if ov["start_ms"] <= cur_ms < ov["end_ms"]:
                        is_user_speaking = True
                        # Send STT partial transcript at onset + 40 ms
                        if not ov["stt_sent"] and cur_ms >= ov["start_ms"] + 40:
                            ov["stt_sent"] = True
                            if ov.get("is_final_turn", False):
                                stt_ev = SttEvent(
                                    kind="final",
                                    text=ov["text"],
                                    conf=0.95,
                                    ts_ns=clock.now_ns(),
                                )
                            else:
                                stt_ev = SttEvent(
                                    kind="partial",
                                    text=ov["text"],
                                    conf=0.9,
                                    ts_ns=clock.now_ns(),
                                )
                            self.handle_stt_event(stt_ev)

                # Remove completed overlaps
                for ov in list(active_overlaps):
                    if cur_ms >= ov["end_ms"]:
                        if ov.get("is_final_turn", False) and not ov["stt_sent"]:
                            ov["stt_sent"] = True
                            self.handle_stt_event(
                                SttEvent(kind="final", text=ov["text"], ts_ns=clock.now_ns())
                            )
                        active_overlaps.remove(ov)

                # Synthesize 20 ms mic audio frame
                if is_user_speaking:
                    # Speech frame: 16-bit PCM amplitude ~15000
                    bytes_len = int(16000 * 2 * (frame_ms / 1000.0))
                    user_pcm = bytes([(j * 11) % 256 for j in range(bytes_len)])
                    vad.set_speech(True, confidence=1.0)
                else:
                    user_pcm = b"\x00" * 640
                    vad.set_speech(False, confidence=0.0)

                frame = AudioFrame(
                    pcm=user_pcm,
                    sample_rate=16000,
                    ts_ns=clock.now_ns(),
                    tags=Tag.NONE,
                )

                # Process synchronous fast path frame
                self.handle_frame(frame)

                # 3. Advance virtual clock by 20 ms and drain event loop
                await clock.advance(frame_ms)
                cur_ms += frame_ms

                # Check if agent audio has drained
                if (
                    self._fsm.state in (FsmState.SPEAKING, FsmState.DUCKED)
                    and not self._mixer.is_playing
                    and not timeline_events
                    and not active_overlaps
                ):
                    self._apply_event_fast_path(AgentAudioDrained(), clock.now_ns())

            return trace_cls(
                states=recorded_states,
                effects=recorded_effects,
                frames_out=[r.frame for r in transport.sent_frames],
                telemetry=[],
                flush_timestamps_ms=list(transport.flush_history),
            )

        clock_fake = compiled.fakes.get("clock")
        return run_virtual(_run_scenario_async(), clock=clock_fake)
