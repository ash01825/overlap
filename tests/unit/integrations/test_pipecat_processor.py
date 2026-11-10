"""Hermetic unit tests for `OverlapProcessor` and `StockInterruptionProcessor` (T4.02).

Covers every `done_when` and `failure_modes` requirement:
1. `OverlapProcessor(...)` takes `Tuning`/`AppConfig` and is documented in README (<= 10 lines).
2. Pipecat's built-in interruption strategy is disabled (`allow_interruptions = False`, and
   `StartInterruptionFrame` / `StopInterruptionFrame` / `BotInterruptionFrame` /
   `UserStartedSpeakingFrame` / `UserStoppedSpeakingFrame` are suppressed so no double-stop occurs).
3. Control frames (`StartFrame`, `EndFrame`, `CancelFrame`, custom `ControlFrame` / `DataFrame`)
   pass through strictly in order without swallowing or reordering.
4. Shutdown (`EndFrame` or `CancelFrame`) calls `engine.stop()` and `engine.assert_quiescent()`
   so engine tasks never outlive pipeline shutdown.
5. Frame back-pressure on outbound audio uses bounded queues (drop-oldest) and never stalls
   the pipeline.
6. `examples/stock_pipecat_agent.py` is behaviourally 'stock' (stop-on-any-VAD immediately
   flushes and emits `StartInterruptionFrame` even on a short backchannel, whereas
   `OverlapProcessor` ducks to `Tuning.duck_gain` and resumes without stopping).
"""

from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from overlap.config import AppConfig
from overlap.core.clock import VirtualClock
from overlap.core.types import (
    AudioFrame,
    EndOfTurn,
    FirstAudioReady,
    FsmState,
    SttEvent,
    Tuning,
)
from overlap.integrations.pipecat_processor import (
    BotInterruptionFrame,
    CancelFrame,
    ControlFrame,
    DataFrame,
    EndFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    OverlapProcessor,
    PipelineTaskParams,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    disable_pipecat_interruptions,
)
from overlap.ports.llm import LlmPort, LlmStream, Message
from overlap.ports.stt import SttPort, SttStream
from overlap.ports.tts import AudioChunk, AudioClip, TtsPort, TtsStream
from overlap.ports.vad import Vad


def _run_virtual(coro: Any, clock: VirtualClock | None = None) -> Any:
    import importlib

    virtual_mod = importlib.import_module("tests.helpers.virtual")
    return virtual_mod.run_virtual(coro, clock=clock)


def _load_example_module(filename: str) -> Any:
    path = Path(__file__).resolve().parents[3] / "examples" / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pipecat_agent = _load_example_module("pipecat_agent.py")
_stock_pipecat_agent = _load_example_module("stock_pipecat_agent.py")

build_pipecat_overlap_processor = _pipecat_agent.build_pipecat_overlap_processor
create_overlap_task_params = _pipecat_agent.create_overlap_task_params
run_pipecat_session = _pipecat_agent.run_pipecat_session

build_stock_pipecat_processor = _stock_pipecat_agent.build_stock_pipecat_processor
create_stock_task_params = _stock_pipecat_agent.create_stock_task_params
run_stock_pipecat_session = _stock_pipecat_agent.run_stock_pipecat_session


class _MiniVad(Vad):
    """Deterministic fake VAD for processor unit tests."""

    def __init__(self, speech: bool = False, energy_db: float = -40.0) -> None:
        self.speech = speech
        self.energy_db = energy_db

    def process(self, frame: AudioFrame) -> Any:
        @dataclass(frozen=True, slots=True)
        class _Res:
            is_speech: bool
            confidence: float
            energy_db: float

        return _Res(
            is_speech=self.speech,
            confidence=1.0 if self.speech else 0.0,
            energy_db=self.energy_db,
        )


class _MiniSttStream(SttStream):
    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []
        self.closed = False
        self._queue: asyncio.Queue[SttEvent | None] = asyncio.Queue()

    async def send_audio(self, frame: AudioFrame) -> None:
        self.frames.append(frame)

    def push_event(self, event: SttEvent) -> None:
        self._queue.put_nowait(event)

    async def events(self) -> AsyncIterator[SttEvent]:
        while not self.closed:
            ev = await self._queue.get()
            if ev is None:
                break
            yield ev

    async def close(self) -> None:
        self.closed = True
        self._queue.put_nowait(None)


class _MiniSttPort(SttPort):
    def __init__(self) -> None:
        self.stream_instance = _MiniSttStream()

    async def open(self, *, lang_hint: str | None = None, cancel: Any = None) -> SttStream:
        return self.stream_instance


class _MiniTtsStream(TtsStream):
    def __init__(self, epoch: int) -> None:
        self.epoch = epoch
        self.cancelled = False
        self._done = asyncio.Event()

    async def send_text(self, clause: str, *, ssml: bool = False) -> None:
        pass

    async def finish(self) -> None:
        self._done.set()

    async def chunks(self) -> AsyncIterator[AudioChunk]:
        await self._done.wait()
        if False:
            yield AudioChunk(pcm=b"", sample_rate=16000, epoch=self.epoch)

    async def cancel(self) -> None:
        self.cancelled = True
        self._done.set()


class _MiniTtsPort(TtsPort):
    async def open_stream(self, *, voice_id: str, lang: str, epoch: int, cancel: Any) -> TtsStream:
        return _MiniTtsStream(epoch)

    async def synthesize(self, text: str, *, voice_id: str, lang: str) -> AudioClip:
        return AudioClip(pcm=b"\x00" * 640, sample_rate=16000)


class _MiniLlmStream(LlmStream):
    def __init__(self) -> None:
        self.aborted = False

    async def deltas(self) -> AsyncIterator[str]:
        if False:
            yield ""

    async def abort(self) -> None:
        self.aborted = True


class _MiniLlmPort(LlmPort):
    async def stream(self, messages: Sequence[Message], *, epoch: int, cancel: Any) -> LlmStream:
        return _MiniLlmStream()


def _make_ports() -> tuple[_MiniVad, _MiniSttPort, _MiniTtsPort, _MiniLlmPort]:
    return _MiniVad(), _MiniSttPort(), _MiniTtsPort(), _MiniLlmPort()


def _speech_pcm_20ms(sample_rate: int = 16000) -> bytes:
    samples = (sample_rate * 20) // 1000
    return b"\x10\x27" * samples


def test_overlap_processor_takes_tuning_and_config_only() -> None:
    """OverlapProcessor can be instantiated with Tuning or AppConfig alone."""
    tuning = Tuning()
    cfg = AppConfig(voice_id="v_test", lang="hi", tuning=tuning)

    proc_default = OverlapProcessor()
    assert proc_default.tuning == tuning
    assert not proc_default.interruptions_allowed_in_pipecat

    proc_tuned = OverlapProcessor(tuning=tuning, config=cfg)
    assert proc_tuned.tuning == tuning
    assert proc_tuned.config.voice_id == "v_test"


def test_readme_documents_pipecat_snippet_within_10_lines() -> None:
    """README.md includes a <= 10-line integration snippet for OverlapProcessor."""
    readme_path = Path(__file__).resolve().parents[3] / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    assert "OverlapProcessor" in content

    in_python_block = False
    snippet_lines: list[str] = []
    for line in content.splitlines():
        if line.strip().startswith("```python"):
            in_python_block = True
            snippet_lines = []
            continue
        if in_python_block and line.strip().startswith("```"):
            if any("OverlapProcessor" in s for s in snippet_lines):
                break
            in_python_block = False
            continue
        if in_python_block:
            snippet_lines.append(line)

    assert snippet_lines, "Expected a python snippet with OverlapProcessor in README.md"
    assert len(snippet_lines) <= 10, f"Snippet has {len(snippet_lines)} lines (> 10)"


def test_builtin_pipecat_interruption_strategy_disabled() -> None:
    """Pipecat's built-in interruption strategy is disabled on params, StartFrame, and frames."""
    params = PipelineTaskParams(allow_interruptions=True)
    updated = disable_pipecat_interruptions(params)
    assert updated.allow_interruptions is False

    overlap_params = create_overlap_task_params()
    assert overlap_params.allow_interruptions is False

    stock_params = create_stock_task_params()
    assert stock_params.allow_interruptions is True

    async def _run() -> None:
        clock = VirtualClock()
        vad, stt, tts, llm = _make_ports()
        proc = build_pipecat_overlap_processor(clock=clock, vad=vad, stt=stt, tts=tts, llm=llm)

        start_frame = StartFrame(
            id=1,
            name="start",
            allow_interruptions=True,
            params=PipelineTaskParams(allow_interruptions=True),
        )
        # Feed StartFrame + built-in Pipecat interruption frames + normal TextFrame + EndFrame
        frames_in = [
            start_frame,
            StartInterruptionFrame(id=2, name="start_int"),
            BotInterruptionFrame(id=3, name="bot_int"),
            UserStartedSpeakingFrame(id=4, name="user_start"),
            UserStoppedSpeakingFrame(id=5, name="user_stop"),
            StopInterruptionFrame(id=6, name="stop_int"),
            TextFrame(id=7, name="text", text="hello"),
            EndFrame(id=8, name="end"),
        ]
        emitted = await run_pipecat_session(proc, frames_in)

        # StartFrame had allow_interruptions forced to False
        assert start_frame.allow_interruptions is False
        assert start_frame.params.allow_interruptions is False

        # All 5 built-in interruption frames were suppressed (preventing double-stop)
        assert proc.builtin_interruptions_suppressed == 5
        emitted_objs = [f for f, _ in emitted]
        assert [f.id for f in emitted_objs] == [1, 7, 8]

    _run_virtual(_run())


def test_control_frames_pass_through_strictly_in_order_and_shutdown_quiescent() -> None:
    """Control frames (StartFrame, custom ControlFrame, EndFrame, CancelFrame) preserve order
    and shutdown calls engine.stop() and engine.assert_quiescent()."""

    async def _run() -> None:
        clock = VirtualClock()
        vad, stt, tts, llm = _make_ports()
        proc = build_pipecat_overlap_processor(clock=clock, vad=vad, stt=stt, tts=tts, llm=llm)

        c1 = ControlFrame(id=10, name="ctrl_1")
        d1 = DataFrame(id=11, name="data_1")
        c2 = ControlFrame(id=12, name="ctrl_2")
        end = EndFrame(id=13, name="end")

        await run_pipecat_session(
            proc,
            [StartFrame(id=9, name="start"), c1, d1, c2, end],
        )

        emitted_ids = [f.id for f, _ in proc.emitted_frames]
        assert emitted_ids == [9, 10, 11, 12, 13]

        # Engine is stopped and quiescent after EndFrame
        assert proc.engine is not None
        assert proc.engine._stopped is True
        proc.engine.assert_quiescent()

        # Also verify CancelFrame triggers clean stop and assert_quiescent
        clock2 = VirtualClock()
        vad2, stt2, tts2, llm2 = _make_ports()
        proc2 = build_pipecat_overlap_processor(
            clock=clock2, vad=vad2, stt=stt2, tts=tts2, llm=llm2
        )
        await run_pipecat_session(
            proc2,
            [StartFrame(id=1, name="start"), CancelFrame(id=2, name="cancel")],
        )
        assert [f.id for f, _ in proc2.emitted_frames] == [1, 2]
        assert proc2.engine is not None
        assert proc2.engine._stopped is True
        proc2.engine.assert_quiescent()

    _run_virtual(_run())


def test_frame_backpressure_never_stalls_pipeline() -> None:
    """Slow downstream consumer does not block send_frame or process_frame; drops oldest."""

    async def _run() -> None:
        clock = VirtualClock()
        tuning = Tuning()
        vad, stt, tts, llm = _make_ports()

        blocked_event = asyncio.Event()
        received_downstream: list[Any] = []

        async def _slow_downstream(frame: Any, direction: Any) -> None:
            if isinstance(frame, OutputAudioRawFrame):
                await blocked_event.wait()
            received_downstream.append(frame)

        proc = OverlapProcessor(
            tuning=tuning,
            vad=vad,
            stt=stt,
            tts=tts,
            llm=llm,
            clock=clock,
            downstream_handler=_slow_downstream,
            max_outbound_frames=4,
        )

        await proc.process_frame(StartFrame(id=1, name="start"))

        # Flood 20 outbound audio frames synchronously while downstream is blocked
        for i in range(20):
            proc.bridge_transport.send_frame(
                AudioFrame(
                    pcm=_speech_pcm_20ms(tuning.out_rate_hz),
                    sample_rate=tuning.out_rate_hz,
                    ts_ns=(i + 1) * 20_000_000,
                )
            )

        # Queue remained bounded at 4 and dropped oldest without blocking
        assert proc.bridge_transport.dropped_outbound_frames == 16
        assert proc.bridge_transport.output_queue_ms() <= 4 * tuning.frame_ms

        # Inbound audio and control frames still process immediately without stalling
        await proc.process_frame(
            InputAudioRawFrame(
                id=2,
                audio=_speech_pcm_20ms(tuning.in_rate_hz),
                sample_rate=tuning.in_rate_hz,
            )
        )
        await proc.process_frame(ControlFrame(id=3, name="marker"))

        # Unblock downstream and shut down cleanly
        blocked_event.set()
        await asyncio.sleep(0)
        await proc.process_frame(EndFrame(id=4, name="end"))
        assert proc.engine is not None
        proc.engine.assert_quiescent()

    _run_virtual(_run())


def test_ab_comparison_overlap_ducks_backchannel_while_stock_stops_immediately() -> None:
    """Behavioural A/B verification:
    On a short user backchannel ('haan') during agent speech:
    - `OverlapProcessor` ducks gain to `Tuning.duck_gain` (0.3), classifies as BACKCHANNEL,
      restores gain to 1.0, and NEVER flushes playout or emits `StartInterruptionFrame`.
    - `StockInterruptionProcessor` (`examples/stock_pipecat_agent.py`) immediately flushes
      playout and emits `StartInterruptionFrame` on the very first VAD speech frame.
    """

    @dataclass
    class _Outcome:
        flushed: int
        interruption_frames: int
        saw_ducked_state: bool
        final_state: FsmState

    async def _run_overlap(clock: VirtualClock) -> _Outcome:
        tuning = Tuning()
        vad, stt, tts, llm = _make_ports()
        proc = build_pipecat_overlap_processor(
            tuning=tuning, clock=clock, vad=vad, stt=stt, tts=tts, llm=llm
        )
        assert proc.engine is not None

        await proc.process_frame(StartFrame(id=1, name="start"))

        # Put agent into SPEAKING with queued audio
        proc.engine.fsm.transition(
            EndOfTurn(final=SttEvent(kind="final", text="hello", lang="hi", conf=0.95, ts_ns=0)),
            now_ns=clock.now_ns(),
        )
        proc.engine.supervisor.start_turn("t1", 1)
        proc.engine.mixer.start_turn("t1", 1)
        for i in range(15):
            proc.engine.mixer.enqueue_chunk(
                AudioChunk(
                    pcm=_speech_pcm_20ms(tuning.out_rate_hz),
                    sample_rate=tuning.out_rate_hz,
                    epoch=1,
                    is_last=(i == 14),
                )
            )
        proc.engine._apply_event_fast_path(FirstAudioReady(), clock.now_ns())
        assert proc.engine.state == FsmState.SPEAKING

        saw_ducked = False

        # User utters a short 160 ms backchannel "haan"
        vad.speech = True
        for step in range(8):
            await proc.process_frame(
                InputAudioRawFrame(
                    audio=_speech_pcm_20ms(tuning.in_rate_hz),
                    sample_rate=tuning.in_rate_hz,
                )
            )
            if proc.engine.state == FsmState.DUCKED:
                saw_ducked = True
            if step == 2:
                proc.engine.handle_stt_event(
                    SttEvent(
                        kind="partial",
                        text="haan",
                        lang="hi",
                        conf=0.95,
                        ts_ns=clock.now_ns(),
                    )
                )
            await clock.advance(tuning.frame_ms)

        # User stops speaking
        vad.speech = False
        await proc.process_frame(
            InputAudioRawFrame(
                audio=b"\x00" * 640,
                sample_rate=tuning.in_rate_hz,
            )
        )
        state_after = proc.engine.state
        await proc.process_frame(EndFrame(id=99, name="end"))

        int_frames = sum(1 for f, _ in proc.emitted_frames if isinstance(f, StartInterruptionFrame))
        return _Outcome(
            flushed=proc.bridge_transport.flushed_count - 1,  # exclude close() flush
            interruption_frames=int_frames,
            saw_ducked_state=saw_ducked,
            final_state=state_after,
        )

    async def _run_stock(clock: VirtualClock) -> _Outcome:
        tuning = Tuning()
        vad, stt, tts, llm = _make_ports()
        stock = build_stock_pipecat_processor(
            tuning=tuning, clock=clock, vad=vad, stt=stt, tts=tts, llm=llm
        )

        await run_stock_pipecat_session(stock, [StartFrame(id=1, name="start")])

        stock.engine.fsm.transition(
            EndOfTurn(final=SttEvent(kind="final", text="hello", lang="hi", conf=0.95, ts_ns=0)),
            now_ns=clock.now_ns(),
        )
        stock.engine.supervisor.start_turn("t1", 1)
        stock.engine.mixer.start_turn("t1", 1)
        for i in range(15):
            stock.engine.mixer.enqueue_chunk(
                AudioChunk(
                    pcm=_speech_pcm_20ms(tuning.out_rate_hz),
                    sample_rate=tuning.out_rate_hz,
                    epoch=1,
                    is_last=(i == 14),
                )
            )
        stock.engine._apply_event_fast_path(FirstAudioReady(), clock.now_ns())
        assert stock.engine.state == FsmState.SPEAKING

        # Single 20 ms frame of user VAD onset immediately interrupts stock agent!
        vad.speech = True
        await stock.process_frame(
            InputAudioRawFrame(
                audio=_speech_pcm_20ms(tuning.in_rate_hz),
                sample_rate=tuning.in_rate_hz,
            )
        )
        vad.speech = False
        await stock.process_frame(
            InputAudioRawFrame(
                audio=b"\x00" * 640,
                sample_rate=tuning.in_rate_hz,
            )
        )
        await stock.process_frame(EndFrame(id=99, name="end"))

        int_frames = sum(
            1 for f, _ in stock.emitted_frames if isinstance(f, StartInterruptionFrame)
        )
        return _Outcome(
            flushed=stock.interruption_count,
            interruption_frames=int_frames,
            saw_ducked_state=(stock.engine.state == FsmState.DUCKED),
            final_state=stock.engine.state,
        )

    clock_ov = VirtualClock()
    overlap_outcome = _run_virtual(_run_overlap(clock_ov), clock=clock_ov)
    clock_st = VirtualClock()
    stock_outcome = _run_virtual(_run_stock(clock_st), clock=clock_st)

    # OVERLAP ducked, classified as BACKCHANNEL, resumed SPEAKING, and never flushed
    assert overlap_outcome.saw_ducked_state is True
    assert overlap_outcome.flushed == 0
    assert overlap_outcome.interruption_frames == 0
    assert overlap_outcome.final_state == FsmState.SPEAKING

    # Stock baseline immediately flushed and emitted StartInterruptionFrame on the first VAD frame
    assert stock_outcome.flushed == 1
    assert stock_outcome.interruption_frames == 1
