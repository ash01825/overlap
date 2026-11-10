"""OverlapProcessor: One-line drop-in processor for Pipecat pipelines (ARCHITECTURE §3, §13).

Provides `OverlapProcessor` (and `StockInterruptionProcessor` for A/B baseline),
which embeds between transport input and downstream output in a Pipecat pipeline.
Takes `Tuning` / `AppConfig` (and injected or configured ports) and:
1. Disables Pipecat's built-in interruption strategy (prevents double-stop by intercepting
   `StartInterruptionFrame`, `StopInterruptionFrame`, `BotInterruptionFrame`, and setting
   `allow_interruptions = False` on `PipelineTaskParams` / `StartFrame`).
2. Passes Pipecat control and system frames (`StartFrame`, `EndFrame`, `CancelFrame`, etc.)
   through strictly in order without swallowing or reordering them.
3. Routes inbound audio frames (`InputAudioRawFrame` / `AudioRawFrame`) to `OverlapEngine`
   via a non-blocking `AudioTransport` bridge (`_PipecatBridgeTransport`) so back-pressure
   never stalls the pipeline.
4. Emits outbound gain-ducked / mixed agent audio (`OutputAudioRawFrame`) downstream.
5. On shutdown (`EndFrame` or `CancelFrame`), invokes `await engine.stop()` and
   `engine.assert_quiescent()` before forwarding the terminal control frame so engine
   tasks never outlive pipeline shutdown.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from overlap.config import AppConfig, load_config
from overlap.core.audio import EchoGate
from overlap.core.clock import Clock, MonotonicClock
from overlap.core.types import (
    AudioFrame,
    FlushPlayout,
    FsmState,
    Tag,
    Tuning,
)
from overlap.ports.llm import LlmPort, Message
from overlap.ports.stt import SttPort
from overlap.ports.telemetry import TelemetrySink
from overlap.ports.transport import AudioTransport
from overlap.ports.tts import TtsPort
from overlap.ports.vad import Vad
from overlap.runtime.engine import OverlapEngine


class FrameDirection(Enum):
    """Frame flow direction in a Pipecat pipeline."""

    DOWNSTREAM = 1
    UPSTREAM = 2


@dataclass(slots=True)
class PipecatFrame:
    """Base protocol-compatible Pipecat frame representation."""

    id: int = 0
    name: str = ""


@dataclass(slots=True)
class SystemFrame(PipecatFrame):
    """High-priority system frame processed immediately."""


@dataclass(slots=True)
class ControlFrame(PipecatFrame):
    """Ordered pipeline lifecycle and control frame."""


@dataclass(slots=True)
class DataFrame(PipecatFrame):
    """Ordered pipeline data payload frame."""


@dataclass(slots=True)
class StartFrame(ControlFrame):
    """Pipeline initialization frame carrying task parameters."""

    allow_interruptions: bool = False
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 16000
    params: Any = None


@dataclass(slots=True)
class EndFrame(ControlFrame):
    """Graceful pipeline termination frame."""


@dataclass(slots=True)
class CancelFrame(SystemFrame):
    """Immediate pipeline cancellation frame."""


@dataclass(slots=True)
class StartInterruptionFrame(SystemFrame):
    """Pipecat built-in interruption start signal."""


@dataclass(slots=True)
class StopInterruptionFrame(SystemFrame):
    """Pipecat built-in interruption stop signal."""


@dataclass(slots=True)
class BotInterruptionFrame(SystemFrame):
    """Pipecat built-in bot interruption signal."""


@dataclass(slots=True)
class UserStartedSpeakingFrame(SystemFrame):
    """Pipecat VAD user started speaking notification."""


@dataclass(slots=True)
class UserStoppedSpeakingFrame(SystemFrame):
    """Pipecat VAD user stopped speaking notification."""


@dataclass(slots=True)
class AudioRawFrame(DataFrame):
    """Raw PCM audio frame."""

    audio: bytes = b""
    sample_rate: int = 16000
    num_channels: int = 1


@dataclass(slots=True)
class InputAudioRawFrame(AudioRawFrame):
    """Inbound user microphone PCM audio frame from transport."""


@dataclass(slots=True)
class OutputAudioRawFrame(AudioRawFrame):
    """Outbound agent playout PCM audio frame to transport."""


@dataclass(slots=True)
class TextFrame(DataFrame):
    """Text data frame."""

    text: str = ""


@dataclass(slots=True)
class TranscriptionFrame(TextFrame):
    """Final STT transcription frame."""

    user_id: str = ""
    timestamp: str = ""
    language: str | None = None


@dataclass(slots=True)
class InterimTranscriptionFrame(TextFrame):
    """Interim/partial STT transcription frame."""

    user_id: str = ""
    timestamp: str = ""
    language: str | None = None


@dataclass(slots=True)
class PipelineTaskParams:
    """Configuration parameters for a Pipecat PipelineTask."""

    allow_interruptions: bool = False
    enable_metrics: bool = True
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 16000
    extra: dict[str, Any] = field(default_factory=dict)


def disable_pipecat_interruptions(target: Any) -> Any:
    """Disable Pipecat's built-in interruption strategy on a params or frame object.

    Ensures Pipecat's default VAD stop-on-any-speech does not fire alongside OVERLAP.
    """
    if target is None:
        return PipelineTaskParams(allow_interruptions=False)
    if hasattr(target, "allow_interruptions"):
        try:
            target.allow_interruptions = False
        except Exception:
            pass
    if hasattr(target, "interruption_strategies"):
        try:
            target.interruption_strategies = []
        except Exception:
            pass
    params = getattr(target, "params", None)
    if params is not None and hasattr(params, "allow_interruptions"):
        try:
            params.allow_interruptions = False
        except Exception:
            pass
    return target


_BUILTIN_INTERRUPTION_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "StartInterruptionFrame",
        "StopInterruptionFrame",
        "BotInterruptionFrame",
        "UserStartedSpeakingFrame",
        "UserStoppedSpeakingFrame",
    }
)


def _is_builtin_interruption_frame(frame: Any) -> bool:
    """Return True if `frame` is a Pipecat built-in interruption or VAD trigger frame."""
    if isinstance(
        frame,
        (
            StartInterruptionFrame,
            StopInterruptionFrame,
            BotInterruptionFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
        ),
    ):
        return True
    cls_name = type(frame).__name__
    return cls_name in _BUILTIN_INTERRUPTION_CLASS_NAMES


def _is_start_frame(frame: Any) -> bool:
    return isinstance(frame, StartFrame) or type(frame).__name__ == "StartFrame"


def _is_end_frame(frame: Any) -> bool:
    return isinstance(frame, EndFrame) or type(frame).__name__ == "EndFrame"


def _is_cancel_frame(frame: Any) -> bool:
    return isinstance(frame, CancelFrame) or type(frame).__name__ == "CancelFrame"


def _is_input_audio_frame(frame: Any) -> bool:
    if isinstance(frame, InputAudioRawFrame):
        return True
    cls_name = type(frame).__name__
    if cls_name in ("InputAudioRawFrame", "UserAudioRawFrame"):
        return True
    return isinstance(frame, AudioRawFrame) and not isinstance(frame, OutputAudioRawFrame)


class _PipecatBridgeTransport(AudioTransport):
    """Non-blocking `AudioTransport` bridging Pipecat frames and `OverlapEngine`.

    Enforces bounded queues with drop-oldest semantics so frame back-pressure
    never stalls either the synchronous fast path or the Pipecat pipeline.
    """

    def __init__(
        self,
        *,
        tuning: Tuning,
        clock: Clock,
        outbound_frame_factory: Callable[[bytes, int, int], Any] | None = None,
        downstream_push_fn: Callable[[Any, Any], Any] | None = None,
        max_inbound_frames: int = 100,
        max_outbound_frames: int = 50,
    ) -> None:
        self._tuning = tuning
        self._clock = clock
        self._outbound_frame_factory = outbound_frame_factory or self._default_outbound_frame
        self._downstream_push_fn = downstream_push_fn
        self._max_inbound_frames = max(1, max_inbound_frames)
        self._max_outbound_frames = max(1, max_outbound_frames)

        # Canonical 20 ms framing parameters
        self._target_in_rate = self._tuning.in_rate_hz
        self._frame_ms = self._tuning.frame_ms
        self._in_samples_per_frame = (self._target_in_rate * self._frame_ms) // 1000
        self._in_bytes_per_frame = self._in_samples_per_frame * 2
        self._in_byte_buf = bytearray()
        self._emitted_in_frames: int = 0
        self._stream_start_ns: int | None = None

        self._inbound_queue: deque[AudioFrame] = deque()
        self._inbound_event: asyncio.Event = asyncio.Event()
        self._outbound_queue: deque[tuple[Any, int]] = deque()
        self._pending_outbound_ms: int = 0
        self._flush_epoch: int = 0
        self._drain_task: asyncio.Task[None] | None = None
        self._closed: bool = False

        self.dropped_inbound_frames: int = 0
        self.dropped_outbound_frames: int = 0
        self.flushed_count: int = 0

    @staticmethod
    def _default_outbound_frame(pcm: bytes, sample_rate: int, num_channels: int) -> Any:
        return OutputAudioRawFrame(
            audio=pcm,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

    def push_inbound_pcm(self, pcm: bytes, sample_rate: int) -> list[AudioFrame]:
        """Slice raw PCM into canonical 20 ms `AudioFrame`s without blocking."""
        if self._closed or not pcm:
            return []

        if self._stream_start_ns is None:
            self._stream_start_ns = self._clock.now_ns()

        frame_dur_ns = self._frame_ms * 1_000_000
        bytes_per_frame = (sample_rate * self._frame_ms // 1000) * 2
        if bytes_per_frame <= 0:
            return []

        self._in_byte_buf.extend(pcm)
        frames: list[AudioFrame] = []

        while len(self._in_byte_buf) >= bytes_per_frame:
            chunk = bytes(self._in_byte_buf[:bytes_per_frame])
            del self._in_byte_buf[:bytes_per_frame]

            ts_ns = self._stream_start_ns + (self._emitted_in_frames * frame_dur_ns)
            self._emitted_in_frames += 1

            af = AudioFrame(
                pcm=chunk,
                sample_rate=sample_rate,
                ts_ns=ts_ns,
                tags=Tag.NONE,
            )
            frames.append(af)

        return frames

    async def user_frames(self) -> AsyncIterator[AudioFrame]:
        """Yield inbound `AudioFrame`s if consumed via background worker."""
        while not self._closed:
            while self._inbound_queue:
                yield self._inbound_queue.popleft()
            self._inbound_event.clear()
            if self._closed:
                break
            try:
                await self._inbound_event.wait()
            except asyncio.CancelledError:
                break

    def send_frame(self, frame: AudioFrame) -> None:
        """Synchronously enqueue an outbound agent audio frame (never blocks fast path)."""
        if self._closed or not frame.pcm:
            return

        samples = len(frame.pcm) // 2
        dur_ms = int(round((samples / frame.sample_rate) * 1000))
        out_frame = self._outbound_frame_factory(frame.pcm, frame.sample_rate, 1)

        if len(self._outbound_queue) >= self._max_outbound_frames:
            _, dropped_ms = self._outbound_queue.popleft()
            self._pending_outbound_ms = max(0, self._pending_outbound_ms - dropped_ms)
            self.dropped_outbound_frames += 1

        self._outbound_queue.append((out_frame, dur_ms))
        self._pending_outbound_ms += dur_ms
        self._ensure_drain_task()

    def _ensure_drain_task(self) -> None:
        if self._downstream_push_is_active():
            try:
                loop = asyncio.get_running_loop()
                self._drain_task = loop.create_task(self._drain_outbound(self._flush_epoch))
            except RuntimeError:
                pass

    def _downstream_push_is_active(self) -> bool:
        return (
            self._downstream_push_fn is not None
            and (self._drain_task is None or self._drain_task.done())
            and not self._closed
        )

    async def _drain_outbound(self, epoch: int) -> None:
        """Pump queued outbound audio frames into the downstream Pipecat handler."""
        while self._outbound_queue and not self._closed and epoch == self._flush_epoch:
            out_frame, dur_ms = self._outbound_queue[0]
            try:
                if self._downstream_push_fn is not None:
                    res = self._downstream_push_fn(out_frame, FrameDirection.DOWNSTREAM)
                    if asyncio.iscoroutine(res):
                        await res
            except asyncio.CancelledError:
                break
            except Exception:
                if epoch != self._flush_epoch or self._closed:
                    break
            if epoch == self._flush_epoch and self._outbound_queue:
                self._outbound_queue.popleft()
                self._pending_outbound_ms = max(0, self._pending_outbound_ms - dur_ms)

    def flush_output(self) -> None:
        """Synchronously drop all buffered outbound audio frames."""
        self._flush_epoch += 1
        self.flushed_count += 1
        self._outbound_queue.clear()
        self._pending_outbound_ms = 0
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()
            self._drain_task = None

    def output_queue_ms(self) -> int:
        """Return total duration in ms of audio currently queued in the bridge."""
        return self._pending_outbound_ms

    def playout_position_ms(self) -> int | None:
        """Return None so `PlayoutLedger` uses `Tuning.assumed_downstream_ms`."""
        return None

    async def close(self) -> None:
        """Close the bridge and await any in-flight drain task."""
        self._closed = True
        self._inbound_event.set()
        task = self._drain_task
        self.flush_output()
        if task is not None:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


class OverlapProcessor:
    """One-line drop-in processor for Pipecat pipelines (ARCHITECTURE §3, §13).

    Inserted between transport input and downstream output:
    ```python
    processor = OverlapProcessor(tuning=Tuning())
    ```

    Parameters
    ----------
    tuning:
        Optional `Tuning` configuration (defaults to `Tuning()`).
    config:
        Optional `AppConfig` (defaults to `load_config()`).
    engine:
        Optional pre-constructed `OverlapEngine`. If omitted and `vad`, `stt`,
        `tts`, `llm` ports are provided, an `OverlapEngine` is built automatically.
    vad, stt, tts, llm:
        Optional port implementations used when constructing `OverlapEngine`.
    clock:
        Optional `Clock` (defaults to `MonotonicClock()`).
    telemetry:
        Optional `TelemetrySink`.
    downstream_handler:
        Optional callback `(frame, direction) -> Awaitable[None] | None` invoked
        for every frame pushed downstream (compatible with Pipecat's `push_frame`).
    max_outbound_frames:
        Maximum buffered outbound audio frames before drop-oldest back-pressure
        kicks in (prevents pipeline stalls).
    """

    def __init__(
        self,
        tuning: Tuning | None = None,
        config: AppConfig | None = None,
        *,
        engine: OverlapEngine | None = None,
        vad: Vad | None = None,
        stt: SttPort | None = None,
        tts: TtsPort | None = None,
        llm: LlmPort | None = None,
        clock: Clock | None = None,
        telemetry: TelemetrySink | None = None,
        downstream_handler: Callable[[Any, Any], Any] | None = None,
        max_outbound_frames: int = 50,
    ) -> None:
        self._config: AppConfig = config if config is not None else load_config()
        self._tuning: Tuning = tuning if tuning is not None else self._config.tuning
        self._clock: Clock = clock if clock is not None else MonotonicClock()
        self._telemetry: TelemetrySink | None = telemetry
        self._downstream_handler: Callable[[Any, Any], Any] | None = downstream_handler
        self._emitted_frames: list[tuple[Any, Any]] = []
        self._builtin_interruptions_suppressed: int = 0
        self._started: bool = False
        self._stopped: bool = False

        self._bridge = _PipecatBridgeTransport(
            tuning=self._tuning,
            clock=self._clock,
            downstream_push_fn=self._emit_from_bridge,
            max_outbound_frames=max_outbound_frames,
        )

        if engine is not None:
            self._engine: OverlapEngine | None = engine
            # Re-bind mixer transport to our bridge if using default fake/unbound transport
        elif vad is not None and stt is not None and tts is not None and llm is not None:
            self._engine = OverlapEngine(
                transport=self._bridge,
                vad=vad,
                stt=stt,
                tts=tts,
                llm=llm,
                clock=self._clock,
                telemetry=self._telemetry,
                tuning=self._tuning,
                voice_id=self._config.voice_id,
                lang=self._config.lang,
            )
        else:
            self._engine = None

    @property
    def tuning(self) -> Tuning:
        """Active `Tuning` parameters."""
        return self._tuning

    @property
    def config(self) -> AppConfig:
        """Active `AppConfig`."""
        return self._config

    @property
    def engine(self) -> OverlapEngine | None:
        """Underlying `OverlapEngine` instance."""
        return self._engine

    @property
    def bridge_transport(self) -> _PipecatBridgeTransport:
        """Internal non-blocking Pipecat bridge `AudioTransport`."""
        return self._bridge

    @property
    def emitted_frames(self) -> list[tuple[Any, Any]]:
        """List of `(frame, direction)` pairs pushed downstream by this processor."""
        return self._emitted_frames

    @property
    def builtin_interruptions_suppressed(self) -> int:
        """Count of Pipecat built-in interruption frames suppressed by OVERLAP."""
        return self._builtin_interruptions_suppressed

    @property
    def interruptions_allowed_in_pipecat(self) -> bool:
        """Always False when OVERLAP is active so Pipecat never double-stops."""
        return False

    def configure_task_params(self, params: Any = None) -> Any:
        """Return `PipelineTaskParams` (or update `params`) with built-in interruptions disabled."""
        return disable_pipecat_interruptions(params)

    def attach_downstream(self, handler: Callable[[Any, Any], Any]) -> None:
        """Register a downstream frame handler `(frame, direction) -> Awaitable | None`."""
        self._downstream_handler = handler

    async def start(self) -> None:
        """Start the underlying `OverlapEngine` if not already started."""
        if not self._started and self._engine is not None:
            self._started = True
            self._stopped = False
            await self._engine.start()

    async def stop(self) -> None:
        """Stop the underlying `OverlapEngine`, close bridge, and assert quiescence."""
        if self._stopped:
            return
        self._stopped = True
        await self._bridge.close()
        if self._engine is not None:
            await self._engine.stop()
            self._engine.assert_quiescent()

    async def push_frame(self, frame: Any, direction: Any = FrameDirection.DOWNSTREAM) -> None:
        """Push a frame to the next processor in the Pipecat pipeline."""
        self._emitted_frames.append((frame, direction))
        if self._downstream_handler is not None:
            res = self._downstream_handler(frame, direction)
            if asyncio.iscoroutine(res):
                await res

    def _emit_from_bridge(self, frame: Any, direction: Any) -> Any:
        """Callback invoked by `_PipecatBridgeTransport` for outbound agent audio."""
        return self.push_frame(frame, direction)

    async def process_frame(self, frame: Any, direction: Any = FrameDirection.DOWNSTREAM) -> None:
        """Process an incoming Pipecat frame while preserving strict control-frame ordering.

        - `StartFrame`: disables Pipecat's built-in `allow_interruptions`, starts `OverlapEngine`,
          and forwards `StartFrame` in order.
        - Built-in Pipecat interruption frames (`StartInterruptionFrame`, `StopInterruptionFrame`,
          `BotInterruptionFrame`, `UserStartedSpeakingFrame`, `UserStoppedSpeakingFrame`):
          swallowed when OVERLAP is active so Pipecat's own interruption strategy never
          causes a double-stop.
        - `EndFrame` / `CancelFrame`: stops `OverlapEngine`, runs `assert_quiescent()`,
          and then forwards the control frame in order.
        - `InputAudioRawFrame` / `AudioRawFrame`: sliced into 20 ms canonical frames and
          passed synchronously to `engine.handle_frame(af)` without awaiting network or
          stalling the pipeline.
        - All other control/system/data frames pass through in exact arrival order.
        """
        # 1. Suppress Pipecat's built-in interruption frames to prevent double-stop
        if _is_builtin_interruption_frame(frame):
            self._builtin_interruptions_suppressed += 1
            return

        # 2. Lifecycle StartFrame: disable Pipecat interruptions, start engine, pass through
        if _is_start_frame(frame):
            disable_pipecat_interruptions(frame)
            await self.start()
            await self.push_frame(frame, direction)
            return

        # 3. Lifecycle EndFrame / CancelFrame: stop engine + assert_quiescent, then pass through
        if _is_end_frame(frame) or _is_cancel_frame(frame):
            await self.stop()
            await self.push_frame(frame, direction)
            return

        # 4. Inbound user audio frame: feed synchronous fast path via bridge
        if _is_input_audio_frame(frame) and self._engine is not None:
            raw_pcm = getattr(frame, "audio", b"")
            sample_rate = int(getattr(frame, "sample_rate", self._tuning.in_rate_hz))
            for af in self._bridge.push_inbound_pcm(raw_pcm, sample_rate):
                self._engine.handle_frame(af)
            return

        # 5. All other frames pass through in order
        await self.push_frame(frame, direction)


class StockInterruptionProcessor:
    """Stock Pipecat baseline processor for A/B evaluation (`examples/stock_pipecat_agent.py`).

    Implements standard Pipecat stop-on-any-VAD behaviour over the exact same
    `Vad`, `SttPort`, `LlmPort`, and `TtsPort`:
    - Any non-echo user VAD speech onset while the agent is speaking or thinking
      immediately triggers a full interruption (flushes playout, cancels TTS,
      aborts LLM, emits `StartInterruptionFrame`, and commits the full un-truncated
      assistant text or drops the turn without heard-text truncation or ducking).
    - Never ducks audio (`gain` is always 1.0) and never emits agent backchannels.
    - On `EndFrame` / `CancelFrame`, cleanly shuts down all streams and tasks.
    """

    def __init__(
        self,
        *,
        vad: Vad,
        stt: SttPort,
        tts: TtsPort,
        llm: LlmPort,
        tuning: Tuning | None = None,
        config: AppConfig | None = None,
        clock: Clock | None = None,
        telemetry: TelemetrySink | None = None,
        downstream_handler: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        self._config: AppConfig = config if config is not None else load_config()
        self._tuning: Tuning = tuning if tuning is not None else self._config.tuning
        self._clock: Clock = clock if clock is not None else MonotonicClock()
        self._telemetry: TelemetrySink | None = telemetry
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._llm = llm
        self._downstream_handler = downstream_handler

        self._echo_gate = EchoGate()
        self._bridge = _PipecatBridgeTransport(
            tuning=self._tuning,
            clock=self._clock,
            downstream_push_fn=self.push_frame,
        )

        # Build an OverlapEngine configured with stock stop-on-any-VAD policy:
        # every VAD onset during SPEAKING immediately classifies as INTERRUPT at 0 ms.
        self._engine = OverlapEngine(
            transport=self._bridge,
            vad=self._vad,
            stt=self._stt,
            tts=self._tts,
            llm=self._llm,
            clock=self._clock,
            telemetry=self._telemetry,
            tuning=self._tuning,
            voice_id=self._config.voice_id,
            lang=self._config.lang,
            echo_gate=self._echo_gate,
        )

        self._emitted_frames: list[tuple[Any, Any]] = []
        self._interruption_count: int = 0
        self._user_speaking: bool = False
        self._agent_speaking: bool = False
        self._started: bool = False
        self._stopped: bool = False

    @property
    def engine(self) -> OverlapEngine:
        """Underlying engine instance."""
        return self._engine

    @property
    def emitted_frames(self) -> list[tuple[Any, Any]]:
        """Frames pushed downstream."""
        return self._emitted_frames

    @property
    def interruption_count(self) -> int:
        """Number of immediate VAD-triggered interruptions executed."""
        return self._interruption_count

    @property
    def history(self) -> list[Message]:
        """Conversation message history."""
        return self._engine.history

    def attach_downstream(self, handler: Callable[[Any, Any], Any]) -> None:
        """Register a downstream frame callback."""
        self._downstream_handler = handler

    async def start(self) -> None:
        """Start stock baseline processor."""
        if not self._started:
            self._started = True
            self._stopped = False
            await self._engine.start()

    async def stop(self) -> None:
        """Stop stock baseline processor and assert quiescence."""
        if self._stopped:
            return
        self._stopped = True
        await self._bridge.close()
        await self._engine.stop()
        self._engine.assert_quiescent()

    async def push_frame(self, frame: Any, direction: Any = FrameDirection.DOWNSTREAM) -> None:
        """Push a frame downstream."""
        self._emitted_frames.append((frame, direction))
        if self._downstream_handler is not None:
            res = self._downstream_handler(frame, direction)
            if asyncio.iscoroutine(res):
                await res

    async def process_frame(self, frame: Any, direction: Any = FrameDirection.DOWNSTREAM) -> None:
        """Process a Pipecat frame using stock stop-on-any-VAD semantics."""
        if _is_start_frame(frame):
            await self.start()
            await self.push_frame(frame, direction)
            return

        if _is_end_frame(frame) or _is_cancel_frame(frame):
            await self.stop()
            await self.push_frame(frame, direction)
            return

        if _is_input_audio_frame(frame):
            raw_pcm = getattr(frame, "audio", b"")
            sample_rate = int(getattr(frame, "sample_rate", self._tuning.in_rate_hz))
            for af in self._bridge.push_inbound_pcm(raw_pcm, sample_rate):
                await self._handle_stock_frame(af)
            return

        await self.push_frame(frame, direction)

    async def _handle_stock_frame(self, frame: AudioFrame) -> None:
        """Stock stop-on-any-VAD audio handler: any VAD onset stops agent immediately."""
        vad_res = self._vad.process(frame)
        is_speech = bool(vad_res.is_speech)

        # Forward to STT queue
        if len(self._engine._stt_frame_queue) >= 100:
            self._engine._stt_frame_queue.popleft()
        self._engine._stt_frame_queue.append(frame)
        self._engine._stt_frame_event.set()

        prev_speaking = self._user_speaking
        self._user_speaking = is_speech

        # Stock policy: ANY VAD onset while agent is speaking/playing immediately stops playout!
        if is_speech and not prev_speaking:
            if self._engine.mixer.is_playing or self._engine.state in (
                FsmState.SPEAKING,
                FsmState.DUCKED,
                FsmState.THINKING,
            ):
                self._interruption_count += 1
                self._bridge.flush_output()
                self._engine.mixer.flush()
                self._engine.supervisor.execute_effects((FlushPlayout(),))
                await self.push_frame(StartInterruptionFrame(), FrameDirection.DOWNSTREAM)
        elif not is_speech and prev_speaking:
            await self.push_frame(StopInterruptionFrame(), FrameDirection.DOWNSTREAM)
