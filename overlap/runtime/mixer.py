"""Audio playout mixer coordinating TTS, clip injection, gain ramps, and transport pacing.

Per ARCHITECTURE §8.1 and §10:
- Pulls/receives AudioChunks, re-framing into uniform 20 ms frames (INV-L1).
- Linear per-sample GainRamp for click-free ducking (INV-L3) and release (INV-L8).
- Synchronous set_gain executed inline on the fast path (INV-L2, INV-L13).
- Bounded downstream look-ahead <= Tuning.max_lookahead_ms (INV-L9).
- Clip injection when TTS is idle, tagging frames with Tag.SELF_BACKCHANNEL.
- Synchronous flush dropping un-emitted frames with zero leakage (INV-L5, INV-L7).
- Ledger accounting tracking samples queued, emitted, and heard (INV-T1).
"""

from __future__ import annotations

import asyncio

from overlap.core.audio import EchoGate, GainRamp
from overlap.core.clock import Clock, TimerHandle
from overlap.core.playout import PlayoutLedger, PlayoutSnapshot
from overlap.core.types import AudioFrame, Tag, Tuning
from overlap.ports.telemetry import TelemetrySink
from overlap.ports.transport import AudioTransport
from overlap.ports.tts import AudioChunk, AudioClip, TtsStream


class PlayoutMixer:
    """Asyncio runtime playout mixer for full-duplex conversational audio."""

    def __init__(
        self,
        transport: AudioTransport,
        clock: Clock,
        ledger: PlayoutLedger | None = None,
        tuning: Tuning | None = None,
        echo_gate: EchoGate | None = None,
        target_lookahead_ms: int | None = None,
        telemetry: TelemetrySink | None = None,
    ) -> None:
        self._transport = transport
        self._clock = clock
        self._tuning = tuning or Tuning()
        self._ledger = ledger or PlayoutLedger(tuning=self._tuning)
        self._echo_gate = echo_gate
        self._telemetry = telemetry

        self._sample_rate_hz = self._tuning.out_rate_hz
        self._frame_ms = self._tuning.frame_ms
        self._frame_samples = int(self._sample_rate_hz * (self._frame_ms / 1000.0))
        self._frame_bytes = self._frame_samples * 2

        # Bounded lookahead target: at least 1 frame, capped by max_lookahead_ms (INV-L9)
        if target_lookahead_ms is None:
            self._target_lookahead_ms = min(self._frame_ms, self._tuning.max_lookahead_ms)
        else:
            self._target_lookahead_ms = max(
                self._frame_ms, min(target_lookahead_ms, self._tuning.max_lookahead_ms)
            )

        self._gain_ramp = GainRamp(
            current=1.0,
            target=1.0,
            ramp_ms=0,
            sample_rate=self._sample_rate_hz,
        )

        self._pcm_buffer = bytearray()
        self._clip_buffer = bytearray()
        self._clip_tag = Tag.SELF_BACKCHANNEL
        self._is_playing_clip = False
        self._tts_finished = False

        self._current_turn_id = ""
        self._current_epoch = 0
        self._flushed_epochs: set[int] = set()
        self._duck_pending_emit: bool = False
        self._clip_first_sample_pending_emit: bool = False

        self._timer_handle: TimerHandle | None = None
        self._emitted_until_ns = self._clock.now_ns()
        self._active_tts_task: asyncio.Task[None] | None = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def lookahead_ms(self) -> int:
        """Current duration of audio emitted downstream ahead of clock time in ms."""
        now_ns = self._clock.now_ns()
        if self._emitted_until_ns <= now_ns:
            return 0
        return int((self._emitted_until_ns - now_ns) // 1_000_000)

    @property
    def is_tts_active(self) -> bool:
        """True if TTS audio is currently queued or being pulled."""
        has_task = self._active_tts_task is not None and not self._active_tts_task.done()
        return len(self._pcm_buffer) > 0 or has_task

    @property
    def is_clip_active(self) -> bool:
        """True if backchannel clip audio is currently queued or playing."""
        return self._is_playing_clip or len(self._clip_buffer) > 0

    @property
    def is_playing(self) -> bool:
        """True if any audio is queued, actively emitting, or within lookahead."""
        return self.is_tts_active or self.is_clip_active or self.lookahead_ms > 0

    @property
    def current_gain(self) -> float:
        """Current instantaneous gain applied by the gain ramp."""
        return self._gain_ramp.current_gain

    @property
    def target_gain(self) -> float:
        """Target gain factor of the gain ramp."""
        return self._gain_ramp.target_gain

    @property
    def ledger(self) -> PlayoutLedger:
        """Underlying playout ledger."""
        return self._ledger

    @property
    def transport(self) -> AudioTransport:
        """Underlying audio transport."""
        return self._transport

    @property
    def clock(self) -> Clock:
        """Clock driving cadence."""
        return self._clock

    @property
    def tuning(self) -> Tuning:
        """Tuning configuration parameters."""
        return self._tuning

    @property
    def turn_id(self) -> str:
        """Current turn identifier."""
        return self._current_turn_id

    @property
    def epoch(self) -> int:
        """Current epoch."""
        return self._current_epoch

    # =========================================================================
    # Turn and Session Lifecycle
    # =========================================================================

    def start_turn(self, turn_id: str, epoch: int) -> None:
        """Begin accounting for a new turn and epoch."""
        self._current_turn_id = turn_id
        self._current_epoch = epoch
        self._tts_finished = False
        self._flushed_epochs.discard(epoch)
        self._ledger.start_turn(turn_id, epoch)

    # =========================================================================
    # Audio Ingestion (TTS and Clips)
    # =========================================================================

    def enqueue_chunk(self, chunk: AudioChunk) -> None:
        """Enqueue an incoming AudioChunk from TTS synthesis.

        Re-frames into 20 ms frames and enforces epoch hygiene (INV-L7).
        """
        if chunk.sample_rate != self._sample_rate_hz:
            msg = (
                f"Unsupported chunk sample rate {chunk.sample_rate}; "
                f"expected {self._sample_rate_hz}"
            )
            raise ValueError(msg)

        # Stale epoch filter: ignore flushed or obsolete epochs
        if chunk.epoch in self._flushed_epochs:
            return
        if self._current_epoch != 0 and chunk.epoch != self._current_epoch:
            return

        if chunk.pcm:
            num_samples = len(chunk.pcm) // 2
            self._ledger.record_queued(num_samples)
            self._pcm_buffer.extend(chunk.pcm)

        if chunk.is_last:
            self._tts_finished = True

        self._schedule_or_pump()

    def play_clip(self, clip: AudioClip | bytes, tag: Tag = Tag.SELF_BACKCHANNEL) -> None:
        """Inject a pre-synthesized audio clip into playout mixer (ARCHITECTURE §8.1).

        Only permitted when agent TTS is idle (failure mode 4).
        Registers self-audio window with EchoGate if present.
        """
        if self.is_tts_active:
            raise RuntimeError("Cannot inject clip while TTS audio is playing")

        if tag == Tag.NONE or not bool(tag & Tag.SELF_BACKCHANNEL):
            raise ValueError("Clip frames must be tagged with Tag.SELF_BACKCHANNEL")

        pcm = clip.pcm if isinstance(clip, AudioClip) else clip
        if not pcm:
            return

        num_samples = len(pcm) // 2
        self._ledger.record_queued(num_samples)
        self._clip_buffer.extend(pcm)
        self._clip_tag = tag
        self._is_playing_clip = True
        self._clip_first_sample_pending_emit = True

        if self._echo_gate is not None:
            start_ns = self._clock.now_ns()
            dur_ns = int((num_samples * 1_000_000_000) / self._sample_rate_hz)
            end_ns = start_ns + dur_ns
            self._echo_gate.register_self_audio_window(start_ns, end_ns, tag=tag)

        self._schedule_or_pump()

    def fade_out_clip(self, dur_ms: int = 40) -> None:
        """Quickly stop/fade out playing backchannel clip."""
        del dur_ms  # Fast-path drops buffered clip audio immediately
        self._clip_buffer.clear()
        self._is_playing_clip = False

    # =========================================================================
    # Streaming TTS Task Attachment
    # =========================================================================

    async def play_tts_stream(self, stream: TtsStream, epoch: int) -> None:
        """Pull AudioChunks asynchronously from active TtsStream and enqueue them."""
        try:
            async for chunk in stream.chunks():
                if epoch in self._flushed_epochs or epoch < self._current_epoch:
                    break
                self.enqueue_chunk(chunk)
        except asyncio.CancelledError:
            pass

    def attach_tts_stream(self, stream: TtsStream, epoch: int) -> asyncio.Task[None]:
        """Attach a TtsStream, pulling chunks in a background asyncio task."""
        task = asyncio.create_task(self.play_tts_stream(stream, epoch))
        self._active_tts_task = task
        return task

    # =========================================================================
    # Gain Control (Synchronous Fast Path)
    # =========================================================================

    def set_gain(self, gain: float, ramp_ms: int = 0) -> None:
        """Synchronously adjust playout mixer gain with optional linear ramp (INV-L3, INV-L8).

        Executed inline on fast path; never awaits or yields (INV-L2, INV-L13).
        """
        self._gain_ramp.set_target(gain, ramp_ms)
        if gain < 1.0:
            self._duck_pending_emit = True
        else:
            self._duck_pending_emit = False

    def duck(self, gain: float | None = None, ramp_ms: int | None = None) -> None:
        """Helper to apply ducking gain (INV-L3: default 0.3, <= 20 ms linear ramp)."""
        target_gain = self._tuning.duck_gain if gain is None else gain
        target_ramp = self._tuning.duck_attack_ms if ramp_ms is None else ramp_ms
        self.set_gain(target_gain, target_ramp)

    def release_gain(self, ramp_ms: int | None = None) -> None:
        """Helper to release ducking back to unity (INV-L8: 150 ms linear ramp to 1.0)."""
        target_ramp = self._tuning.duck_release_ms if ramp_ms is None else ramp_ms
        self.set_gain(1.0, target_ramp)

    # =========================================================================
    # Flush (Synchronous Fast Path)
    # =========================================================================

    def flush(
        self,
        downstream_latency_ms: int | None = None,
        transport_queue_ms: int = 0,
        clause_audio_total_ms: int | None = None,
    ) -> PlayoutSnapshot:
        """Drop un-emitted audio, flush transport, and return ledger snapshot (INV-L5, INV-L7).

        Zero frames leak to transport after flush.
        """
        if self._timer_handle is not None:
            self._clock.cancel(self._timer_handle)
            self._timer_handle = None

        if self._active_tts_task is not None and not self._active_tts_task.done():
            self._active_tts_task.cancel()
            self._active_tts_task = None

        self._pcm_buffer.clear()
        self._clip_buffer.clear()
        self._is_playing_clip = False
        self._tts_finished = False

        if self._current_epoch != 0:
            self._flushed_epochs.add(self._current_epoch)

        self._emitted_until_ns = self._clock.now_ns()
        self._transport.flush_output()

        return self._ledger.flush(
            downstream_latency_ms=downstream_latency_ms,
            transport_queue_ms=transport_queue_ms,
            clause_audio_total_ms=clause_audio_total_ms,
        )

    def close(self) -> None:
        """Clean up timers and tasks."""
        if self._timer_handle is not None:
            self._clock.cancel(self._timer_handle)
            self._timer_handle = None
        if self._active_tts_task is not None and not self._active_tts_task.done():
            self._active_tts_task.cancel()
            self._active_tts_task = None

    # =========================================================================
    # Internal Cadence, Lookahead & Re-framing
    # =========================================================================

    def has_audio_to_emit(self) -> bool:
        """True if sufficient audio exists to form at least one 20 ms frame."""
        if len(self._clip_buffer) > 0:
            return True
        if len(self._pcm_buffer) >= self._frame_bytes:
            return True
        if self._tts_finished and len(self._pcm_buffer) > 0:
            return True
        return False

    def _schedule_or_pump(self) -> None:
        """Pump frames up to target lookahead and schedule clock cadence."""
        while self.has_audio_to_emit():
            # Never exceed max_lookahead_ms (INV-L9)
            if self.lookahead_ms + self._frame_ms > self._tuning.max_lookahead_ms:
                break
            # Do not exceed target lookahead
            if self.lookahead_ms >= self._target_lookahead_ms:
                break
            if not self._emit_one_frame():
                break

        if self.has_audio_to_emit() and self._timer_handle is None:
            delay_ms = max(
                1,
                self.lookahead_ms - self._target_lookahead_ms + self._frame_ms,
            )
            self._timer_handle = self._clock.call_later(delay_ms, self._on_tick)

    def _on_tick(self) -> None:
        """Cadence timer callback driven by Clock."""
        self._timer_handle = None
        self._schedule_or_pump()

    def _emit_one_frame(self) -> bool:
        """Construct, ramp, ledger-record, and send a single 20 ms frame."""
        # Hard check for bounded lookahead (INV-L9)
        if self.lookahead_ms + self._frame_ms > self._tuning.max_lookahead_ms:
            return False

        if len(self._clip_buffer) > 0:
            tag = self._clip_tag
            if len(self._clip_buffer) >= self._frame_bytes:
                raw_pcm = bytes(self._clip_buffer[: self._frame_bytes])
                del self._clip_buffer[: self._frame_bytes]
            else:
                raw_pcm = bytes(self._clip_buffer) + b"\x00" * (
                    self._frame_bytes - len(self._clip_buffer)
                )
                self._clip_buffer.clear()
            if len(self._clip_buffer) == 0:
                self._is_playing_clip = False
        elif len(self._pcm_buffer) >= self._frame_bytes or (
            self._tts_finished and len(self._pcm_buffer) > 0
        ):
            tag = Tag.SELF_TTS
            if len(self._pcm_buffer) >= self._frame_bytes:
                raw_pcm = bytes(self._pcm_buffer[: self._frame_bytes])
                del self._pcm_buffer[: self._frame_bytes]
            else:
                raw_pcm = bytes(self._pcm_buffer) + b"\x00" * (
                    self._frame_bytes - len(self._pcm_buffer)
                )
                self._pcm_buffer.clear()
        else:
            return False

        # Apply GainRamp (INV-L3, INV-L8)
        processed_pcm = self._gain_ramp.apply(raw_pcm)

        # Monotonic, contiguous timestamp aligned with Clock
        now_ns = self._clock.now_ns()
        frame_dur_ns = int(self._frame_ms * 1_000_000)
        if self._emitted_until_ns < now_ns:
            ts_ns = now_ns
            self._emitted_until_ns = now_ns + frame_dur_ns
        else:
            ts_ns = self._emitted_until_ns
            self._emitted_until_ns += frame_dur_ns

        frame = AudioFrame(
            pcm=processed_pcm,
            sample_rate=self._sample_rate_hz,
            ts_ns=ts_ns,
            tags=tag,
        )

        self._ledger.record_emitted(self._frame_samples)
        self._transport.send_frame(frame)

        if self._duck_pending_emit and self._gain_ramp.target_gain < 1.0:
            self._duck_pending_emit = False
            if self._telemetry is not None:
                self._telemetry.emit(
                    "duck.applied",
                    now_ns,
                    gain=self._gain_ramp.target_gain,
                    epoch=self._current_epoch,
                )

        if self._clip_first_sample_pending_emit and bool(tag & Tag.SELF_BACKCHANNEL):
            self._clip_first_sample_pending_emit = False
            if self._telemetry is not None:
                self._telemetry.emit(
                    "bc.agent.clip_start",
                    ts_ns,
                    epoch=self._current_epoch,
                )

        return True
