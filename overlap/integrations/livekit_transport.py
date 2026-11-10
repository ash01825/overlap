"""LiveKit WebRTC audio transport adapter.

Implements the AudioTransport protocol over a LiveKit RTC interface (or fake RTC
layer for hermetic tests). Supports:
- Inbound user track resampling to core's canonical rate (e.g. 16 kHz mono int16)
  and re-chunking to exact 20 ms (`Tuning.frame_ms`) frames with zero sample drift.
- Non-blocking synchronous `send_frame` with bounded queue (drops oldest on backpressure)
  and outbound resampling if the RTC AudioSource uses a different sample rate (e.g. 48 kHz).
- Synchronous `flush_output` clearing both the local outbound buffer and the
  underlying LiveKit `AudioSource` queue (`clear_queue()`).
- Accurate `output_queue_ms` and `playout_position_ms` estimation accounting for
  queued audio and downstream jitter/playout buffer latency.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from overlap.core.clock import Clock, MonotonicClock
from overlap.core.types import AudioFrame, Tag, Tuning
from overlap.ports.transport import AudioTransport


@runtime_checkable
class RtcAudioFrame(Protocol):
    """Protocol matching `livekit.rtc.AudioFrame`."""

    @property
    def data(self) -> bytes | memoryview | bytearray:
        """Raw 16-bit signed little-endian PCM bytes."""
        ...

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz (e.g., 16000, 48000)."""
        ...

    @property
    def num_channels(self) -> int:
        """Number of audio channels (1 for mono, 2 for stereo)."""
        ...

    @property
    def samples_per_channel(self) -> int:
        """Number of samples per channel in this frame."""
        ...


@runtime_checkable
class RtcAudioFrameEvent(Protocol):
    """Protocol matching `livekit.rtc.AudioFrameEvent` yielded by `AudioStream`."""

    @property
    def frame(self) -> RtcAudioFrame:
        """The underlying RTC audio frame."""
        ...


@runtime_checkable
class RtcAudioSource(Protocol):
    """Protocol matching `livekit.rtc.AudioSource`."""

    @property
    def sample_rate(self) -> int:
        """Sample rate expected by the AudioSource."""
        ...

    @property
    def num_channels(self) -> int:
        """Number of channels expected by the AudioSource."""
        ...

    @property
    def queued_duration(self) -> float:
        """Duration in seconds of audio currently queued in the native AudioSource."""
        ...

    def clear_queue(self) -> None:
        """Synchronously clear all audio queued in the native AudioSource."""
        ...

    async def capture_frame(self, frame: Any) -> None:
        """Submit an RTC audio frame for WebRTC playout."""
        ...


@dataclass(frozen=True, slots=True)
class SimpleRtcAudioFrame:
    """Lightweight RTC frame representation used when constructing outbound frames."""

    data: bytes
    sample_rate: int
    num_channels: int
    samples_per_channel: int


def _extract_rtc_frame(item: Any) -> RtcAudioFrame:
    """Extract an RtcAudioFrame from either an AudioFrameEvent or direct AudioFrame."""
    if hasattr(item, "frame") and not hasattr(item, "sample_rate"):
        return item.frame  # type: ignore[no-any-return]
    return item  # type: ignore[no-any-return]


def _downmix_to_mono_int16(raw_bytes: bytes | memoryview | bytearray, num_channels: int) -> bytes:
    """Convert multi-channel int16 PCM to mono int16 PCM by averaging channels."""
    if num_channels <= 1:
        return bytes(raw_bytes)
    samples = np.frombuffer(raw_bytes, dtype="<i2")
    total_frames = len(samples) // num_channels
    if total_frames == 0:
        return b""
    reshaped = samples[: total_frames * num_channels].reshape(total_frames, num_channels)
    mono = np.mean(reshaped.astype(np.float32), axis=1)
    mono_i16 = np.clip(np.round(mono), -32768, 32767).astype("<i2")
    return bytes(mono_i16.tobytes())


class _StreamResampler:
    """Stateful streaming linear-interpolating resampler for mono int16 PCM.

    Maintains fractional sample phase across frame boundaries so that arbitrary
    chunk sizes and rate ratios (e.g., 48000 <-> 16000, 44100 <-> 16000) never
    accumulate timing drift over long sessions.
    """

    def __init__(self, in_rate: int, out_rate: int) -> None:
        if in_rate <= 0 or out_rate <= 0:
            raise ValueError(f"Sample rates must be positive: {in_rate} -> {out_rate}")
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._step = in_rate / out_rate
        self._pos: float = 0.0
        self._prev_sample: float | None = None

    def reset(self) -> None:
        """Reset internal resampler state."""
        self._pos = 0.0
        self._prev_sample = None

    def process(self, pcm: bytes) -> bytes:
        """Resample mono int16 PCM bytes from `in_rate` to `out_rate`."""
        if not pcm:
            return b""
        if self.in_rate == self.out_rate:
            return pcm

        in_samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
        n_in = len(in_samples)
        if n_in == 0:
            return b""

        # Fast exact decimation for integer multiples when phase-aligned (e.g. 48k -> 16k)
        if (
            self.in_rate % self.out_rate == 0
            and self._pos == 0.0
            and self._prev_sample is None
            and n_in % (self.in_rate // self.out_rate) == 0
        ):
            factor = self.in_rate // self.out_rate
            reshaped = in_samples.reshape(-1, factor)
            out_f = np.mean(reshaped, axis=1)
            return bytes(np.clip(np.round(out_f), -32768, 32767).astype("<i2").tobytes())

        # Fast exact upsampling for integer multiples when phase-aligned (e.g. 16k -> 48k)
        if self.out_rate % self.in_rate == 0 and self._pos == 0.0 and self._prev_sample is None:
            up_factor = self.out_rate // self.in_rate
            num_out = n_in * up_factor
            positions = np.arange(num_out, dtype=np.float64) / float(up_factor)
            idx0 = np.floor(positions).astype(np.int64)
            frac = positions - idx0
            idx1 = np.minimum(idx0 + 1, n_in - 1)
            out_vals = in_samples[idx0] * (1.0 - frac) + in_samples[idx1] * frac
            return bytes(np.clip(np.round(out_vals), -32768, 32767).astype("<i2").tobytes())

        # Prepend previous boundary sample if needed so indices in [-1, n_in - 1) can interpolate
        if self._prev_sample is not None:
            ext = np.empty(n_in + 1, dtype=np.float64)
            ext[0] = self._prev_sample
            ext[1:] = in_samples
            offset = 1.0  # index 0 in in_samples is index 1 in ext
        else:
            ext = in_samples
            offset = 0.0

        max_idx = float(n_in - 1)
        if self._pos > max_idx:
            self._pos -= float(n_in)
            self._prev_sample = float(in_samples[-1])
            return b""

        num_out = int(np.floor((max_idx - self._pos) / self._step)) + 1
        if num_out <= 0:
            self._pos -= float(n_in)
            self._prev_sample = float(in_samples[-1])
            return b""

        positions = self._pos + np.arange(num_out, dtype=np.float64) * self._step
        ext_positions = positions + offset

        idx0 = np.floor(ext_positions).astype(np.int64)
        frac = ext_positions - idx0
        idx1 = np.minimum(idx0 + 1, len(ext) - 1)

        out_vals = ext[idx0] * (1.0 - frac) + ext[idx1] * frac
        out_i16 = np.clip(np.round(out_vals), -32768, 32767).astype("<i2")

        next_pos = self._pos + num_out * self._step
        self._pos = next_pos - float(n_in)
        self._prev_sample = float(in_samples[-1])

        return bytes(out_i16.tobytes())


_DEFAULT_TUNING = Tuning()


class LiveKitTransport(AudioTransport):
    """LiveKit WebRTC audio transport adapter implementing `AudioTransport`.

    Parameters
    ----------
    audio_stream:
        Async iterable yielding `livekit.rtc.AudioFrameEvent` or `RtcAudioFrame`
        objects from the subscribed remote user audio track.
    audio_source:
        `livekit.rtc.AudioSource` (or fake equivalent) for publishing agent audio.
    tuning:
        `Tuning` configuration providing canonical `frame_ms`, `in_rate_hz`, `out_rate_hz`.
    clock:
        Injected `Clock` for deterministic timestamp generation.
    rtc_frame_factory:
        Optional factory `(data, sample_rate, num_channels, samples_per_channel) -> frame`
        used to wrap outbound PCM bytes for `audio_source.capture_frame`. Defaults to
        `SimpleRtcAudioFrame` (or `livekit.rtc.AudioFrame` if available).
    downstream_playout_delay_ms_fn:
        Optional callback returning the measured downstream WebRTC jitter/playout delay
        in milliseconds (e.g. from RTC stats or loopback probe), or `None` if unavailable.
    max_outbound_frames:
        Maximum number of 20 ms frames buffered in the Python outbound queue before
        dropping the oldest frame under backpressure (protects the synchronous fast path).
    """

    def __init__(
        self,
        audio_stream: AsyncIterable[Any],
        audio_source: RtcAudioSource,
        *,
        tuning: Tuning = _DEFAULT_TUNING,
        clock: Clock | None = None,
        rtc_frame_factory: Callable[[bytes, int, int, int], Any] | None = None,
        downstream_playout_delay_ms_fn: Callable[[], int | None] | None = None,
        max_outbound_frames: int = 50,
    ) -> None:
        self._audio_stream = audio_stream
        self._audio_source = audio_source
        self._tuning = tuning
        self._clock: Clock = clock if clock is not None else MonotonicClock()
        self._rtc_frame_factory = rtc_frame_factory or self._default_frame_factory
        self._downstream_delay_fn = downstream_playout_delay_ms_fn
        self._max_outbound_frames = max(1, max_outbound_frames)

        # Canonical frame sizing for core (20 ms at in_rate_hz = 320 samples = 640 bytes)
        self._target_in_rate = self._tuning.in_rate_hz
        self._target_out_rate = self._tuning.out_rate_hz
        self._frame_ms = self._tuning.frame_ms
        self._in_samples_per_frame = (self._target_in_rate * self._frame_ms) // 1000
        self._in_bytes_per_frame = self._in_samples_per_frame * 2

        # Inbound re-chunking & resampling state
        self._in_resampler: _StreamResampler | None = None
        self._in_byte_buf = bytearray()
        self._emitted_in_frames: int = 0
        self._stream_start_ns: int | None = None

        # Outbound resampling & non-blocking queue state
        self._source_rate = getattr(self._audio_source, "sample_rate", self._target_out_rate)
        self._source_channels = getattr(self._audio_source, "num_channels", 1)
        self._out_resampler = _StreamResampler(self._target_out_rate, self._source_rate)

        self._outbound_queue: deque[tuple[Any, int]] = deque()
        self._pending_Outbound_ms: int = 0
        self._flush_epoch: int = 0
        self._drain_task: asyncio.Task[None] | None = None
        self._closed: bool = False

        # Playout accounting (cumulative ms sent vs queued since last flush/reset)
        self._emitted_samples_at_core_rate: int = 0
        self._dropped_samples_at_core_rate: int = 0
        self.dropped_outbound_frames: int = 0

    @staticmethod
    def _default_frame_factory(
        data: bytes, sample_rate: int, num_channels: int, samples_per_channel: int
    ) -> Any:
        try:
            import importlib

            rtc_mod = importlib.import_module("livekit.rtc")
            af_cls = getattr(rtc_mod, "AudioFrame", None)
            if af_cls is not None:
                if hasattr(af_cls, "create"):
                    frame = af_cls.create(sample_rate, num_channels, samples_per_channel)
                    frame.data[: len(data)] = data
                    return frame
                return af_cls(
                    data=data,
                    sample_rate=sample_rate,
                    num_channels=num_channels,
                    samples_per_channel=samples_per_channel,
                )
        except Exception:
            pass
        return SimpleRtcAudioFrame(
            data=data,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=samples_per_channel,
        )

    async def user_frames(self) -> AsyncIterator[AudioFrame]:
        """Yield canonical 20 ms `AudioFrame`s at `Tuning.in_rate_hz` with zero drift.

        Re-chunks arbitrary incoming frame sizes (e.g. 10 ms or 30 ms WebRTC frames)
        and resamples from the remote track's sample rate (e.g. 48 kHz) to
        `Tuning.in_rate_hz` (16 kHz). Frame timestamps advance strictly by
        `frame_ms` (20,000,000 ns) per frame from the initial stream timestamp so
        cumulative cadence never drifts over long sessions.
        """
        frame_dur_ns = self._frame_ms * 1_000_000

        async for raw_item in self._audio_stream:
            if self._closed:
                break

            rtc_frame = _extract_rtc_frame(raw_item)
            src_rate = int(rtc_frame.sample_rate)
            src_channels = int(getattr(rtc_frame, "num_channels", 1))
            raw_bytes = rtc_frame.data

            if self._stream_start_ns is None:
                self._stream_start_ns = self._clock.now_ns()

            mono_pcm = _downmix_to_mono_int16(raw_bytes, src_channels)
            if not mono_pcm:
                continue

            if src_rate != self._target_in_rate:
                if self._in_resampler is None or self._in_resampler.in_rate != src_rate:
                    self._in_resampler = _StreamResampler(src_rate, self._target_in_rate)
                resampled_pcm = self._in_resampler.process(mono_pcm)
            else:
                resampled_pcm = mono_pcm

            self._in_byte_buf.extend(resampled_pcm)

            while len(self._in_byte_buf) >= self._in_bytes_per_frame:
                chunk = bytes(self._in_byte_buf[: self._in_bytes_per_frame])
                del self._in_byte_buf[: self._in_bytes_per_frame]

                ts_ns = self._stream_start_ns + (self._emitted_in_frames * frame_dur_ns)
                self._emitted_in_frames += 1

                yield AudioFrame(
                    pcm=chunk,
                    sample_rate=self._target_in_rate,
                    ts_ns=ts_ns,
                    tags=Tag.NONE,
                )

    def send_frame(self, frame: AudioFrame) -> None:
        """Synchronously enqueue an outbound agent audio frame without blocking.

        Resamples `frame.pcm` from `frame.sample_rate` to `audio_source.sample_rate`
        if needed, enforces `max_outbound_frames` bounded queue with drop-oldest
        backpressure handling, and schedules background draining to `audio_source`.
        """
        if self._closed or not frame.pcm:
            return

        core_samples = len(frame.pcm) // 2
        frame_dur_ms = int(round((core_samples / frame.sample_rate) * 1000))

        # Resample to Core -> Source rate
        if frame.sample_rate != self._source_rate:
            if self._out_resampler.in_rate != frame.sample_rate:
                self._out_resampler = _StreamResampler(frame.sample_rate, self._source_rate)
            out_pcm = self._out_resampler.process(frame.pcm)
        else:
            out_pcm = frame.pcm

        samples_per_ch = len(out_pcm) // 2
        if self._source_channels > 1:
            mono_arr = np.frombuffer(out_pcm, dtype="<i2")
            tiled = np.tile(mono_arr[:, np.newaxis], (1, self._source_channels))
            out_pcm = tiled.astype("<i2").tobytes()

        rtc_frame = self._rtc_frame_factory(
            out_pcm,
            self._source_rate,
            self._source_channels,
            samples_per_ch,
        )

        # Bounded queue: drop oldest on overflow so send_frame NEVER blocks or grows unboundedly
        if len(self._outbound_queue) >= self._max_outbound_frames:
            _dropped_frame, dropped_ms = self._outbound_queue.popleft()
            self._pending_Outbound_ms = max(0, self._pending_Outbound_ms - dropped_ms)
            dropped_samples = (dropped_ms * self._target_out_rate) // 1000
            self._dropped_samples_at_core_rate += dropped_samples
            self.dropped_outbound_frames += 1

        self._outbound_queue.append((rtc_frame, frame_dur_ms))
        self._pending_Outbound_ms += frame_dur_ms
        self._emitted_samples_at_core_rate += core_samples

        self._ensure_drain_task()

    def _ensure_drain_task(self) -> None:
        """Schedule the background async drain task if an event loop is running."""
        if self._drain_task is not None and not self._drain_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
            self._drain_task = loop.create_task(self._drain_outbound(self._flush_epoch))
        except RuntimeError:
            # No running event loop (e.g., purely synchronous contract check):
            # if capture_frame is synchronous or queued in memory, keep in _outbound_queue.
            pass

    async def _drain_outbound(self, epoch: int) -> None:
        """Background task pumping frames from `_outbound_queue` into `audio_source`."""
        while self._outbound_queue and not self._closed and epoch == self._flush_epoch:
            rtc_frame, dur_ms = self._outbound_queue[0]
            try:
                await self._audio_source.capture_frame(rtc_frame)
            except asyncio.CancelledError:
                break
            except Exception:
                # If flushed mid-await or closed, stop draining
                if epoch != self._flush_epoch or self._closed:
                    break
            if epoch == self._flush_epoch and self._outbound_queue:
                self._outbound_queue.popleft()
                self._pending_Outbound_ms = max(0, self._pending_Outbound_ms - dur_ms)

    def flush_output(self) -> None:
        """Synchronously drop all audio queued in the adapter and LiveKit `AudioSource`."""
        self._flush_epoch += 1
        self._outbound_queue.clear()
        self._pending_Outbound_ms = 0
        self._out_resampler.reset()
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()
            self._drain_task = None
        self._audio_source.clear_queue()

    def output_queue_ms(self) -> int:
        """Return total duration in ms of audio currently queued in adapter + `AudioSource`."""
        source_queued_sec = float(getattr(self._audio_source, "queued_duration", 0.0))
        source_queued_ms = max(0, int(round(source_queued_sec * 1000.0)))
        return self._pending_Outbound_ms + source_queued_ms

    def playout_position_ms(self) -> int | None:
        """Return best estimate of cumulative ms heard by the user, or `None` if unmeasured.

        When a `downstream_playout_delay_ms_fn` is provided and returns a non-None
        measured WebRTC jitter/playout delay `d_ms`, returns:
        `max(0, emitted_ms - output_queue_ms() - d_ms)`.
        Otherwise returns `None` so `PlayoutLedger` falls back to `Tuning.assumed_downstream_ms`.
        """
        if self._downstream_delay_fn is None:
            return None
        downstream_ms = self._downstream_delay_fn()
        if downstream_ms is None:
            return None

        active_samples = max(
            0, self._emitted_samples_at_core_rate - self._dropped_samples_at_core_rate
        )
        emitted_ms = (active_samples * 1000) // self._target_out_rate
        heard_ms = emitted_ms - self.output_queue_ms() - max(0, downstream_ms)
        return max(0, heard_ms)

    def reset_playout_tracking(self) -> None:
        """Reset cumulative playout sample counters for a new turn."""
        self._emitted_samples_at_core_rate = 0
        self._dropped_samples_at_core_rate = 0

    def close(self) -> None:
        """Close the transport and cancel any active drain task."""
        self._closed = True
        self.flush_output()
