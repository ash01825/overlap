"""Unit and contract tests for LiveKitTransport against a faked LiveKit RTC layer."""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_helpers_dir = str(Path(__file__).resolve().parent.parent.parent / "helpers")
if _helpers_dir not in sys.path:
    sys.path.insert(0, _helpers_dir)

_contracts_dir = str(Path(__file__).resolve().parent.parent.parent / "contracts")
if _contracts_dir not in sys.path:
    sys.path.insert(0, _contracts_dir)

from suites import TransportContract  # noqa: E402
from virtual import run_virtual  # noqa: E402

from overlap.core.clock import VirtualClock  # noqa: E402
from overlap.core.types import AudioFrame, Tuning  # noqa: E402
from overlap.integrations.livekit_transport import (  # noqa: E402
    LiveKitTransport,
    SimpleRtcAudioFrame,
)

# =============================================================================
# Fake LiveKit RTC Layer (Hermetic, Zero Network)
# =============================================================================


@dataclass(frozen=True, slots=True)
class FakeRtcAudioFrameEvent:
    """Mimics `livekit.rtc.AudioFrameEvent` yielded by `livekit.rtc.AudioStream`."""

    frame: SimpleRtcAudioFrame


class FakeRtcAudioStream:
    """Fake `livekit.rtc.AudioStream` yielding queued RTC frames."""

    def __init__(self, events: list[FakeRtcAudioFrameEvent] | None = None) -> None:
        self._queue: deque[FakeRtcAudioFrameEvent] = deque(events or [])
        self._event = asyncio.Event()
        self._closed = False
        if self._queue:
            self._event.set()

    def push_frame(
        self,
        pcm: bytes,
        sample_rate: int = 48000,
        num_channels: int = 1,
    ) -> None:
        samples_per_ch = len(pcm) // (2 * num_channels)
        frame = SimpleRtcAudioFrame(
            data=pcm,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=samples_per_ch,
        )
        self._queue.append(FakeRtcAudioFrameEvent(frame=frame))
        self._event.set()

    def close(self) -> None:
        self._closed = True
        self._event.set()

    def __aiter__(self) -> AsyncIterator[FakeRtcAudioFrameEvent]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[FakeRtcAudioFrameEvent]:
        while True:
            if self._queue:
                yield self._queue.popleft()
                continue
            if self._closed:
                break
            self._event.clear()
            await self._event.wait()


class FakeRtcAudioSource:
    """Fake `livekit.rtc.AudioSource` tracking captured frames and queued duration."""

    def __init__(
        self,
        sample_rate: int = 48000,
        num_channels: int = 1,
        *,
        block_capture: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.captured_frames: list[Any] = []
        self._queued_frames: deque[Any] = deque()
        self.clear_queue_calls: int = 0
        self._block_capture = block_capture
        self._unblock_event = asyncio.Event()
        if not block_capture:
            self._unblock_event.set()

    @property
    def queued_duration(self) -> float:
        total_samples = int(sum(int(f.samples_per_channel) for f in self._queued_frames))
        return float(total_samples) / float(self.sample_rate)

    def clear_queue(self) -> None:
        self.clear_queue_calls += 1
        self._queued_frames.clear()

    async def capture_frame(self, frame: Any) -> None:
        await self._unblock_event.wait()
        self.captured_frames.append(frame)
        self._queued_frames.append(frame)

    def consume_played_ms(self, ms: int) -> None:
        """Simulate WebRTC playout draining `ms` milliseconds from the native queue."""
        samples_to_drain = (ms * self.sample_rate) // 1000
        drained = 0
        while self._queued_frames and drained < samples_to_drain:
            head = self._queued_frames[0]
            if drained + head.samples_per_channel <= samples_to_drain:
                drained += head.samples_per_channel
                self._queued_frames.popleft()
            else:
                break


# =============================================================================
# 1. Contract Suite Compliance
# =============================================================================


class TestLiveKitTransportContract(TransportContract):
    """Run the standard AudioTransport contract suite against LiveKitTransport."""

    @pytest.fixture
    def transport(self) -> LiveKitTransport:
        stream = FakeRtcAudioStream()
        source = FakeRtcAudioSource(sample_rate=48000, num_channels=1)
        clock = VirtualClock()
        return LiveKitTransport(stream, source, clock=clock)


def test_transport_contract_verify_direct() -> None:
    """Verify `TransportContract.verify` passes on `LiveKitTransport`."""
    stream = FakeRtcAudioStream()
    source = FakeRtcAudioSource(sample_rate=48000, num_channels=1)
    transport = LiveKitTransport(stream, source, clock=VirtualClock())
    TransportContract.verify(transport)


# =============================================================================
# 2. 10-Minute Virtual-Time Re-chunking & Zero-Drift Test
# =============================================================================


@pytest.mark.parametrize("input_frame_ms,in_rate_hz", [(10, 48000), (30, 48000), (10, 16000)])
def test_rechunking_to_20ms_no_drift_over_10_min_virtual_time(
    input_frame_ms: int, in_rate_hz: int
) -> None:
    """Feed 10 minutes of audio at 10 ms or 30 ms cadence and assert exact 20 ms output
    with zero sample or timestamp drift.
    """
    tuning = Tuning()
    total_duration_ms = 10 * 60 * 1000  # 600,000 ms (10 minutes)
    expected_out_frames = total_duration_ms // tuning.frame_ms  # 30,000 frames of 20 ms
    expected_bytes_per_frame = (tuning.in_rate_hz * tuning.frame_ms // 1000) * 2  # 640 bytes

    num_in_frames = total_duration_ms // input_frame_ms
    samples_per_in_frame = (in_rate_hz * input_frame_ms) // 1000
    in_pcm = (np.ones(samples_per_in_frame, dtype="<i2") * 1000).tobytes()

    events = [
        FakeRtcAudioFrameEvent(
            frame=SimpleRtcAudioFrame(
                data=in_pcm,
                sample_rate=in_rate_hz,
                num_channels=1,
                samples_per_channel=samples_per_in_frame,
            )
        )
        for _ in range(num_in_frames)
    ]

    stream = FakeRtcAudioStream(events)
    stream.close()
    clock = VirtualClock(start_ns=1_000_000_000)
    source = FakeRtcAudioSource(sample_rate=48000, num_channels=1)
    transport = LiveKitTransport(stream, source, tuning=tuning, clock=clock)

    async def _run() -> None:
        count = 0
        first_ts_ns: int | None = None
        last_ts_ns: int | None = None
        expected_step_ns = tuning.frame_ms * 1_000_000

        async for frame in transport.user_frames():
            assert frame.sample_rate == tuning.in_rate_hz
            assert len(frame.pcm) == expected_bytes_per_frame
            if first_ts_ns is None:
                first_ts_ns = frame.ts_ns
            else:
                assert last_ts_ns is not None
                assert frame.ts_ns - last_ts_ns == expected_step_ns
            last_ts_ns = frame.ts_ns
            count += 1

        assert count == expected_out_frames
        assert first_ts_ns == 1_000_000_000
        assert last_ts_ns == 1_000_000_000 + (expected_out_frames - 1) * expected_step_ns

    run_virtual(_run())


# =============================================================================
# 3. Flush Output & Queue Accounting Tests
# =============================================================================


def test_flush_output_clears_audio_source_queue_and_updates_output_queue_ms() -> None:
    """`flush_output` clears both local outbound buffer and AudioSource queue;
    `output_queue_ms` reflects queued duration before and after flush.
    """
    tuning = Tuning()
    stream = FakeRtcAudioStream()
    source = FakeRtcAudioSource(sample_rate=48000, num_channels=1)
    transport = LiveKitTransport(stream, source, tuning=tuning, clock=VirtualClock())

    async def _run() -> None:
        pcm_20ms = b"\x10\x00" * 320  # 320 samples at 16 kHz = 20 ms
        for i in range(5):
            transport.send_frame(
                AudioFrame(
                    pcm=pcm_20ms,
                    sample_rate=tuning.out_rate_hz,
                    ts_ns=i * 20_000_000,
                )
            )

        # Before event loop drains or after it drains into source, total queued is 100 ms
        assert transport.output_queue_ms() == 5 * tuning.frame_ms
        await asyncio.sleep(0)
        assert len(source.captured_frames) == 5
        assert transport.output_queue_ms() == 5 * tuning.frame_ms

        # Simulate 40 ms played out by WebRTC
        source.consume_played_ms(2 * tuning.frame_ms)
        assert transport.output_queue_ms() == 3 * tuning.frame_ms

        # Flush output must clear remaining 60 ms immediately
        transport.flush_output()
        assert source.clear_queue_calls == 1
        assert transport.output_queue_ms() == 0

    run_virtual(_run())


# =============================================================================
# 4. Backpressure / Non-Blocking Send Frame Failure Mode Test
# =============================================================================


def test_send_frame_does_not_block_or_allocate_unboundedly_under_backpressure() -> None:
    """When `AudioSource.capture_frame` is blocked under backpressure, `send_frame`
    never blocks and drops the oldest frames beyond `max_outbound_frames`.
    """
    tuning = Tuning()
    max_frames = 10
    stream = FakeRtcAudioStream()
    source = FakeRtcAudioSource(sample_rate=48000, num_channels=1, block_capture=True)
    transport = LiveKitTransport(
        stream,
        source,
        tuning=tuning,
        clock=VirtualClock(),
        max_outbound_frames=max_frames,
    )

    async def _run() -> None:
        pcm_20ms = b"\x01\x00" * 320
        for i in range(50):
            transport.send_frame(
                AudioFrame(
                    pcm=pcm_20ms,
                    sample_rate=tuning.out_rate_hz,
                    ts_ns=i * 20_000_000,
                )
            )

        # Queue is strictly bounded at max_frames (10 * 20 ms = 200 ms)
        assert transport.output_queue_ms() == max_frames * tuning.frame_ms
        assert transport.dropped_outbound_frames == 50 - max_frames

        # Flushing clears the bounded queue immediately
        transport.flush_output()
        assert transport.output_queue_ms() == 0

    run_virtual(_run())


# =============================================================================
# 5. Playout Position Feedback & Jitter Buffer Accounting
# =============================================================================


def test_playout_position_ms_returns_measured_estimate_or_none() -> None:
    """`playout_position_ms()` returns None when downstream delay is unknown,
    and returns `emitted_ms - output_queue_ms() - downstream_delay_ms` when available
    so jitter buffer latency does not cause unheard words to be counted as heard.
    """
    tuning = Tuning()
    stream = FakeRtcAudioStream()
    source = FakeRtcAudioSource(sample_rate=48000, num_channels=1)
    clock = VirtualClock()

    # 1. Without downstream delay telemetry -> returns None (use Tuning.assumed_downstream_ms)
    transport_no_probe = LiveKitTransport(stream, source, tuning=tuning, clock=clock)
    assert transport_no_probe.playout_position_ms() is None

    # 2. With downstream delay telemetry -> accounts for both source queue and jitter buffer
    measured_jitter_ms: int | None = 80
    transport_with_probe = LiveKitTransport(
        stream,
        source,
        tuning=tuning,
        clock=clock,
        downstream_playout_delay_ms_fn=lambda: measured_jitter_ms,
    )

    async def _run() -> None:
        nonlocal measured_jitter_ms
        pcm_20ms = b"\x05\x00" * 320
        # Emit 10 frames = 200 ms total
        for i in range(10):
            transport_with_probe.send_frame(
                AudioFrame(
                    pcm=pcm_20ms,
                    sample_rate=tuning.out_rate_hz,
                    ts_ns=i * 20_000_000,
                )
            )
        await asyncio.sleep(0)

        # All 200 ms still in AudioSource queue + 80 ms jitter buffer => 0 ms heard
        assert transport_with_probe.output_queue_ms() == 200
        assert transport_with_probe.playout_position_ms() == 0

        # Drain 160 ms from AudioSource queue (40 ms left in source queue)
        # Emitted 200 ms - 40 ms queued - 80 ms jitter buffer = 80 ms actually heard
        source.consume_played_ms(160)
        assert transport_with_probe.output_queue_ms() == 40
        assert transport_with_probe.playout_position_ms() == 80

        # If telemetry temporarily returns None, playout_position_ms() returns None
        measured_jitter_ms = None
        assert transport_with_probe.playout_position_ms() is None

    run_virtual(_run())


# =============================================================================
# 6. Consistent Resampling Between User Track (48 kHz Stereo) & Agent Track
# =============================================================================


def test_user_and_agent_tracks_resampled_consistently() -> None:
    """Verify inbound 48 kHz stereo user track is downmixed and resampled to 16 kHz mono,
    and outbound 16 kHz mono agent track is resampled to 48 kHz for LiveKit AudioSource
    while preserving signal frequency and amplitude.
    """
    tuning = Tuning()
    stream = FakeRtcAudioStream()
    source = FakeRtcAudioSource(sample_rate=48000, num_channels=1)
    transport = LiveKitTransport(stream, source, tuning=tuning, clock=VirtualClock())

    # Generate 20 ms of a 400 Hz sine wave at 48 kHz stereo (960 samples per channel)
    t_48k = np.arange(960, dtype=np.float64) / 48000.0
    sine_48k = np.round(10000.0 * np.sin(2.0 * np.pi * 400.0 * t_48k)).astype("<i2")
    stereo_48k = np.column_stack((sine_48k, sine_48k)).flatten().tobytes()

    stream.push_frame(stereo_48k, sample_rate=48000, num_channels=2)
    stream.close()

    async def _run() -> None:
        in_frames = [f async for f in transport.user_frames()]
        assert len(in_frames) == 1
        user_frame = in_frames[0]
        assert user_frame.sample_rate == tuning.in_rate_hz
        user_samples = np.frombuffer(user_frame.pcm, dtype="<i2")
        assert len(user_samples) == 320

        # Verify downmixed & decimated 16 kHz signal matches expected 400 Hz sine
        t_16k = np.arange(320, dtype=np.float64) / 16000.0
        expected_16k = 10000.0 * np.sin(2.0 * np.pi * 400.0 * t_16k)
        corr_in = float(np.corrcoef(user_samples.astype(np.float64), expected_16k)[0, 1])
        assert corr_in > 0.99

        # Now send that 16 kHz frame out to the 48 kHz AudioSource
        transport.send_frame(user_frame)
        await asyncio.sleep(0)

        assert len(source.captured_frames) == 1
        out_rtc_frame = source.captured_frames[0]
        assert out_rtc_frame.sample_rate == 48000
        assert out_rtc_frame.num_channels == 1
        assert out_rtc_frame.samples_per_channel == 960
        out_samples = np.frombuffer(out_rtc_frame.data, dtype="<i2")
        assert len(out_samples) == 960
        # Amplitude RMS should remain consistent within 5%
        in_rms = float(np.sqrt(np.mean(user_samples.astype(np.float64) ** 2)))
        out_rms = float(np.sqrt(np.mean(out_samples.astype(np.float64) ** 2)))
        assert abs(in_rms - out_rms) / in_rms < 0.05

    run_virtual(_run())
