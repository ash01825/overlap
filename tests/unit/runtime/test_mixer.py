# ruff: noqa: E402
"""Unit tests for PlayoutMixer (T2.01).

Tests:
- Re-framing of variable-length AudioChunks into exact 20 ms frames.
- Sample-exact re-chunking across chunk boundaries (no dropped/duplicated samples).
- Clock-driven frame emission cadence.
- Synchronous SetGain, click-free duck attack (INV-L3) and 150 ms release ramp (INV-L8).
- Property test: Look-ahead never exceeds Tuning.max_lookahead_ms (INV-L9).
- Clip injection tagged SELF_BACKCHANNEL, EchoGate registration, rejection if TTS active.
- Synchronous flush dropping un-emitted audio, calling transport.flush_output,
  returning ledger snapshot, and ensuring zero post-flush leakage (INV-L5, INV-L7).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

_fakes_dir = str(Path(__file__).resolve().parents[2] / "fakes")
if _fakes_dir not in sys.path:
    sys.path.insert(0, _fakes_dir)

from transport import FakeTransport

from overlap.core.audio import EchoGate, pcm16_to_float32
from overlap.core.clock import VirtualClock
from overlap.core.types import Tag, Tuning
from overlap.ports.tts import AudioChunk
from overlap.runtime.mixer import PlayoutMixer


def _generate_tone_pcm(
    num_samples: int,
    freq_hz: float = 440.0,
    amplitude: float = 10000.0,
    sample_rate: int = 16000,
) -> bytes:
    """Generate deterministic int16 mono sine wave PCM."""
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    wave = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
    int16_samples = np.clip(np.rint(wave), -32768, 32767).astype(np.int16)
    return bytes(int16_samples.tobytes())


def _generate_counter_pcm(num_samples: int, start: int = 0) -> bytes:
    """Generate sequential 16-bit integer PCM for exact sample-tracking."""
    samples = [(start + i) % 32767 for i in range(num_samples)]
    return struct.pack(f"<{num_samples}h", *samples)


class TestPlayoutMixerRechunking:
    """Tests for sample-exact re-framing of AudioChunks into 20 ms frames."""

    def test_sample_exact_rechunking(self) -> None:
        """Verify audio samples across irregular chunk boundaries are preserved byte-for-byte."""
        clock = VirtualClock()
        transport = FakeTransport(clock=clock)
        tuning = Tuning(frame_ms=20, out_rate_hz=16000)
        mixer = PlayoutMixer(transport=transport, clock=clock, tuning=tuning)

        total_samples = 320 * 5  # Exactly 5 frames (100 ms)
        raw_pcm = _generate_counter_pcm(total_samples)

        # Split into irregular chunks: 73 samples, 247 samples, 500 samples, 780 samples
        chunk_sizes = [73, 247, 500, 780]
        assert sum(chunk_sizes) == total_samples

        offset = 0
        for i, sz in enumerate(chunk_sizes):
            sub_pcm = raw_pcm[offset * 2 : (offset + sz) * 2]
            offset += sz
            is_last = i == len(chunk_sizes) - 1
            chunk = AudioChunk(
                pcm=sub_pcm,
                sample_rate=16000,
                epoch=1,
                is_last=is_last,
            )
            mixer.enqueue_chunk(chunk)

        # Advance virtual time through the 5 frames (100 ms)
        clock.advance(100)

        assert len(transport.sent_frames) == 5
        # Verify frame sizes
        for rec in transport.sent_frames:
            assert len(rec.frame.pcm) == 640  # 320 samples * 2 bytes
            assert rec.frame.sample_rate == 16000
            assert rec.frame.tags == Tag.SELF_TTS

        # Concatenate emitted frames and compare against raw PCM
        reconstructed = b"".join(rec.frame.pcm for rec in transport.sent_frames)
        assert reconstructed == raw_pcm

    def test_rechunking_zero_padding_on_last(self) -> None:
        """Verify incomplete trailing frame is zero-padded when is_last=True."""
        clock = VirtualClock()
        transport = FakeTransport(clock=clock)
        tuning = Tuning(frame_ms=20, out_rate_hz=16000)
        mixer = PlayoutMixer(transport=transport, clock=clock, tuning=tuning)

        # 400 samples = 1 full frame (320) + 80 samples remainder
        raw_pcm = _generate_counter_pcm(400)
        chunk = AudioChunk(
            pcm=raw_pcm,
            sample_rate=16000,
            epoch=1,
            is_last=True,
        )
        mixer.enqueue_chunk(chunk)
        clock.advance(40)

        assert len(transport.sent_frames) == 2
        f1 = transport.sent_frames[0].frame.pcm
        f2 = transport.sent_frames[1].frame.pcm
        assert len(f1) == 640
        assert len(f2) == 640

        # First frame matches first 320 samples
        assert f1 == raw_pcm[:640]
        # Second frame has 80 samples (160 bytes) + 240 samples of zero padding (480 bytes)
        assert f2[:160] == raw_pcm[640:800]
        assert f2[160:] == b"\x00" * 480


class TestPlayoutMixerCadenceAndLookahead:
    """Tests for clock-driven cadence and bounded look-ahead (INV-L9)."""

    def test_clock_driven_cadence(self) -> None:
        """Verify frames are emitted on an exact 20 ms clock cadence."""
        clock = VirtualClock()
        transport = FakeTransport(clock=clock)
        tuning = Tuning(frame_ms=20, out_rate_hz=16000, max_lookahead_ms=60)
        mixer = PlayoutMixer(
            transport=transport, clock=clock, tuning=tuning, target_lookahead_ms=20
        )

        # Enqueue 100 ms of audio (5 frames) at t = 0
        raw_pcm = _generate_tone_pcm(320 * 5)
        chunk = AudioChunk(pcm=raw_pcm, sample_rate=16000, epoch=1, is_last=True)
        mixer.enqueue_chunk(chunk)

        # At t = 0, first frame is emitted immediately to start playback
        assert len(transport.sent_frames) == 1
        assert transport.sent_frames[0].ts == 0

        # Advance 19 ms: no new frame yet
        clock.advance(19)
        assert len(transport.sent_frames) == 1

        # Advance to 20 ms: second frame emitted
        clock.advance(1)
        assert len(transport.sent_frames) == 2
        assert transport.sent_frames[1].ts == 20_000_000

        # Advance to 40 ms: third frame emitted
        clock.advance(20)
        assert len(transport.sent_frames) == 3
        assert transport.sent_frames[2].ts == 40_000_000

        # Advance to 60 ms: fourth frame emitted
        clock.advance(20)
        assert len(transport.sent_frames) == 4
        assert transport.sent_frames[3].ts == 60_000_000

        # Advance to 80 ms: fifth frame emitted
        clock.advance(20)
        assert len(transport.sent_frames) == 5
        assert transport.sent_frames[4].ts == 80_000_000

        # Advance past 100 ms: no more frames emitted (idle)
        clock.advance(100)
        assert len(transport.sent_frames) == 5

    @given(
        chunk_sample_counts=st.lists(
            st.integers(min_value=50, max_value=800), min_size=1, max_size=10
        ),
        step_ms_list=st.lists(st.integers(min_value=5, max_value=40), min_size=1, max_size=15),
    )
    @settings(max_examples=40, deadline=None)
    def test_property_lookahead_never_exceeds_max(
        self,
        chunk_sample_counts: list[int],
        step_ms_list: list[int],
    ) -> None:
        """Property test: lookahead_ms <= Tuning.max_lookahead_ms at every point."""
        clock = VirtualClock()
        transport = FakeTransport(clock=clock)
        tuning = Tuning(frame_ms=20, max_lookahead_ms=60)
        mixer = PlayoutMixer(transport=transport, clock=clock, tuning=tuning)

        for count in chunk_sample_counts:
            pcm = _generate_tone_pcm(count)
            mixer.enqueue_chunk(AudioChunk(pcm=pcm, sample_rate=16000, epoch=1))
            assert mixer.lookahead_ms <= tuning.max_lookahead_ms

        for step in step_ms_list:
            clock.advance(step)
            assert mixer.lookahead_ms <= tuning.max_lookahead_ms


class TestPlayoutMixerGainAndDucking:
    """Tests for GainRamp application, attack (INV-L3) and release (INV-L8)."""

    def test_set_gain_is_synchronous(self) -> None:
        """Verify set_gain executes synchronously inline without awaiting."""
        clock = VirtualClock()
        transport = FakeTransport(clock=clock)
        mixer = PlayoutMixer(transport=transport, clock=clock)

        assert mixer.current_gain == 1.0
        mixer.set_gain(0.3, ramp_ms=20)
        assert mixer.target_gain == 0.3

    def test_duck_attack_and_release_ramp(self) -> None:
        """Verify linear attack to 0.3 (INV-L3) and 150 ms release to 1.0 (INV-L8)."""
        clock = VirtualClock()
        transport = FakeTransport(clock=clock)
        tuning = Tuning(duck_gain=0.3, duck_attack_ms=20, duck_release_ms=150)
        mixer = PlayoutMixer(transport=transport, clock=clock, tuning=tuning)

        # 400 ms of audio (20 frames)
        raw_pcm = _generate_tone_pcm(320 * 20, amplitude=10000.0)
        chunk = AudioChunk(pcm=raw_pcm, sample_rate=16000, epoch=1, is_last=True)
        mixer.enqueue_chunk(chunk)

        # At t = 0: frame 0 emitted with gain 1.0
        f0 = transport.sent_frames[0].frame.pcm
        floats0 = pcm16_to_float32(f0)
        assert np.max(np.abs(floats0)) > 0.25

        # At t = 20: advance clock to emit frame 1 at unity gain
        clock.advance(20)
        assert len(transport.sent_frames) == 2

        # Apply ducking to 0.3 over 20 ms ramp
        mixer.set_gain(tuning.duck_gain, ramp_ms=tuning.duck_attack_ms)
        assert mixer.target_gain == 0.3

        # Advance to t = 40: frame 2 is emitted with attack ramp from 1.0 down to 0.3
        clock.advance(20)
        assert len(transport.sent_frames) == 3
        # End of frame 2 reaches duck_gain (0.3)
        assert np.isclose(mixer.current_gain, 0.3, atol=1e-3)

        # Advance to t = 60: frame 3 should be completely at ducked gain 0.3
        clock.advance(20)
        assert len(transport.sent_frames) == 4
        f3 = transport.sent_frames[3].frame.pcm
        floats3 = pcm16_to_float32(f3)
        ratio = np.max(np.abs(floats3)) / np.max(np.abs(floats0))
        assert np.isclose(ratio, 0.3, atol=0.05)

        # Release gain over 150 ms (INV-L8)
        mixer.release_gain()
        assert mixer.target_gain == 1.0

        # Advance 160 ms (8 frames: covers 150 ms ramp)
        clock.advance(160)
        assert np.isclose(mixer.current_gain, 1.0, atol=1e-3)

    def test_click_free_transition(self) -> None:
        """Verify bounded per-sample delta during duck attack and release."""
        clock = VirtualClock()
        transport = FakeTransport(clock=clock)
        tuning = Tuning(duck_gain=0.3, duck_attack_ms=20, duck_release_ms=150)
        mixer = PlayoutMixer(transport=transport, clock=clock, tuning=tuning)

        raw_pcm = _generate_tone_pcm(320 * 15, amplitude=25000.0)
        mixer.enqueue_chunk(AudioChunk(pcm=raw_pcm, sample_rate=16000, epoch=1, is_last=True))

        clock.advance(20)
        mixer.duck()
        clock.advance(40)
        mixer.release_gain()
        clock.advance(160)

        # Concatenate all emitted samples
        all_samples = np.concatenate(
            [pcm16_to_float32(rec.frame.pcm) for rec in transport.sent_frames]
        )
        diffs = np.abs(np.diff(all_samples))
        # Ensure no harsh discontinuity/clicks
        assert np.max(diffs) < 0.35


class TestPlayoutMixerClipInjection:
    """Tests for backchannel clip injection, Tag.SELF_BACKCHANNEL, and EchoGate."""

    def test_clip_injection_tagged_self_backchannel(self) -> None:
        """Verify clip frames carry Tag.SELF_BACKCHANNEL."""
        clock = VirtualClock()
        transport = FakeTransport(clock=clock)
        echo_gate = EchoGate()
        mixer = PlayoutMixer(transport=transport, clock=clock, echo_gate=echo_gate)

        # Inject 40 ms clip (2 frames)
        clip_pcm = _generate_tone_pcm(640)
        mixer.play_clip(clip_pcm, tag=Tag.SELF_BACKCHANNEL)

        assert mixer.is_clip_active
        clock.advance(40)

        assert len(transport.sent_frames) == 2
        for rec in transport.sent_frames:
            assert rec.frame.tags == Tag.SELF_BACKCHANNEL

    def test_clip_registers_echo_gate_window(self) -> None:
        """Verify clip injection registers self-audio window in EchoGate."""
        clock = VirtualClock(start_ns=1_000_000_000)
        transport = FakeTransport(clock=clock)
        echo_gate = EchoGate()
        mixer = PlayoutMixer(transport=transport, clock=clock, echo_gate=echo_gate)

        clip_pcm = _generate_tone_pcm(320)  # 20 ms clip
        mixer.play_clip(clip_pcm, tag=Tag.SELF_BACKCHANNEL)

        # Window covers [start, start + 20ms + 150ms expiry]
        now_ns = 1_000_000_000
        assert echo_gate.is_in_self_audio_window(now_ns + 10_000_000)  # during clip
        assert echo_gate.is_in_self_audio_window(now_ns + 100_000_000)  # within +150ms hangover
        assert not echo_gate.is_in_self_audio_window(now_ns + 200_000_000)  # expired

    def test_clip_rejected_when_tts_playing(self) -> None:
        """Verify play_clip raises RuntimeError when agent TTS is active (failure mode 4)."""
        clock = VirtualClock()
        transport = FakeTransport(clock=clock)
        mixer = PlayoutMixer(transport=transport, clock=clock)

        mixer.enqueue_chunk(AudioChunk(pcm=_generate_tone_pcm(640), sample_rate=16000, epoch=1))
        assert mixer.is_tts_active

        with pytest.raises(RuntimeError, match="Cannot inject clip while TTS audio is playing"):
            mixer.play_clip(_generate_tone_pcm(320))

    def test_clip_untagged_rejected(self) -> None:
        """Verify play_clip rejects untagged or Tag.NONE clips."""
        clock = VirtualClock()
        transport = FakeTransport(clock=clock)
        mixer = PlayoutMixer(transport=transport, clock=clock)

        with pytest.raises(ValueError, match="Tag.SELF_BACKCHANNEL"):
            mixer.play_clip(_generate_tone_pcm(320), tag=Tag.NONE)


class TestPlayoutMixerFlushAndHygiene:
    """Tests for flush, zero post-flush leakage (INV-L5, INV-L7), and ledger tracking."""

    def test_flush_drops_mixer_queue_and_calls_transport_flush(self) -> None:
        """Verify flush drops queue, calls transport.flush_output, and returns snapshot."""
        clock = VirtualClock()
        transport = FakeTransport(clock=clock)
        tuning = Tuning(assumed_downstream_ms=120)
        mixer = PlayoutMixer(transport=transport, clock=clock, tuning=tuning)
        mixer.start_turn("turn_1", epoch=1)

        # Queue 200 ms of audio (10 frames)
        audio = _generate_tone_pcm(320 * 10)
        mixer.enqueue_chunk(AudioChunk(pcm=audio, sample_rate=16000, epoch=1))

        # Advance 40 ms (emits frames at t=0, 20)
        clock.advance(40)
        frames_before = len(transport.sent_frames)
        assert frames_before >= 2

        # Flush
        snapshot = mixer.flush()
        assert transport.flushes == 1
        assert mixer.lookahead_ms == 0
        assert not mixer.is_playing
        assert snapshot.dropped_samples > 0
        assert snapshot.dropped_ms > 0

        # Advance clock further: zero frames emitted after flush (INV-L7)
        clock.advance(100)
        assert len(transport.sent_frames) == frames_before

    def test_stale_epoch_chunks_dropped_after_flush(self) -> None:
        """Verify chunks arriving from an old/flushed epoch are dropped (INV-L7)."""
        clock = VirtualClock()
        transport = FakeTransport(clock=clock)
        mixer = PlayoutMixer(transport=transport, clock=clock)
        mixer.start_turn("turn_1", epoch=1)

        mixer.enqueue_chunk(AudioChunk(pcm=_generate_tone_pcm(640), sample_rate=16000, epoch=1))
        mixer.flush()
        count_at_flush = len(transport.sent_frames)

        # Enqueue another chunk with flushed epoch 1
        mixer.enqueue_chunk(AudioChunk(pcm=_generate_tone_pcm(640), sample_rate=16000, epoch=1))
        clock.advance(60)

        # Zero new frames sent
        assert len(transport.sent_frames) == count_at_flush
