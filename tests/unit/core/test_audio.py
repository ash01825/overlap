"""Unit tests for audio primitives in overlap.core.audio.

Covers PCM utils, click-free GainRamp, LoudnessTracker, EchoGate, and EnergyVad.
Enforces INV-L1, INV-L3, INV-L8, INV-L12, and VadContract.
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from overlap.core.audio import (
    EchoGate,
    EnergyVad,
    GainRamp,
    LoudnessTracker,
    float32_to_pcm16,
    mix_pcm16,
    pcm16_to_float32,
    rms_dbfs,
)
from overlap.core.types import AudioFrame, Tag, Tuning

_contracts_dir = str(Path(__file__).resolve().parent.parent.parent / "contracts")
if _contracts_dir not in sys.path:
    sys.path.insert(0, _contracts_dir)

from suites import VadContract  # noqa: E402

# =============================================================================
# PCM conversion & math tests
# =============================================================================


def test_pcm16_to_float32_and_back_roundtrip() -> None:
    # Silence
    zero_pcm = b"\x00\x00" * 320
    floats = pcm16_to_float32(zero_pcm)
    assert len(floats) == 320
    assert np.all(floats == 0.0)
    assert float32_to_pcm16(floats) == zero_pcm

    # Extremes: 32767 and -32768
    samples = np.array([32767, -32768, 0, 16384, -16384], dtype=np.int16)
    pcm = samples.tobytes()
    floats = pcm16_to_float32(pcm)

    assert pytest.approx(floats[0], abs=1e-4) == 32767 / 32768.0
    assert pytest.approx(floats[1], abs=1e-4) == -1.0
    assert pytest.approx(floats[2], abs=1e-4) == 0.0
    assert pytest.approx(floats[3], abs=1e-4) == 0.5
    assert pytest.approx(floats[4], abs=1e-4) == -0.5

    reconstructed_pcm = float32_to_pcm16(floats)
    reconstructed_samples = np.frombuffer(reconstructed_pcm, dtype=np.int16)
    np.testing.assert_array_equal(reconstructed_samples, samples)


def test_float32_to_pcm16_saturating_extremes() -> None:
    # Overflow > 1.0 and underflow < -1.0 must saturate without wraparound
    floats = np.array([1.5, 100.0, -1.5, -100.0, 1.0, -1.0], dtype=np.float32)
    pcm = float32_to_pcm16(floats)
    samples = np.frombuffer(pcm, dtype=np.int16)

    assert samples[0] == 32767
    assert samples[1] == 32767
    assert samples[2] == -32768
    assert samples[3] == -32768
    assert samples[4] == 32767
    assert samples[5] == -32768


def test_rms_dbfs_calculations_and_silence_floor() -> None:
    # Silence -> floor at -90.0 dBFS
    zero_pcm = b"\x00\x00" * 320
    assert rms_dbfs(zero_pcm) == -90.0
    assert rms_dbfs(np.zeros(320, dtype=np.float32)) == -90.0

    # Full-scale square wave (+1.0, -1.0) -> RMS is 1.0 -> 0.0 dBFS
    full_scale = np.array([1.0, -1.0] * 160, dtype=np.float32)
    assert pytest.approx(rms_dbfs(full_scale), abs=0.01) == 0.0

    # Full-scale sine wave -> RMS is 1 / sqrt(2) ≈ 0.7071 -> 20 * log10(1/sqrt(2)) ≈ -3.01 dBFS
    t = np.linspace(0, 2 * np.pi, 320, endpoint=False)
    sine = np.sin(t).astype(np.float32)
    assert pytest.approx(rms_dbfs(sine), abs=0.1) == -3.01

    # Extremely low non-zero audio below -90 dBFS must be clamped to -90.0
    very_quiet = np.full(320, 1e-6, dtype=np.float32)
    assert rms_dbfs(very_quiet) == -90.0


def test_mix_pcm16_saturating_add_extremes() -> None:
    # Positive overflow: 30000 + 30000 = 60000 -> saturate at 32767
    pcm1 = struct.pack("<h", 30000) * 10
    pcm2 = struct.pack("<h", 30000) * 10
    mixed = mix_pcm16(pcm1, pcm2)
    samples = np.frombuffer(mixed, dtype=np.int16)
    assert np.all(samples == 32767)

    # Negative underflow: -30000 + -30000 = -60000 -> saturate at -32768
    pcm_neg1 = struct.pack("<h", -30000) * 10
    pcm_neg2 = struct.pack("<h", -30000) * 10
    mixed_neg = mix_pcm16(pcm_neg1, pcm_neg2)
    samples_neg = np.frombuffer(mixed_neg, dtype=np.int16)
    assert np.all(samples_neg == -32768)

    # Different lengths
    short_pcm = struct.pack("<h", 1000) * 5
    long_pcm = struct.pack("<h", 2000) * 10
    mixed_diff = mix_pcm16(short_pcm, long_pcm)
    samples_diff = np.frombuffer(mixed_diff, dtype=np.int16)
    assert len(samples_diff) == 10
    assert np.all(samples_diff[:5] == 3000)
    assert np.all(samples_diff[5:] == 2000)


# =============================================================================
# GainRamp tests (INV-L3, INV-L8, click-free continuity)
# =============================================================================


def test_gain_ramp_linear_per_sample_and_attack_within_one_frame() -> None:
    tuning = Tuning()
    # Attack to duck_gain (0.3) within duck_attack_ms (20 ms <= 1 frame) per INV-L3
    ramp = GainRamp(
        current=1.0,
        target=tuning.duck_gain,
        ramp_ms=tuning.duck_attack_ms,
        sample_rate=tuning.out_rate_hz,
    )

    frame_samples = int(tuning.out_rate_hz * (tuning.frame_ms / 1000.0))  # 320 samples
    dc_pcm = struct.pack("<h", 10000) * frame_samples

    ducked_pcm = ramp.apply(dc_pcm)
    ducked_samples = np.frombuffer(ducked_pcm, dtype=np.int16)

    # Ramp must complete within the first frame
    assert ramp.is_finished
    assert pytest.approx(ramp.current_gain, abs=1e-5) == tuning.duck_gain
    # Last sample must have reached target gain (10000 * 0.3 = 3000)
    assert pytest.approx(ducked_samples[-1], abs=5) == 3000


def test_gain_ramp_release_over_duck_release_ms() -> None:
    tuning = Tuning()
    # Release over duck_release_ms (150 ms) to 1.0 per INV-L8
    ramp = GainRamp(
        current=tuning.duck_gain,
        target=1.0,
        ramp_ms=tuning.duck_release_ms,
        sample_rate=tuning.out_rate_hz,
    )

    frame_samples = int(tuning.out_rate_hz * (tuning.frame_ms / 1000.0))  # 320 samples
    dc_pcm = struct.pack("<h", 10000) * frame_samples

    # 150 ms / 20 ms = 7.5 frames -> should finish on 8th frame
    for _ in range(7):
        assert not ramp.is_finished
        ramp.apply(dc_pcm)

    # 8th frame
    ramp.apply(dc_pcm)
    assert ramp.is_finished
    assert pytest.approx(ramp.current_gain, abs=1e-5) == 1.0


def test_gain_ramp_click_free_max_delta_sample_bound() -> None:
    tuning = Tuning()
    # Continuous DC audio across 5 frames with ducking attack then release
    frame_samples = int(tuning.out_rate_hz * (tuning.frame_ms / 1000.0))
    dc_val = 20000
    pcm_frame = struct.pack("<h", dc_val) * frame_samples

    ramp = GainRamp(current=1.0, target=1.0, ramp_ms=0, sample_rate=tuning.out_rate_hz)
    ramp.set_target(tuning.duck_gain, tuning.duck_attack_ms)

    processed_frames: list[np.ndarray] = []
    for _ in range(3):
        out_bytes = ramp.apply(pcm_frame)
        processed_frames.append(np.frombuffer(out_bytes, dtype=np.int16))

    all_samples = np.concatenate(processed_frames)
    deltas = np.abs(np.diff(all_samples))

    # Max delta between any two consecutive samples (including across frame boundaries)
    # Step per sample is 0.7 / 320 ≈ 0.0021875. For 20000 DC, step is ~43.75.
    # Discontinuity click bound: no delta may exceed 100
    assert np.max(deltas) <= 100


# =============================================================================
# LoudnessTracker tests
# =============================================================================


def test_loudness_tracker_ema_and_silent_reference() -> None:
    tracker = LoudnessTracker()

    # Defined behavior when reference is completely silent (no divide by zero, no NaN/-inf)
    user_silence_rms = 0.0
    rel_db = tracker.rel_loudness_db(user_silence_rms)
    assert not math.isnan(rel_db)
    assert not math.isinf(rel_db)
    assert rel_db == 0.0

    # User speaking with silent agent reference -> defined positive finite relative loudness
    user_speaking_rms = 0.1  # approx -20 dBFS
    rel_speaking_db = tracker.rel_loudness_db(user_speaking_rms)
    assert not math.isnan(rel_speaking_db)
    assert not math.isinf(rel_speaking_db)
    assert rel_speaking_db > 0.0

    # Agent starts speaking
    for _ in range(20):
        tracker.update_agent(0.1)  # agent RMS is 0.1 (-20 dBFS)

    # When user RMS equals agent RMS, rel_loudness_db is approx 0.0 dB
    rel_equal_db = tracker.rel_loudness_db(0.1)
    assert pytest.approx(rel_equal_db, abs=0.5) == 0.0

    # When user is 10 dB louder (RMS ~0.316), rel_loudness_db is approx +10 dB
    rel_louder_db = tracker.rel_loudness_db(0.316)
    assert pytest.approx(rel_louder_db, abs=0.5) == 10.0


# =============================================================================
# EchoGate tests
# =============================================================================


def test_echo_gate_pure_echo_suppressed() -> None:
    gate = EchoGate()
    ref_rms = 0.2
    # Pure echo: mic sees the acoustic echo return of ref audio
    # Cold start conservative ERL ≈ 0.25 (-12 dB return loss). Echo RMS ≈ 0.2 * 0.25 = 0.05
    pure_echo_mic_rms = 0.04
    assert gate.is_echo(pure_echo_mic_rms, ref_rms=ref_rms)


def test_echo_gate_double_talk_not_suppressed() -> None:
    gate = EchoGate()
    ref_rms = 0.2
    # User speech 10 dB louder than the echo signal
    # Echo level ≈ 0.05; +10 dB louder is ≈ 0.05 * 3.16 ≈ 0.158
    double_talk_mic_rms = 0.16
    assert not gate.is_echo(double_talk_mic_rms, ref_rms=ref_rms)


def test_echo_gate_cold_start_conservative() -> None:
    gate = EchoGate()
    # At cold start with no agent speech (ref_rms = 0), user speech is NEVER classified as echo
    user_speech_rms = 0.1
    assert not gate.is_echo(user_speech_rms, ref_rms=0.0)


def test_echo_gate_self_audio_window_suppression_and_expiry() -> None:
    gate = EchoGate()
    start_ns = 1_000_000_000  # 1.0 s
    end_ns = 1_200_000_000  # 1.2 s (200 ms clip)
    gate.register_self_audio_window(start_ns, end_ns, tag=Tag.SELF_BACKCHANNEL)

    # During the clip (at 1.1 s) -> suppressed
    assert gate.is_echo(mic=0.05, ref_rms=0.0, now_ns=1_100_000_000)

    # In hangover window (+100 ms after clip ends, at 1.3 s < 1.35 s) -> still suppressed
    assert gate.is_echo(mic=0.05, ref_rms=0.0, now_ns=1_300_000_000)

    # Expired (+160 ms after clip ends, at 1.36 s > 1.35 s) -> NOT suppressed
    assert not gate.is_echo(mic=0.05, ref_rms=0.0, now_ns=1_360_000_000)


# =============================================================================
# EnergyVad tests (INV-L1, hangover, VadContract)
# =============================================================================


def test_energy_vad_onset_on_one_frame() -> None:
    vad = EnergyVad()

    # Adapt noise floor on silence
    for i in range(5):
        res = vad.process(AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=i * 20_000_000))
        assert not res.is_speech

    # Loud speech frame: onset must occur on the very first frame per INV-L1
    speech_pcm = struct.pack("<h", 15000) * 320
    speech_frame = AudioFrame(pcm=speech_pcm, sample_rate=16000, ts_ns=100_000_000)
    res_speech = vad.process(speech_frame)
    assert res_speech.is_speech
    assert res_speech.confidence >= 0.5


def test_energy_vad_hangover_prevents_flapping_and_offsets() -> None:
    hangover_frames = 3
    vad = EnergyVad(hangover_frames=hangover_frames)

    speech_pcm = struct.pack("<h", 15000) * 320
    silent_pcm = b"\x00" * 640

    # 1. Speech frame
    res1 = vad.process(AudioFrame(pcm=speech_pcm, sample_rate=16000, ts_ns=0))
    assert res1.is_speech

    # 2. Single silent frame: hangover keeps is_speech=True to prevent flapping
    res_gap = vad.process(AudioFrame(pcm=silent_pcm, sample_rate=16000, ts_ns=20_000_000))
    assert res_gap.is_speech

    # 3. Continued silence: hangover counts down
    res_gap2 = vad.process(AudioFrame(pcm=silent_pcm, sample_rate=16000, ts_ns=40_000_000))
    assert res_gap2.is_speech
    res_gap3 = vad.process(AudioFrame(pcm=silent_pcm, sample_rate=16000, ts_ns=60_000_000))
    assert res_gap3.is_speech

    # 4. Hangover expired: offset fires
    res_offset = vad.process(AudioFrame(pcm=silent_pcm, sample_rate=16000, ts_ns=80_000_000))
    assert not res_offset.is_speech


def test_energy_vad_passes_vad_contract() -> None:
    vad = EnergyVad()
    VadContract.verify(vad)
