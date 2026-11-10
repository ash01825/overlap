# ruff: noqa: E402
"""Invariant test for ducking latency (INV-L2).

ARCHITECTURE §10:
INV-L2: Speech onset -> gain-ducked frame leaves mixer <= 40 ms
(<= 1 frame detect + <= 1 frame apply) at all phase offsets within the frame.

Enforced in virtual time using VirtualClock, EnergyVad, and PlayoutMixer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_fakes_dir = str(Path(__file__).resolve().parents[1] / "fakes")
if _fakes_dir not in sys.path:
    sys.path.insert(0, _fakes_dir)

from transport import FakeTransport

from overlap.core.audio import EnergyVad, pcm16_to_float32
from overlap.core.clock import VirtualClock
from overlap.core.types import AudioFrame, Tag, Tuning
from overlap.ports.tts import AudioChunk
from overlap.runtime.mixer import PlayoutMixer


def _generate_tone_pcm(
    num_samples: int,
    freq_hz: float = 440.0,
    amplitude: float = 15000.0,
    sample_rate: int = 16000,
) -> bytes:
    """Generate int16 mono sine wave PCM."""
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    wave = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
    int16_samples = np.clip(np.rint(wave), -32768, 32767).astype(np.int16)
    return bytes(int16_samples.tobytes())


def _generate_speech_frame(
    offset_ms: int,
    frame_ms: int = 20,
    sample_rate: int = 16000,
    speech_amp: float = 18000.0,
) -> bytes:
    """Generate a 20 ms frame where speech starts at offset_ms within the frame."""
    total_samples = int(sample_rate * (frame_ms / 1000.0))
    offset_samples = int(sample_rate * (offset_ms / 1000.0))

    silence = np.zeros(offset_samples, dtype=np.int16)
    speech_count = total_samples - offset_samples
    if speech_count > 0:
        t = np.arange(speech_count, dtype=np.float32) / sample_rate
        speech = (speech_amp * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.int16)
        samples = np.concatenate([silence, speech])
    else:
        samples = silence

    return bytes(samples.tobytes())


@pytest.mark.parametrize("offset_ms", [0, 2, 5, 8, 10, 12, 15, 18, 19])
def test_duck_latency_invariant_all_phase_offsets(
    offset_ms: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Speech onset at t0 -> first ducked frame leaves mixer <= 40 ms virtual (INV-L2)."""
    clock = VirtualClock(start_ns=0)
    transport = FakeTransport(clock=clock)
    tuning = Tuning()
    mixer = PlayoutMixer(transport=transport, clock=clock, tuning=tuning)
    mixer.start_turn("test_turn", epoch=1)

    vad = EnergyVad(speech_threshold_db=10.0, min_noise_floor_db=-60.0)

    # 1. Start agent TTS playout with continuous tone
    # 500 ms of agent audio
    agent_audio = _generate_tone_pcm(320 * 25, amplitude=12000.0)
    mixer.enqueue_chunk(AudioChunk(pcm=agent_audio, sample_rate=16000, epoch=1, is_last=True))

    # Let agent speak for 40 ms (2 frames: [0..20] and [20..40])
    clock.advance(20)
    clock.advance(20)
    base_t_ms = 40

    # Capture un-ducked reference amplitude from sent frames
    ref_records = [r for r in transport.sent_frames if r.rms > 0]
    assert len(ref_records) >= 2
    unducked_rms = ref_records[-1].rms

    # 2. User speech begins at t0 = base_t_ms + offset_ms
    t0_ms = base_t_ms + offset_ms
    t0_ns = int(t0_ms * 1_000_000)

    # The user audio frame spans [base_t_ms, base_t_ms + 20 ms]
    user_frame_pcm = _generate_speech_frame(offset_ms=offset_ms, speech_amp=18000.0)
    user_frame = AudioFrame(
        pcm=user_frame_pcm,
        sample_rate=16000,
        ts_ns=int(base_t_ms * 1_000_000),
        tags=Tag.NONE,
    )

    # At the end of the frame (t = base_t_ms + 20 ms), VAD processes the frame
    clock.advance(20)

    vad_result = vad.process(user_frame)
    assert vad_result.is_speech, f"VAD failed to detect speech onset at offset {offset_ms}ms"

    # Fast path: VAD onset -> EchoGate -> FSM -> SetGain inline (INV-L2, INV-L13)
    mixer.duck(gain=tuning.duck_gain, ramp_ms=tuning.duck_attack_ms)

    # Find the first ducked frame emitted to transport at or after detection
    # A frame is ducked when its RMS or samples are reduced by duck_gain ramp
    ducked_frame_found = False
    first_ducked_t_ns = 0

    for rec in transport.sent_frames:
        if rec.ts >= base_t_ms * 1_000_000:
            samples = pcm16_to_float32(rec.frame.pcm)
            ducked_rms = rec.rms < 0.85 * unducked_rms
            ducked_peak = np.max(np.abs(samples)) < 0.85 * (unducked_rms / 32768.0)
            if ducked_rms or ducked_peak:
                ducked_frame_found = True
                first_ducked_t_ns = rec.ts
                break

    if not ducked_frame_found:
        # Advance 20 ms to allow mixer tick to emit ducked frame if needed
        clock.advance(20)
        for rec in transport.sent_frames:
            if rec.ts >= base_t_ms * 1_000_000:
                if rec.rms < 0.85 * unducked_rms:
                    ducked_frame_found = True
                    first_ducked_t_ns = rec.ts
                    break

    assert ducked_frame_found, "No ducked frame was observed at transport"

    # Measured virtual latency from speech onset t0 to first ducked frame at transport
    measured_latency_ms = (first_ducked_t_ns - t0_ns) / 1_000_000.0

    # INV-L2: speech onset -> gain-ducked frame leaves mixer <= 40 ms
    assert measured_latency_ms <= tuning.duck_latency_cap_ms, (
        f"INV-L2 VIOLATION at offset {offset_ms}ms: "
        f"latency {measured_latency_ms:.2f} ms > {tuning.duck_latency_cap_ms} ms"
    )

    result_line = (
        f"INV-L2 PASSED: onset at t0={t0_ms}ms (offset {offset_ms}ms) -> "
        f"first ducked frame at {first_ducked_t_ns / 1_000_000:.1f}ms, "
        f"virtual latency = {measured_latency_ms:.1f} ms <= {tuning.duck_latency_cap_ms} ms"
    )
    with capsys.disabled():
        print(result_line)
