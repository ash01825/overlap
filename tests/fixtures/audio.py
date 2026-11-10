"""Seeded audio synthesis fixtures with exact sample count and level verification."""

from __future__ import annotations

import math
import random
import struct

from overlap.core.types import AudioFrame, Tag


def measure_dbfs(pcm: bytes, sample_rate: int = 16000) -> float:
    """Measure dBFS of 16-bit linear PCM audio relative to full-scale square/sine."""
    num_samples = len(pcm) // 2
    if num_samples == 0:
        return -120.0
    samples = struct.unpack(f"<{num_samples}h", pcm)
    sum_sq = sum(s * s for s in samples)
    if sum_sq == 0:
        return -120.0
    rms = math.sqrt(sum_sq / num_samples)
    return 20.0 * math.log10(rms / 32768.0)


def silence(
    dur_ms: int | float,
    dbfs: float = -96.0,
    sample_rate: int = 16000,
    seed: int = 0,
) -> bytes:
    """Generate exact-duration silence in 16-bit PCM."""
    num_samples = int(round(sample_rate * (dur_ms / 1000.0)))
    if dbfs <= -96.0:
        return b"\x00" * (num_samples * 2)

    rng = random.Random(seed)
    target_rms = 32768.0 * (10.0 ** (dbfs / 20.0))
    raw = [rng.gauss(0.0, 1.0) for _ in range(num_samples)]
    raw_rms = math.sqrt(sum(x * x for x in raw) / num_samples) if num_samples > 0 else 1.0
    scale = target_rms / raw_rms if raw_rms > 0 else 0.0

    samples = [max(-32768, min(32767, int(round(x * scale)))) for x in raw]
    return struct.pack(f"<{num_samples}h", *samples)


def tone(
    dur_ms: int | float,
    dbfs: float = -20.0,
    freq_hz: float = 440.0,
    sample_rate: int = 16000,
    seed: int = 0,
) -> bytes:
    """Generate exact-duration sine tone in 16-bit PCM with target dBFS."""
    num_samples = int(round(sample_rate * (dur_ms / 1000.0)))
    target_rms = 32768.0 * (10.0 ** (dbfs / 20.0))
    peak_amplitude = math.sqrt(2.0) * target_rms

    raw = [
        peak_amplitude * math.sin(2.0 * math.pi * freq_hz * i / sample_rate)
        for i in range(num_samples)
    ]
    # Fine-tune scale to match target dBFS after integer rounding
    raw_rms = math.sqrt(sum(x * x for x in raw) / num_samples) if num_samples > 0 else 1.0
    scale = target_rms / raw_rms if raw_rms > 0 else 1.0

    samples = [max(-32768, min(32767, int(round(x * scale)))) for x in raw]
    return struct.pack(f"<{num_samples}h", *samples)


def noise(
    dur_ms: int | float,
    dbfs: float = -20.0,
    sample_rate: int = 16000,
    seed: int = 0,
) -> bytes:
    """Generate exact-duration Gaussian white noise in 16-bit PCM."""
    num_samples = int(round(sample_rate * (dur_ms / 1000.0)))
    rng = random.Random(seed)
    target_rms = 32768.0 * (10.0 ** (dbfs / 20.0))

    raw = [rng.gauss(0.0, 1.0) for _ in range(num_samples)]
    raw_rms = math.sqrt(sum(x * x for x in raw) / num_samples) if num_samples > 0 else 1.0
    scale = target_rms / raw_rms if raw_rms > 0 else 0.0

    samples = [max(-32768, min(32767, int(round(x * scale)))) for x in raw]
    return struct.pack(f"<{num_samples}h", *samples)


def speechlike(
    dur_ms: int | float,
    dbfs: float = -20.0,
    sample_rate: int = 16000,
    seed: int = 0,
) -> bytes:
    """Generate exact-duration speech-like harmonic burst with amplitude modulation."""
    num_samples = int(round(sample_rate * (dur_ms / 1000.0)))
    rng = random.Random(seed)
    f0 = 160.0 + rng.uniform(-10.0, 10.0)

    target_rms = 32768.0 * (10.0 ** (dbfs / 20.0))

    raw: list[float] = []
    for i in range(num_samples):
        t = i / sample_rate
        # 4 Hz syllabic modulation
        mod = 0.5 + 0.5 * math.cos(2.0 * math.pi * 4.0 * t)
        # Fundamental + formants
        signal = (
            math.sin(2.0 * math.pi * f0 * t)
            + 0.5 * math.sin(2.0 * math.pi * 2.0 * f0 * t)
            + 0.3 * math.sin(2.0 * math.pi * 4.0 * f0 * t)
            + 0.1 * (rng.random() - 0.5)
        )
        raw.append(signal * mod)

    raw_rms = math.sqrt(sum(x * x for x in raw) / num_samples) if num_samples > 0 else 1.0
    scale = target_rms / raw_rms if raw_rms > 0 else 0.0

    samples = [max(-32768, min(32767, int(round(x * scale)))) for x in raw]
    return struct.pack(f"<{num_samples}h", *samples)


def assert_audio_properties(
    pcm: bytes,
    dur_ms: int | float,
    expected_dbfs: float,
    sample_rate: int = 16000,
    tolerance_db: float = 0.5,
) -> None:
    """Assert exact sample count and measured dBFS within tolerance."""
    expected_samples = int(round(sample_rate * (dur_ms / 1000.0)))
    actual_samples = len(pcm) // 2
    assert actual_samples == expected_samples, (
        f"Sample count mismatch: expected {expected_samples} samples ({dur_ms} ms), "
        f"got {actual_samples} samples"
    )

    measured = measure_dbfs(pcm, sample_rate=sample_rate)
    if expected_dbfs <= -90.0:
        assert measured <= expected_dbfs, (
            f"Silence level too loud: expected <= {expected_dbfs} dBFS, got {measured:.2f} dBFS"
        )
    else:
        diff = abs(measured - expected_dbfs)
        assert diff <= tolerance_db, (
            f"dBFS mismatch: expected {expected_dbfs:.2f} ± {tolerance_db} dBFS, "
            f"got {measured:.2f} dBFS (diff {diff:.2f} dB)"
        )


def to_audio_frame(
    pcm: bytes,
    sample_rate: int = 16000,
    ts_ns: int = 0,
    tags: Tag = Tag.NONE,
) -> AudioFrame:
    """Convenience helper converting PCM bytes to an AudioFrame."""
    return AudioFrame(pcm=pcm, sample_rate=sample_rate, ts_ns=ts_ns, tags=tags)
