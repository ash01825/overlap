# ruff: noqa: E402
"""Unit tests for seeded synthetic audio generators."""

from __future__ import annotations

import sys
from pathlib import Path

_fixtures_dir = str(Path(__file__).resolve().parents[2] / "fixtures")
if _fixtures_dir not in sys.path:
    sys.path.insert(0, _fixtures_dir)

import pytest
from audio import (
    assert_audio_properties,
    measure_dbfs,
    noise,
    silence,
    speechlike,
    tone,
)


def test_measure_dbfs_silence() -> None:
    pcm_zeros = b"\x00" * 3200  # 100ms at 16kHz
    dbfs = measure_dbfs(pcm_zeros)
    assert dbfs <= -96.0


def test_silence_exact_samples_and_level() -> None:
    dur_ms = 100
    pcm = silence(dur_ms=dur_ms, dbfs=-96.0, seed=42)
    # 100 ms at 16kHz = 1600 samples = 3200 bytes
    assert len(pcm) == 3200
    assert_audio_properties(pcm, dur_ms=dur_ms, expected_dbfs=-96.0)


def test_tone_exact_samples_and_dbfs() -> None:
    for target_dbfs in [-6.0, -12.0, -20.0, -30.0]:
        dur_ms = 250
        pcm = tone(dur_ms=dur_ms, dbfs=target_dbfs, freq_hz=440.0, seed=123)
        # 250 ms at 16kHz = 4000 samples = 8000 bytes
        assert len(pcm) == 8000
        measured = measure_dbfs(pcm)
        assert abs(measured - target_dbfs) <= 0.5, (
            f"Expected {target_dbfs} dBFS, got {measured:.2f} dBFS"
        )
        assert_audio_properties(pcm, dur_ms=dur_ms, expected_dbfs=target_dbfs, tolerance_db=0.5)


def test_noise_exact_samples_and_dbfs() -> None:
    for target_dbfs in [-10.0, -20.0, -26.0]:
        dur_ms = 300
        pcm = noise(dur_ms=dur_ms, dbfs=target_dbfs, seed=999)
        # 300 ms at 16kHz = 4800 samples = 9600 bytes
        assert len(pcm) == 9600
        measured = measure_dbfs(pcm)
        assert abs(measured - target_dbfs) <= 0.5, (
            f"Expected {target_dbfs} dBFS, got {measured:.2f} dBFS"
        )
        assert_audio_properties(pcm, dur_ms=dur_ms, expected_dbfs=target_dbfs, tolerance_db=0.5)


def test_speechlike_exact_samples_and_dbfs() -> None:
    for target_dbfs in [-12.0, -18.0, -24.0]:
        dur_ms = 400
        pcm = speechlike(dur_ms=dur_ms, dbfs=target_dbfs, seed=777)
        # 400 ms at 16kHz = 6400 samples = 12800 bytes
        assert len(pcm) == 12800
        measured = measure_dbfs(pcm)
        assert abs(measured - target_dbfs) <= 0.5, (
            f"Expected {target_dbfs} dBFS, got {measured:.2f} dBFS"
        )
        assert_audio_properties(pcm, dur_ms=dur_ms, expected_dbfs=target_dbfs, tolerance_db=0.5)


def test_seeded_determinism() -> None:
    pcm1 = noise(dur_ms=100, dbfs=-20.0, seed=42)
    pcm2 = noise(dur_ms=100, dbfs=-20.0, seed=42)
    assert pcm1 == pcm2

    pcm3 = speechlike(dur_ms=100, dbfs=-15.0, seed=100)
    pcm4 = speechlike(dur_ms=100, dbfs=-15.0, seed=100)
    assert pcm3 == pcm4


def test_assert_audio_properties_fails_on_mismatch() -> None:
    pcm = tone(dur_ms=100, dbfs=-10.0, seed=1)
    with pytest.raises(AssertionError, match="dBFS mismatch"):
        assert_audio_properties(pcm, dur_ms=100, expected_dbfs=-30.0, tolerance_db=0.5)

    with pytest.raises(AssertionError, match="Sample count mismatch"):
        assert_audio_properties(pcm, dur_ms=200, expected_dbfs=-10.0)
