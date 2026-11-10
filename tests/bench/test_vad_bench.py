"""Microbenchmark for VAD per-frame processing latency (INV-L12).

Asserts per-frame inference cost p99 <= 2.0 ms on the dev machine.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from overlap.adapters.vad_silero import SileroVad
from overlap.core.audio import float32_to_pcm16
from overlap.core.types import AudioFrame


class FastStubSession:
    """Fast stub session returning constant speech prob with mock v4 outputs."""

    def __init__(self) -> None:
        self._out_prob = np.array([[0.8]], dtype=np.float32)
        self._hn = np.zeros((2, 1, 64), dtype=np.float32)
        self._cn = np.zeros((2, 1, 64), dtype=np.float32)

    def get_inputs(self) -> list[Any]:
        class Meta:
            def __init__(self, name: str) -> None:
                self.name = name

        return [Meta("input"), Meta("sr"), Meta("h"), Meta("c")]

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[np.ndarray]:
        return [self._out_prob, self._hn, self._cn]


@pytest.mark.bench
def test_silero_vad_per_frame_latency_p99() -> None:
    """Benchmark SileroVad per-frame processing time: p99 <= 2.0 ms (INV-L12)."""
    session = FastStubSession()
    vad = SileroVad(session=session)

    # Pre-generate 1000 frames of 20 ms synthetic audio at 16 kHz
    num_frames = 1000
    t = np.linspace(0, 0.02, 320, endpoint=False)
    synth_audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    pcm = float32_to_pcm16(synth_audio)
    frames = [
        AudioFrame(pcm=pcm, sample_rate=16000, ts_ns=i * 20_000_000) for i in range(num_frames)
    ]

    # Warmup
    for f in frames[:50]:
        vad.process(f)

    # Measured benchmark run
    durations_ms: list[float] = []
    for f in frames:
        t0 = time.perf_counter_ns()
        vad.process(f)
        t1 = time.perf_counter_ns()
        durations_ms.append((t1 - t0) / 1_000_000.0)

    p50 = float(np.percentile(durations_ms, 50))
    p95 = float(np.percentile(durations_ms, 95))
    p99 = float(np.percentile(durations_ms, 99))

    print(
        f"\nSileroVad benchmark (1000 frames): p50={p50:.3f} ms, p95={p95:.3f} ms, p99={p99:.3f} ms"
    )
    assert p99 <= 2.0, f"Per-frame latency p99 exceeded 2.0 ms: {p99:.3f} ms"


@pytest.mark.bench
def test_energy_vad_fallback_per_frame_latency_p99() -> None:
    """Benchmark EnergyVad fallback per-frame processing time: p99 <= 2.0 ms."""
    vad = SileroVad(session=None, model_path=None)
    assert vad.is_fallback is True

    num_frames = 1000
    t = np.linspace(0, 0.02, 320, endpoint=False)
    synth_audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    pcm = float32_to_pcm16(synth_audio)
    frames = [
        AudioFrame(pcm=pcm, sample_rate=16000, ts_ns=i * 20_000_000) for i in range(num_frames)
    ]

    # Warmup
    for f in frames[:50]:
        vad.process(f)

    durations_ms: list[float] = []
    for f in frames:
        t0 = time.perf_counter_ns()
        vad.process(f)
        t1 = time.perf_counter_ns()
        durations_ms.append((t1 - t0) / 1_000_000.0)

    p50 = float(np.percentile(durations_ms, 50))
    p95 = float(np.percentile(durations_ms, 95))
    p99 = float(np.percentile(durations_ms, 99))

    print(
        f"\nEnergyVad fallback benchmark (1000 frames): "
        f"p50={p50:.3f} ms, p95={p95:.3f} ms, p99={p99:.3f} ms"
    )
    assert p99 <= 2.0, f"Per-frame latency p99 exceeded 2.0 ms: {p99:.3f} ms"
