"""Microbenchmark for fast-path audio primitives (INV-L12).

Measures per-frame execution time of EnergyVad + EchoGate + GainRamp.
Must satisfy p99 <= 2 ms per INV-L12.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from overlap.core.audio import EchoGate, EnergyVad, GainRamp
from overlap.core.types import AudioFrame, Tuning


@pytest.mark.bench
@pytest.mark.slow
def test_audio_fast_path_p99_latency() -> None:
    tuning = Tuning()
    vad = EnergyVad()
    gate = EchoGate()
    ramp = GainRamp(
        current=1.0,
        target=tuning.duck_gain,
        ramp_ms=tuning.duck_attack_ms,
        sample_rate=tuning.out_rate_hz,
    )

    frame_samples = int(tuning.out_rate_hz * (tuning.frame_ms / 1000.0))  # 320
    sine_t = np.linspace(0, 2 * np.pi, frame_samples, endpoint=False)
    sine_pcm = (np.sin(sine_t) * 15000).astype(np.int16).tobytes()

    num_iterations = 1000
    latencies_ms: list[float] = []

    # Warmup
    for i in range(50):
        frame = AudioFrame(pcm=sine_pcm, sample_rate=tuning.in_rate_hz, ts_ns=i * 20_000_000)
        vad.process(frame)
        gate.is_echo(frame.pcm, ref_rms=0.1)
        ramp.apply(frame.pcm)

    for i in range(num_iterations):
        frame = AudioFrame(pcm=sine_pcm, sample_rate=tuning.in_rate_hz, ts_ns=i * 20_000_000)
        t0 = time.perf_counter_ns()

        vad.process(frame)
        gate.is_echo(frame.pcm, ref_rms=0.1)
        ramp.apply(frame.pcm)

        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1_000_000.0)

    p99_ms = float(np.percentile(latencies_ms, 99))
    print(
        f"\n[BENCH] Fast-path per-frame latency (INV-L12): "
        f"p99 <= 2 ms printed (actual: {p99_ms:.4f} ms)"
    )

    # Assert INV-L12
    assert p99_ms <= 2.0, f"Per-frame cost p99 exceeded 2 ms: {p99_ms:.4f} ms"
