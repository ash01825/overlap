"""Benchmark for OverlapClassifier call cost (INV-L11).

Measures p99 call latency of classify() across diverse feature profiles.
Target: p99 call cost <= 0.2 ms (200 µs).
"""

from __future__ import annotations

import time

import pytest

from overlap.core.classifier import classify
from overlap.core.types import OverlapFeatures, Tuning


@pytest.mark.bench
def test_classifier_call_cost_p99() -> None:
    """Benchmark: classify() p99 latency must be <= 0.2 ms (INV-L11)."""
    tuning = Tuning()

    # Pre-generate a diverse batch of features representing typical call profiles
    samples: tuple[OverlapFeatures, ...] = (
        OverlapFeatures(
            overlap_ms=50, user_speaking=True, rel_loudness_db=0.0, partial_text="ruko"
        ),
        OverlapFeatures(
            overlap_ms=100, user_speaking=True, rel_loudness_db=0.0, partial_text="kya"
        ),
        OverlapFeatures(
            overlap_ms=220, user_speaking=True, rel_loudness_db=0.0, partial_text="haan"
        ),
        OverlapFeatures(
            overlap_ms=300, user_speaking=True, rel_loudness_db=0.0, partial_text="theek hai ji"
        ),
        OverlapFeatures(
            overlap_ms=100, user_speaking=True, rel_loudness_db=0.0, partial_text="flight ticket"
        ),
        OverlapFeatures(
            overlap_ms=260, user_speaking=False, rel_loudness_db=0.0, partial_text=None
        ),
        OverlapFeatures(overlap_ms=260, user_speaking=True, rel_loudness_db=4.5, partial_text=None),
        OverlapFeatures(
            overlap_ms=280,
            user_speaking=True,
            rel_loudness_db=-12.0,
            stt_conf=0.25,
            partial_text="ek glass paani",
        ),
        OverlapFeatures(overlap_ms=400, user_speaking=True, rel_loudness_db=0.0, partial_text=None),
        OverlapFeatures(
            overlap_ms=250, user_speaking=True, rel_loudness_db=0.0, partial_text="yeah theek hai"
        ),
    )

    # Warmup
    for _ in range(500):
        for s in samples:
            classify(s, tuning)

    # Benchmark: 3000 calls
    latencies_ns: list[int] = []
    iterations = 300
    for _ in range(iterations):
        for s in samples:
            t0 = time.perf_counter_ns()
            classify(s, tuning)
            t1 = time.perf_counter_ns()
            latencies_ns.append(t1 - t0)

    latencies_ns.sort()
    n = len(latencies_ns)
    p50_ms = latencies_ns[int(n * 0.50)] / 1_000_000.0
    p90_ms = latencies_ns[int(n * 0.90)] / 1_000_000.0
    p95_ms = latencies_ns[int(n * 0.95)] / 1_000_000.0
    p99_ms = latencies_ns[int(n * 0.99)] / 1_000_000.0

    print(f"\nClassifier Benchmark Results ({n} calls):")
    print(f"  p50: {p50_ms:.4f} ms")
    print(f"  p90: {p90_ms:.4f} ms")
    print(f"  p95: {p95_ms:.4f} ms")
    print(f"  p99: {p99_ms:.4f} ms (Target <= 0.2000 ms)")

    assert p99_ms <= 0.2, f"p99 call cost {p99_ms:.4f} ms exceeded 0.2 ms (INV-L11)"
