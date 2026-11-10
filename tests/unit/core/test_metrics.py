"""Unit tests for overlap.core.metrics (T4.03)."""

from __future__ import annotations

import pytest

from overlap.core.metrics import (
    DerivedMetrics,
    derive_metrics,
    evaluate_targets,
    percentile,
    rate,
    wilson_ci,
)
from overlap.core.types import Label, Tuning


def test_percentile_empty_and_small_samples() -> None:
    """Failure mode 3: percentile math wrong on small samples / empty sets."""
    assert percentile([], 50.0) is None
    assert percentile([], 95.0) is None

    # Single element returns exact element for any q in [0, 100]
    assert percentile([42.0], 0.0) == 42.0
    assert percentile([42.0], 50.0) == 42.0
    assert percentile([42.0], 95.0) == 42.0
    assert percentile([42.0], 100.0) == 42.0

    # Two elements linear interpolation
    assert percentile([10.0, 20.0], 0.0) == 10.0
    assert percentile([10.0, 20.0], 50.0) == 15.0
    assert percentile([10.0, 20.0], 100.0) == 20.0
    assert percentile([20.0, 10.0], 75.0) == pytest.approx(17.5)

    # Three elements
    assert percentile([10, 20, 30], 50.0) == 20.0
    assert percentile([10, 20, 30], 95.0) == pytest.approx(29.0)

    # Out of range q raises ValueError
    with pytest.raises(ValueError, match="Percentile q"):
        percentile([1, 2], -1.0)
    with pytest.raises(ValueError, match="Percentile q"):
        percentile([1, 2], 100.1)


def test_rate_and_wilson_ci() -> None:
    """Test safe rate and Wilson score confidence interval."""
    assert rate(0, 0) == 0.0
    assert rate(5, 0) == 0.0
    assert rate(5, -10) == 0.0
    assert rate(2, 40) == pytest.approx(0.05)

    assert wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = wilson_ci(0, 100)
    assert lo == pytest.approx(0.0)
    assert 0.03 < hi < 0.04

    lo50, hi50 = wilson_ci(50, 100)
    assert 0.40 < lo50 < 0.41
    assert 0.59 < hi50 < 0.60

    lo100, hi100 = wilson_ci(100, 100)
    assert 0.96 < lo100 < 0.97
    assert hi100 == pytest.approx(1.0)

    with pytest.raises(ValueError, match="Invalid successes"):
        wilson_ci(-1, 10)
    with pytest.raises(ValueError, match="Invalid successes"):
        wilson_ci(11, 10)


def test_t_stop_measured_from_onset_never_verdict() -> None:
    """Failure mode 2: metric definitions drift from §12 (e.g. t_stop measured from verdict)."""
    events = [
        {"name": "vad.onset", "ts_ns": 100_000_000, "epoch": 1},  # onset at 100 ms
        {"name": "duck.applied", "ts_ns": 120_000_000, "epoch": 1},  # duck at 120 ms -> 20 ms
        {
            "name": "verdict",
            "ts_ns": 360_000_000,  # verdict at 360 ms -> t_verdict = 260 ms
            "label": "INTERRUPT",
            "ground_truth": "INTERRUPT",
            "epoch": 1,
        },
        {
            "name": "yield.flush",
            "ts_ns": 380_000_000,  # flush at 380 ms -> t_stop from onset = 280 ms (NOT 20 ms!)
            "epoch": 1,
        },
    ]

    m = derive_metrics(events)
    assert m.t_duck_ms == (20.0,)
    assert m.t_verdict_ms == (260.0,)
    # Must be 380 - 100 = 280 ms (from onset), NOT 380 - 360 = 20 ms (from verdict)
    assert m.t_stop_ms == (280.0,)
    assert m.t_stop_p50_ms == 280.0
    assert m.t_stop_p95_ms == 280.0


def test_stale_epoch_events_ignored_in_metrics() -> None:
    """Failure mode 5: events from stale epochs counted in metrics."""
    events = [
        # Epoch 2 is active
        {"name": "vad.onset", "ts_ns": 100_000_000, "epoch": 2},
        {"name": "duck.applied", "ts_ns": 120_000_000, "epoch": 2},
        {
            "name": "verdict",
            "ts_ns": 300_000_000,
            "label": "BACKCHANNEL",
            "ground_truth": "BACKCHANNEL",
            "epoch": 2,
        },
        # Stale epoch 1 events arriving late (should be ignored!)
        {"name": "duck.applied", "ts_ns": 500_000_000, "epoch": 1},
        {
            "name": "verdict",
            "ts_ns": 600_000_000,
            "label": "INTERRUPT",
            "ground_truth": "BACKCHANNEL",
            "epoch": 1,
        },
        {
            "name": "yield.flush",
            "ts_ns": 700_000_000,
            "epoch": 1,
        },
        {
            "name": "yield.truncate",
            "ts_ns": 700_000_000,
            "error_words": 99,
            "epoch": 1,
        },
        # Explicit stale=True event
        {
            "name": "verdict",
            "ts_ns": 800_000_000,
            "label": "BACKCHANNEL",
            "ground_truth": "INTERRUPT",
            "epoch": 2,
            "stale": True,
        },
        # Supervisor / FSM stale notifications
        {
            "name": "fsm.ignored",
            "ts_ns": 810_000_000,
            "reason": "stale_epoch",
            "event_epoch": 1,
            "current_epoch": 2,
        },
    ]

    m = derive_metrics(events)
    assert m.stale_events_dropped == 6
    assert m.t_duck_ms == (20.0,)
    assert m.t_verdict_ms == (200.0,)
    assert m.t_stop_ms == ()
    assert m.false_interruption_rate == 0.0
    assert m.missed_interruption_rate == 0.0
    assert m.truncation_error_max_words == 0


def test_derive_all_section_12_metrics_and_evaluate_targets() -> None:
    """Verify FIR, missed rate, bc_timing, bc_rate, truncation_error, echo_false_duck."""
    events = [
        # 1. Backchannel correctly classified
        {"name": "vad.onset", "ts_ns": 1_000_000_000, "epoch": 1},
        {"name": "duck.applied", "ts_ns": 1_020_000_000, "epoch": 1},
        {"name": "vad.offset", "ts_ns": 1_200_000_000, "epoch": 1},
        {
            "name": "verdict",
            "ts_ns": 1_250_000_000,
            "label": Label.BACKCHANNEL,
            "ground_truth": "BACKCHANNEL",
            "epoch": 1,
        },
        # 2. False interruption (INTERRUPT on ground-truth BACKCHANNEL)
        {"name": "vad.onset", "ts_ns": 2_000_000_000, "epoch": 1},
        {"name": "duck.applied", "ts_ns": 2_030_000_000, "epoch": 1},
        {"name": "vad.offset", "ts_ns": 2_300_000_000, "epoch": 1},
        {
            "name": "verdict",
            "ts_ns": 2_300_000_000,
            "label": "INTERRUPT",
            "ground_truth": "BACKCHANNEL",
            "epoch": 1,
        },
        # 3. True interruption + stop + truncate
        {"name": "vad.onset", "ts_ns": 3_000_000_000, "epoch": 2},
        {"name": "duck.applied", "ts_ns": 3_020_000_000, "epoch": 2},
        {
            "name": "verdict",
            "ts_ns": 3_200_000_000,
            "label": "INTERRUPT",
            "ground_truth": "INTERRUPT",
            "epoch": 2,
        },
        {"name": "yield.flush", "ts_ns": 3_220_000_000, "epoch": 2},
        {
            "name": "yield.truncate",
            "ts_ns": 3_220_000_000,
            "heard_words": "Namaste main",
            "truth_words": "Namaste main aapki",
            "epoch": 2,
        },
        {"name": "vad.offset", "ts_ns": 3_500_000_000, "epoch": 2},
        # 4. Missed interruption (BACKCHANNEL on ground-truth INTERRUPT)
        {"name": "vad.onset", "ts_ns": 4_000_000_000, "epoch": 3},
        {"name": "duck.applied", "ts_ns": 4_020_000_000, "epoch": 3},
        {
            "name": "verdict",
            "ts_ns": 4_300_000_000,
            "label": "BACKCHANNEL",
            "ground_truth": "INTERRUPT",
            "epoch": 3,
        },
        {"name": "vad.offset", "ts_ns": 4_500_000_000, "epoch": 3},
        # 5. Long user turn (10 s) + agent backchannel fire & clip start
        {"name": "vad.onset", "ts_ns": 10_000_000_000, "epoch": 3},
        {"name": "vad.offset", "ts_ns": 20_000_000_000, "epoch": 3},
        {
            "name": "bc.agent.fire",
            "ts_ns": 20_240_000_000,
            "pause_start_ns": 20_000_000_000,
            "epoch": 3,
        },
        {"name": "bc.agent.clip_start", "ts_ns": 20_260_000_000, "epoch": 3},
        # 6. Echo-induced false duck
        {"name": "duck.applied", "ts_ns": 25_000_000_000, "is_echo": True, "epoch": 3},
    ]

    m = derive_metrics(events)
    assert m.false_interruption_rate == pytest.approx(0.5)  # 1 / 2
    assert m.missed_interruption_rate == pytest.approx(0.5)  # 1 / 2
    assert m.bc_timing_ms == (260.0,)
    assert m.bc_count == 1
    # Total user speech: 200 + 300 + 500 + 500 + 10000 = 11500 ms
    assert m.user_speech_ms == pytest.approx(11500.0)
    assert m.bc_rate_per_min == pytest.approx(1.0)
    assert m.truncation_error_words == (1,)
    assert m.truncation_error_max_words == 1
    assert m.echo_false_duck == 1

    checks = evaluate_targets(m, Tuning())
    check_map = {c.name: c for c in checks}
    assert check_map["t_duck"].passed is True
    assert check_map["t_verdict"].passed is True
    assert check_map["t_stop"].passed is True
    assert check_map["false_interruption_rate"].passed is False  # 50% > 5%
    assert check_map["missed_interruption_rate"].passed is False  # 50% > 5%
    assert check_map["bc_timing"].passed is True  # 260 in 200..450
    assert check_map["bc_rate"].passed is True  # ~5.22 <= 6
    assert check_map["truncation_error_words"].passed is True  # 1 <= 1
    assert check_map["echo_false_duck"].passed is False  # 1 != 0


def test_evaluate_targets_all_pass_with_tuning() -> None:
    """Verify evaluate_targets passes when all metrics satisfy Tuning thresholds."""
    tuning = Tuning()
    m = DerivedMetrics(
        t_duck_ms=(20.0, 20.0),
        t_duck_p50_ms=20.0,
        t_duck_p95_ms=20.0,
        t_verdict_ms=(200.0, 250.0),
        t_verdict_p50_ms=225.0,
        t_verdict_p95_ms=247.5,
        t_stop_ms=(220.0,),
        t_stop_p50_ms=220.0,
        t_stop_p95_ms=220.0,
        false_interruption_rate=0.0,
        missed_interruption_rate=0.0,
        bc_timing_ms=(240.0,),
        bc_timing_p50_ms=240.0,
        bc_timing_min_ms=240.0,
        bc_timing_max_ms=240.0,
        bc_rate_per_min=3.0,
        truncation_error_words=(0, 1),
        truncation_error_max_words=1,
        truncation_error_p95_words=1.0,
        echo_false_duck=0,
    )
    checks = evaluate_targets(m, tuning=tuning)
    assert len(checks) == 9
    assert all(c.passed for c in checks)
