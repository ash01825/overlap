"""Hermetic unit tests for `overlap.bench.scripted_caller` and `bench/ab_scenarios/*.yaml`."""

from __future__ import annotations

from pathlib import Path

from overlap.bench.ab import run_hermetic_async
from overlap.bench.scripted_caller import (
    build_stimulus_timeline,
    load_scenario,
    load_scenarios,
    run_scenario_on_arm,
)
from overlap.core.types import Tuning


def test_ab_scenarios_directory_has_at_least_eight_required_scenarios() -> None:
    """Verify >= 8 scenario YAMLs covering all required caller behaviors."""
    scenarios = load_scenarios(Path("bench/ab_scenarios"))
    assert len(scenarios) >= 8

    all_ids = {s.scenario_id for s in scenarios}
    assert "01_mid_reply_haan" in all_ids
    assert "02_real_interruption" in all_ids
    assert "03_long_explanation_with_pauses" in all_ids
    assert "04_side_talk_noise" in all_ids
    assert "05_codeswitch_hinglish" in all_ids
    assert "06_codeswitch_tamil_english" in all_ids
    assert "07_repeated_ok" in all_ids
    assert "08_correction_after_backchannel" in all_ids

    # Check required caller behaviors across the scenarios
    has_haan = any(
        any(ev.text.lower() == "haan" and ev.ground_truth == "BACKCHANNEL" for ev in s.events)
        for s in scenarios
    )
    has_real_interrupt = any(
        any(ev.ground_truth == "INTERRUPT" for ev in s.events) for s in scenarios
    )
    has_long_turn_pauses = any(
        any(ev.kind == "user_long_turn" and len(ev.clauses) >= 2 for ev in s.events)
        for s in scenarios
    )
    has_side_talk = any(
        any(
            ev.kind == "user_overlap" and ev.text == "" and ev.duration_ms <= 200 for ev in s.events
        )
        for s in scenarios
    )
    langs = {s.lang for s in scenarios}
    has_repeated_ok = any(
        sum(1 for ev in s.events if ev.text.lower() in ("ok", "okay")) >= 2 for s in scenarios
    )

    assert has_haan
    assert has_real_interrupt
    assert has_long_turn_pauses
    assert has_side_talk
    assert {"hi-IN", "ta-IN", "en-IN"}.issubset(langs)
    assert has_repeated_ok


def test_stimulus_timeline_is_deterministic_and_seeded() -> None:
    """Same seed produces byte-identical stimulus frames; different seed alters noise jitter."""
    scenario = load_scenario(Path("bench/ab_scenarios/01_mid_reply_haan.yaml"))
    t1 = build_stimulus_timeline(scenario, seed=101)
    t2 = build_stimulus_timeline(scenario, seed=101)
    t3 = build_stimulus_timeline(scenario, seed=202)

    assert t1.digest_sha256 == t2.digest_sha256
    assert len(t1.frames) == len(t2.frames)
    assert [f.ts_ns for f in t1.frames] == [f.ts_ns for f in t2.frames]
    assert [f.pcm for f in t1.frames] == [f.pcm for f in t2.frames]

    # Different seed changes synthesized PCM jitter while keeping timing identical
    assert t1.digest_sha256 != t3.digest_sha256
    assert [f.ts_ns for f in t1.frames] == [f.ts_ns for f in t3.frames]
    assert [f.is_speech for f in t1.frames] == [f.is_speech for f in t3.frames]


def test_both_arms_receive_identical_stimulus_timelines_no_drift() -> None:
    """Hermetic test proving Stock and OVERLAP arms receive identical frame timestamps & PCM."""
    scenarios = load_scenarios(Path("bench/ab_scenarios"))
    tuning = Tuning()

    async def _run() -> None:
        for sc in scenarios:
            trace_stock = await run_scenario_on_arm(sc, arm="stock", seed=77, tuning=tuning)
            trace_overlap = await run_scenario_on_arm(sc, arm="overlap", seed=77, tuning=tuning)

            assert trace_stock.stimulus_digest == trace_overlap.stimulus_digest
            assert trace_stock.frame_timestamps_ns == trace_overlap.frame_timestamps_ns
            assert trace_stock.pcm_hashes == trace_overlap.pcm_hashes
            assert len(trace_stock.frame_timestamps_ns) > 50

            # Verify strict 20 ms step spacing (zero drift)
            step_ns = tuning.frame_ms * 1_000_000
            for idx, ts_ns in enumerate(trace_stock.frame_timestamps_ns):
                assert ts_ns == idx * step_ns

    run_hermetic_async(_run())


def test_stock_vs_overlap_behavioral_divergence_on_backchannel_and_long_turn() -> None:
    """Stock falsely interrupts on 'haan' and never emits agent BCs; OVERLAP ducks and BCs."""
    sc_haan = load_scenario(Path("bench/ab_scenarios/01_mid_reply_haan.yaml"))
    sc_long = load_scenario(Path("bench/ab_scenarios/03_long_explanation_with_pauses.yaml"))

    async def _run() -> None:
        tr_stock_haan = await run_scenario_on_arm(sc_haan, arm="stock", seed=42)
        tr_ov_haan = await run_scenario_on_arm(sc_haan, arm="overlap", seed=42)

        assert tr_stock_haan.false_interruption_count == 1
        assert tr_ov_haan.false_interruption_count == 0
        assert tr_ov_haan.heard_words_preserved is True
        assert tr_stock_haan.heard_words_preserved is False

        tr_stock_long = await run_scenario_on_arm(sc_long, arm="stock", seed=42)
        tr_ov_long = await run_scenario_on_arm(sc_long, arm="overlap", seed=42)

        assert tr_stock_long.agent_backchannel_count == 0
        assert tr_ov_long.agent_backchannel_count >= 1

    run_hermetic_async(_run())
