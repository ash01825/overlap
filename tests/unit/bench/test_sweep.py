"""Unit tests for overlap.bench.sweep (T4.04).

Covers all done_when and failure_modes for threshold sweep:
- Grid search on train split, evaluation on held-out split
- Held-out FIR <= 5% and missed-interruption rate <= 5%
- Failure mode: tuning overfits to the corpus (no held-out split rejected)
- Failure mode: sweep changes Tuning defaults silently (detected / prevented)
- Deterministic sweep report matching reports/classifier_sweep.md
"""

from __future__ import annotations

from pathlib import Path

import pytest

from overlap.bench.corpus import CorpusCase, TimelineStep, load_corpus
from overlap.bench.sweep import (
    ParameterGrid,
    format_sweep_report,
    run_sweep,
)
from overlap.core.types import Label, Tuning


def test_sweep_grid_search_train_and_held_out_evaluation() -> None:
    """done_when 3 & 4: sweep grid-searches on train split and meets held-out targets."""
    cases = load_corpus()
    grid = ParameterGrid(
        classify_min_ms=(220, 250, 280),
        content_min_overlap_ms=(220, 250),
        bc_min_ms=(180, 200),
        assertive_db=(2.0, 3.0, 4.0),
        sidetalk_db=(-10.0, -9.0, -7.5),
    )
    result = run_sweep(cases, grid=grid)

    assert len(result.candidates) == 3 * 2 * 2 * 3 * 3
    assert result.train_size > 0
    assert result.held_out_size > 0
    assert result.train_size + result.held_out_size == len(cases)

    # Held-out targets must be met
    assert result.best_held_out_report.false_interruption_rate <= 0.05
    assert result.best_held_out_report.missed_interruption_rate <= 0.05
    assert result.held_out_passed is True

    # Verify committed report file exists and matches generated report
    report_path = Path(__file__).resolve().parents[3] / "reports" / "classifier_sweep.md"
    assert report_path.exists(), f"Missing {report_path}"
    committed_md = report_path.read_text(encoding="utf-8")
    default_md = format_sweep_report(run_sweep(cases))
    assert committed_md == default_md


def test_failure_mode_overfitting_no_held_out_split_rejected() -> None:
    """Failure mode 2: sweep refuses to run if held-out split or train split is empty."""
    # Construct cases whose IDs all hash to the 'train' split
    cases = load_corpus()
    train_only = [c for c in cases if c.split == "train"]
    assert len(train_only) > 0

    with pytest.raises(ValueError, match="held-out"):
        run_sweep(train_only)

    held_out_only = [c for c in cases if c.split == "held_out"]
    with pytest.raises(ValueError, match="train"):
        run_sweep(held_out_only)


def test_failure_mode_sweep_never_mutates_tuning_defaults_silently() -> None:
    """Failure mode 3: sweep never mutates Tuning defaults and reports deltas explicitly."""
    baseline_before = Tuning()

    # Craft a small synthetic corpus where assertive_db=1.5 wins on train split
    # while baseline assertive_db=3.0 misses an assertive acoustic case at +2.0 dB
    train_cases: list[CorpusCase] = []
    held_out_cases: list[CorpusCase] = []
    for i in range(40):
        cid = f"synth_case_{i:03d}"
        step_bc = (
            TimelineStep(t_ms=0, vad=True, loudness=0.0, text="haan", conf=0.95),
            TimelineStep(t_ms=220, vad=False, loudness=0.0, text="haan", conf=0.95),
        )
        step_int = (
            TimelineStep(t_ms=0, vad=True, loudness=2.0, text=None, conf=None),
            TimelineStep(t_ms=270, vad=True, loudness=2.0, text=None, conf=None),
            TimelineStep(t_ms=390, vad=False, loudness=2.0, text=None, conf=None),
        )
        step_st = (
            TimelineStep(t_ms=0, vad=True, loudness=-12.0, text="bg", conf=0.2),
            TimelineStep(t_ms=260, vad=True, loudness=-12.0, text="bg noise", conf=0.2),
        )
        if i % 3 == 0:
            c = CorpusCase(id=cid, lang="hi", timeline=step_bc, label=Label.BACKCHANNEL)
        elif i % 3 == 1:
            c = CorpusCase(id=cid, lang="en", timeline=step_int, label=Label.INTERRUPT)
        else:
            c = CorpusCase(id=cid, lang="ta", timeline=step_st, label=Label.SIDE_TALK)

        if c.split == "train":
            train_cases.append(c)
        else:
            held_out_cases.append(c)

    synth_corpus = train_cases + held_out_cases
    grid = ParameterGrid(
        classify_min_ms=(250,),
        content_min_overlap_ms=(250,),
        bc_min_ms=(200,),
        assertive_db=(1.5, 3.0),
        sidetalk_db=(-9.0,),
    )
    res = run_sweep(synth_corpus, grid=grid)

    # Verify Tuning() defaults were NOT mutated in-place
    baseline_after = Tuning()
    assert baseline_before == baseline_after
    assert baseline_after.assertive_db == 3.0

    # And the sweep explicitly surfaces the recommended change (assertive_db: 3.0 -> 1.5)
    assert "assertive_db" in res.recommended_changes
    assert res.recommended_changes["assertive_db"] == (3.0, 1.5)

    md = format_sweep_report(res)
    assert "assertive_db" in md
    assert "3.0" in md and "1.5" in md


def test_sweep_determinism() -> None:
    """done_when 5: running sweep twice produces identical results and markdown."""
    cases = load_corpus()
    grid = ParameterGrid(
        classify_min_ms=(250,),
        content_min_overlap_ms=(250,),
        bc_min_ms=(200,),
        assertive_db=(2.5, 3.0),
        sidetalk_db=(-9.0, -8.0),
    )
    s1 = run_sweep(cases, grid=grid)
    s2 = run_sweep(cases, grid=grid)
    assert s1 == s2
    assert format_sweep_report(s1) == format_sweep_report(s2)
