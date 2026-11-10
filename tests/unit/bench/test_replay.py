"""Unit tests for overlap.bench.corpus and overlap.bench.replay (T4.04).

Covers all done_when and failure_modes for corpus validation and offline replay:
- Corpus schema, >= 200 cases, >= 8 languages, balanced across BACKCHANNEL/INTERRUPT/SIDE_TALK
- Deterministic train/held-out split by SHA-256 hash
- Failure mode: corpus imbalance hides missed-interruption rate
- Failure mode: corpus/labels leak real PII
- Failure mode: evaluator measures a different pipeline than the live engine
- Replay reports FIR, missed-interruption rate, confusion matrix, per-language rows
- Held-out FIR <= 5% and missed-interruption rate <= 5%
- Deterministic replay output
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

import overlap.core.classifier as classifier_mod
import overlap.core.fsm as fsm_mod
from overlap.bench.corpus import (
    CorpusCase,
    TimelineStep,
    assign_split,
    load_corpus,
    split_corpus,
    validate_corpus_balance,
    validate_no_pii,
)
from overlap.bench.replay import (
    format_replay_report,
    replay_case,
    replay_corpus,
)
from overlap.core.types import FsmState, Label, Tuning


def test_corpus_format_size_languages_balance_and_hash_split() -> None:
    """Verify default corpus has >= 200 cases, >= 8 languages, balanced labels, and hash split."""
    cases = load_corpus()
    assert len(cases) >= 200, f"Expected >= 200 cases, got {len(cases)}"

    languages = {c.lang for c in cases}
    assert len(languages) >= 8, f"Expected >= 8 languages, got {len(languages)}: {languages}"

    by_label: dict[Label, int] = {
        Label.BACKCHANNEL: 0,
        Label.INTERRUPT: 0,
        Label.SIDE_TALK: 0,
    }
    for case in cases:
        assert case.id.strip() != ""
        assert case.lang.strip() != ""
        assert len(case.timeline) >= 1
        assert case.label in by_label
        by_label[case.label] += 1
        for step in case.timeline:
            assert isinstance(step.t_ms, int)
            assert step.t_ms >= 0
            assert isinstance(step.vad, bool)
            assert isinstance(step.loudness, float)

    # Validate balance and PII cleanliness on the real corpus
    validate_corpus_balance(cases)
    validate_no_pii(cases)

    # Check deterministic hash split
    train_cases, held_out_cases = split_corpus(cases)
    assert len(train_cases) > 0
    assert len(held_out_cases) > 0
    assert len(train_cases) + len(held_out_cases) == len(cases)

    # Repeating split_corpus produces exact same partition
    train_2, held_out_2 = split_corpus(cases)
    assert [c.id for c in train_cases] == [c.id for c in train_2]
    assert [c.id for c in held_out_cases] == [c.id for c in held_out_2]

    # Both splits contain all three labels
    train_labels = {c.label for c in train_cases}
    held_out_labels = {c.label for c in held_out_cases}
    assert train_labels == {Label.BACKCHANNEL, Label.INTERRUPT, Label.SIDE_TALK}
    assert held_out_labels == {Label.BACKCHANNEL, Label.INTERRUPT, Label.SIDE_TALK}


def test_failure_mode_corpus_imbalance_detected() -> None:
    """Failure mode 1: corpus imbalance (mostly backchannels) is rejected."""
    base_bc = CorpusCase(
        id="bc_1",
        lang="hi",
        timeline=(
            TimelineStep(t_ms=0, vad=True, loudness=0.0, text="haan", conf=0.9),
            TimelineStep(t_ms=220, vad=False, loudness=0.0, text="haan", conf=0.95),
        ),
        label=Label.BACKCHANNEL,
    )
    imbalanced: list[CorpusCase] = [replace(base_bc, id=f"bc_{i}") for i in range(195)]
    for i in range(3):
        imbalanced.append(
            CorpusCase(
                id=f"int_{i}",
                lang="en",
                timeline=(
                    TimelineStep(t_ms=0, vad=True, loudness=0.0, text="haan", conf=0.9),
                    TimelineStep(t_ms=220, vad=False, loudness=0.0, text="haan", conf=0.95),
                ),
                label=Label.INTERRUPT,
            )
        )
    for i in range(2):
        imbalanced.append(
            CorpusCase(
                id=f"st_{i}",
                lang="ta",
                timeline=(
                    TimelineStep(t_ms=0, vad=True, loudness=-12.0, text="foo", conf=0.2),
                    TimelineStep(t_ms=260, vad=True, loudness=-12.0, text="foo bar", conf=0.2),
                ),
                label=Label.SIDE_TALK,
            )
        )

    with pytest.raises(ValueError, match="imbalance"):
        validate_corpus_balance(imbalanced, min_languages=3)

    # Even though overall accuracy is 197/200 = 98.5%, missed_interruption_rate is 100%
    # and replay_corpus marks passed=False.
    report = replay_corpus(imbalanced, validate_balance=False)
    assert report.accuracy > 0.95
    assert report.missed_interruption_rate == 1.0
    assert report.passed is False


def test_failure_mode_corpus_pii_leak_detected() -> None:
    """Failure mode 4: corpus or labels leaking real PII raises ValueError."""
    clean_case = CorpusCase(
        id="clean_01",
        lang="en",
        timeline=(TimelineStep(t_ms=0, vad=True, loudness=0.0, text="wait a second", conf=0.9),),
        label=Label.INTERRUPT,
    )
    validate_no_pii([clean_case])

    email_case = replace(
        clean_case,
        timeline=(
            TimelineStep(
                t_ms=0, vad=True, loudness=0.0, text="mail me at rajesh.sharma@gmail.com", conf=0.9
            ),
        ),
    )
    with pytest.raises(ValueError, match="PII"):
        validate_no_pii([email_case])

    phone_case = replace(
        clean_case,
        timeline=(
            TimelineStep(
                t_ms=0, vad=True, loudness=0.0, text="call 9876543210 right now", conf=0.9
            ),
        ),
    )
    with pytest.raises(ValueError, match="PII"):
        validate_no_pii([phone_case])

    aadhaar_case = replace(
        clean_case,
        timeline=(
            TimelineStep(t_ms=0, vad=True, loudness=0.0, text="id is 4567 8901 2345", conf=0.9),
        ),
    )
    with pytest.raises(ValueError, match="PII"):
        validate_no_pii([aadhaar_case])


def test_replay_runs_real_classifier_and_fsm_pipeline() -> None:
    """Failure mode 5 & done_when 2: replay invokes real classify() and FSM transition_context()."""
    bc_case = CorpusCase(
        id="test_bc",
        lang="hi",
        timeline=(
            TimelineStep(t_ms=0, vad=True, loudness=-0.5, text=None, conf=None),
            TimelineStep(t_ms=120, vad=True, loudness=-0.5, text="haan", conf=0.9),
            TimelineStep(t_ms=220, vad=False, loudness=-1.0, text="haan", conf=0.95),
        ),
        label=Label.BACKCHANNEL,
    )
    int_case = CorpusCase(
        id="test_int",
        lang="hi",
        timeline=(
            TimelineStep(t_ms=0, vad=True, loudness=2.0, text=None, conf=None),
            TimelineStep(t_ms=80, vad=True, loudness=2.5, text="ruko", conf=0.95),
        ),
        label=Label.INTERRUPT,
    )
    late_esc_case = CorpusCase(
        id="test_late_esc",
        lang="hi",
        timeline=(
            TimelineStep(t_ms=0, vad=True, loudness=0.0, text="haan", conf=0.9),
            TimelineStep(t_ms=210, vad=True, loudness=0.0, text="haan", conf=0.92),
            TimelineStep(t_ms=340, vad=True, loudness=1.5, text="haan lekin ruko", conf=0.95),
        ),
        label=Label.INTERRUPT,
    )

    with (
        patch.object(classifier_mod, "classify", wraps=classifier_mod.classify) as spy_classify,
        patch.object(fsm_mod, "transition_context", wraps=fsm_mod.transition_context) as spy_fsm,
    ):
        res_bc = replay_case(bc_case, Tuning())
        res_int = replay_case(int_case, Tuning())
        res_esc = replay_case(late_esc_case, Tuning())

        assert spy_classify.call_count >= 3
        assert spy_fsm.call_count >= 6

    assert res_bc.predicted == Label.BACKCHANNEL
    assert res_bc.final_fsm_state == FsmState.SPEAKING
    assert res_bc.flushed is False

    assert res_int.predicted == Label.INTERRUPT
    assert res_int.final_fsm_state == FsmState.YIELDED
    assert res_int.flushed is True
    assert res_int.verdict_ms == 80

    assert res_esc.predicted == Label.INTERRUPT
    assert res_esc.final_fsm_state == FsmState.YIELDED
    assert res_esc.flushed is True
    assert res_esc.verdict_ms == 340


def test_replay_reports_fir_missed_confusion_matrix_and_per_language_rows() -> None:
    """done_when 2 & 4: replay over corpus and held-out split meets FIR <= 5% and missed <= 5%."""
    cases = load_corpus()
    full_report = replay_corpus(cases)

    # Check confusion matrix structure and sum
    expected_keys = {"BACKCHANNEL", "INTERRUPT", "SIDE_TALK"}
    assert set(full_report.confusion_matrix.keys()) == expected_keys
    matrix_sum = sum(
        count for row in full_report.confusion_matrix.values() for count in row.values()
    )
    assert matrix_sum == full_report.total

    # Check per-language rows
    assert len(full_report.per_language) >= 8
    assert sum(row.total for row in full_report.per_language) == full_report.total

    # Check targets on full corpus
    assert full_report.false_interruption_rate <= 0.05
    assert full_report.missed_interruption_rate <= 0.05
    assert full_report.passed is True

    # Check targets on held-out split specifically
    _, held_out = split_corpus(cases)
    held_out_report = replay_corpus(held_out)
    assert held_out_report.false_interruption_rate <= 0.05
    assert held_out_report.missed_interruption_rate <= 0.05
    assert held_out_report.passed is True

    formatted = format_replay_report(full_report)
    assert "false_interruption_rate" in formatted
    assert "missed_interruption_rate" in formatted
    assert "Confusion Matrix" in formatted
    assert "Per-Language" in formatted
    assert "PASS" in formatted


def test_replay_determinism() -> None:
    """done_when 5: identical inputs produce identical ReplayReport and formatted output."""
    cases = load_corpus()
    r1 = replay_corpus(cases)
    r2 = replay_corpus(cases)
    assert r1 == r2
    assert format_replay_report(r1) == format_replay_report(r2)
    assert assign_split("hi_bc_001") == assign_split("hi_bc_001")
