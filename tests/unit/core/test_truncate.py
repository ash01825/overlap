"""Unit tests for SpeakingRateEstimator and truncate_to_heard.

Tests midpoint rule with word timings, calibrated-rate fallback snapping to
word boundaries, multi-clause truncation, Indic script cluster integrity
(Devanagari, Tamil, Telugu, Hinglish), property-based monotonicity, and
the 200-case synthetic ground-truth test (INV-T1).
"""

from __future__ import annotations

import random
import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from overlap.core.truncate import (
    ClauseAudio,
    SpeakingRateEstimator,
    count_grapheme_clusters,
    split_grapheme_clusters,
    truncate_to_heard,
)
from overlap.core.types import Tuning, WordTiming

# =============================================================================
# SpeakingRateEstimator & Grapheme Clusters
# =============================================================================


def test_grapheme_clusters_indic_scripts() -> None:
    """Indic scripts: grapheme clusters split (combining matras/viramas) producing garbled text."""
    samples = [
        ("hindi", "नमस्ते दुनिया"),
        ("tamil", "வணக்கம் உலகம்"),
        ("telugu", "నమస్కారం ప్రపంచం"),
        ("hinglish", "arre bhai kya chal raha hai"),
    ]
    for lang, text in samples:
        clusters = split_grapheme_clusters(text)
        assert len(clusters) == count_grapheme_clusters(text)
        assert "".join(clusters) == text
        # Verify combining marks are never the first character of any cluster
        for c in clusters:
            first_char = c[0]
            cat = unicodedata.category(first_char)
            assert cat not in ("Mn", "Mc", "Me"), f"Orphaned combining mark in {c!r} for {lang}"


def test_speaking_rate_estimator_cold_start() -> None:
    """Failure mode: no word timings: rate estimate cold-start gives 0 or absurd words."""
    est = SpeakingRateEstimator()
    for lang in ["en", "hi", "ta", "te", "kn", "mr", "unknown", None]:
        rate = est.get_ms_per_unit(lang)
        assert rate > 20.0
        assert rate < 200.0


def test_speaking_rate_estimator_ema_update() -> None:
    est = SpeakingRateEstimator(alpha=0.5, default_ms_per_unit=80.0)

    # First update initializes rate for "hi" to observed rate
    first_observed = 1000.0 / count_grapheme_clusters("नमस्ते")
    rate1 = est.update("hi", audio_ms=1000, text="नमस्ते")
    assert abs(rate1 - first_observed) < 1e-4
    assert est.get_ms_per_unit("hi") == rate1

    # Second update updates EMA: 0.5 * second_observed + 0.5 * rate1
    second_observed = 800.0 / count_grapheme_clusters("दुनिया")
    rate2 = est.update("hi", audio_ms=800, text="दुनिया")
    expected = 0.5 * second_observed + 0.5 * rate1
    assert abs(rate2 - expected) < 1e-4
    assert est.get_ms_per_unit("hi") == rate2


def test_speaking_rate_estimator_ignores_invalid_updates() -> None:
    est = SpeakingRateEstimator()
    base_rate = est.get_ms_per_unit("en")
    est.update("en", audio_ms=-100, text="hello")
    assert est.get_ms_per_unit("en") == base_rate
    est.update("en", audio_ms=500, text="")
    assert est.get_ms_per_unit("en") == base_rate


# =============================================================================
# Midpoint Rule with Word Timings (§8.3)
# =============================================================================


def test_midpoint_rule_boundary_conditions() -> None:
    """Failure mode: word counted as heard that was cut mid-word (midpoint rule violated)."""
    # Word "hello": start=100ms, end=200ms => midpoint=150ms
    # Word "world": start=250ms, end=400ms => midpoint=325ms
    clause = ClauseAudio(
        text="hello world",
        duration_ms=500,
        words=(
            WordTiming(word="hello", start_ms=100, end_ms=200),
            WordTiming(word="world", start_ms=250, end_ms=400),
        ),
    )

    # Cut at 149 ms: before midpoint of "hello" (150 ms) => 0 words heard
    res_149 = truncate_to_heard([clause], played_ms=149)
    assert res_149.words == ()
    assert res_149.text == ""

    # Cut at exactly 150 ms: midpoint of "hello" reached => "hello" heard
    res_150 = truncate_to_heard([clause], played_ms=150)
    assert res_150.words == ("hello",)
    assert res_150.text == f"hello{Tuning().truncation_marker}"

    # Cut at 324 ms: after "hello", but before midpoint of "world" (325 ms)
    res_324 = truncate_to_heard([clause], played_ms=324)
    assert res_324.words == ("hello",)

    # Cut at 325 ms: midpoint of "world" reached => both heard
    res_325 = truncate_to_heard([clause], played_ms=325)
    assert res_325.words == ("hello", "world")


# =============================================================================
# Calibrated-Rate Fallback (No Word Timings)
# =============================================================================


def test_calibrated_rate_snaps_down_to_word_boundary() -> None:
    """Failure mode: truncated at non-word boundary; trailing partial word leaks."""
    # Text: "one two three four"
    # Using rate of 100 ms / unit.
    est = SpeakingRateEstimator(default_ms_per_unit=100.0)
    clause = ClauseAudio(
        text="one two three four",
        duration_ms=1800,
        words=(),  # no timings => fallback
    )
    # Clusters: 'o','n','e',' ','t','w','o',' '... (18 clusters)
    # "one" ends at cluster 3 (300 ms).
    # "two" ends at cluster 7 (700 ms).
    # Cut at 500 ms => 5 units ('one t'). Snapped down to word boundary must drop 't'!
    res = truncate_to_heard([clause], played_ms=500, rate_est=est)
    assert res.words == ("one",)
    assert res.text == f"one{Tuning().truncation_marker}"
    assert "t" not in res.words  # No partial word leak!


# =============================================================================
# Multi-Clause Turns & Un-synthesized Clauses
# =============================================================================


def test_multi_clause_turns_later_clauses_dropped() -> None:
    """Failure mode: multi-clause turns: later clauses (not yet spoken) included."""
    clause0 = ClauseAudio(
        text="Wait a second,",
        duration_ms=1000,
        words=(
            WordTiming("Wait", 0, 300),
            WordTiming("a", 310, 400),
            WordTiming("second,", 410, 800),
        ),
    )
    clause1 = ClauseAudio(
        text="I am checking your records.",
        duration_ms=1500,
        words=(
            WordTiming("I", 0, 200),
            WordTiming("am", 210, 350),
            WordTiming("checking", 360, 800),
            WordTiming("your", 810, 1000),
            WordTiming("records.", 1010, 1400),
        ),
    )
    clause2 = ClauseAudio(
        text="This should only take a moment.",
        duration_ms=1200,
        synthesized=False,  # un-synthesized remainder
    )

    clauses = [clause0, clause1, clause2]

    # Played 350 ms: inside clause 0, midpoint of 'a' is (310+400)/2 = 355 ms
    # At 350 ms, 'a' is not yet heard
    res_350 = truncate_to_heard(clauses, played_ms=350)
    assert res_350.words == ("Wait",)
    assert res_350.text == f"Wait{Tuning().truncation_marker}"

    # Played 360 ms: passes midpoint of 'a' (355 ms) => 'Wait a'
    res_360 = truncate_to_heard(clauses, played_ms=360)
    assert res_360.words == ("Wait", "a")
    assert res_360.text == f"Wait a{Tuning().truncation_marker}"

    # Played 1000 ms: all of clause 0, 0 ms of clause 1
    res_1000 = truncate_to_heard(clauses, played_ms=1000)
    assert res_1000.words == ("Wait", "a", "second,")
    assert "checking" not in res_1000.words

    # Played 1500 ms: clause 0 (1000 ms) + 500 ms of clause 1 ("I", "am")
    res_1500 = truncate_to_heard(clauses, played_ms=1500)
    assert res_1500.words == ("Wait", "a", "second,", "I", "am")
    assert "checking" not in res_1500.words

    # Played 3000 ms: all of clause 0 and clause 1, but clause 2 is un-synthesized
    res_3000 = truncate_to_heard(clauses, played_ms=3000)
    assert "records." in res_3000.words
    assert "This" not in res_3000.words
    assert res_3000.fully_played is False  # clause 2 was un-synthesized
    assert res_3000.text.endswith(Tuning().truncation_marker)


def test_fully_played_turn_has_no_truncation_marker() -> None:
    clause = ClauseAudio(
        text="Good morning everyone.",
        duration_ms=1000,
        words=(
            WordTiming("Good", 0, 250),
            WordTiming("morning", 260, 600),
            WordTiming("everyone.", 610, 950),
        ),
        synthesized=True,
    )
    res = truncate_to_heard([clause], played_ms=1200)
    assert res.fully_played is True
    assert res.words == ("Good", "morning", "everyone.")
    assert res.text == "Good morning everyone."
    assert Tuning().truncation_marker not in res.text


# =============================================================================
# Indic Scripts Truncation Tests
# =============================================================================


@pytest.mark.parametrize(
    ("lang", "text"),
    [
        ("hi", "नमस्ते आप कैसे हैं आज का मौसम अच्छा है"),
        ("ta", "வணக்கம் நீங்கள் எப்படி இருக்கிறீர்கள் இன்று நன்றாக உள்ளது"),
        ("te", "నమస్కారం మీరు ఎలా ఉన్నారు ఈ రోజు బాగుంది"),
        ("hi", "arre bhai yeh kya plan hai"),
    ],
)
def test_indic_scripts_truncation_preserves_words(lang: str, text: str) -> None:
    est = SpeakingRateEstimator()
    clause = ClauseAudio(text=text, duration_ms=2500, lang=lang)

    for played_ms in [0, 400, 800, 1500, 2500]:
        res = truncate_to_heard([clause], played_ms=played_ms, rate_est=est)
        all_words = text.split()
        assert len(res.words) <= len(all_words)
        assert res.words == tuple(all_words[: len(res.words)])
        for word in res.words:
            # Check cluster integrity: no orphaned matra/virama
            cat = unicodedata.category(word[0])
            assert cat not in ("Mn", "Mc", "Me")


# =============================================================================
# Property Test: Prefix and Monotonicity
# =============================================================================


@settings(max_examples=50)
@given(
    text=st.sampled_from(
        [
            "The quick brown fox jumps over the lazy dog",
            "नमस्ते आप कैसे हैं आज सब कुछ ठीक है",
            "வணக்கம் நீங்கள் எப்படி இருக்கிறீர்கள் நண்பரே",
            "నమస్కారం మీరు ఎలా ఉన్నారు ఈ రోజు",
        ]
    ),
    play_times=st.lists(st.integers(min_value=-50, max_value=3000), min_size=2, max_size=5),
)
def test_property_heard_words_prefix_and_monotonic(text: str, play_times: list[int]) -> None:
    """Property test: heard_words is always a prefix of full words, monotonic in played_ms."""
    sorted_times = sorted(play_times)
    words = text.split()
    total_dur = 2000
    est = SpeakingRateEstimator()
    clause = ClauseAudio(text=text, duration_ms=total_dur)

    prev_heard: tuple[str, ...] = ()
    for p_ms in sorted_times:
        res = truncate_to_heard([clause], played_ms=p_ms, rate_est=est)
        # 1. Prefix property
        assert res.heard_words == tuple(words[: len(res.heard_words)])
        # 2. Monotonicity property
        assert len(res.heard_words) >= len(prev_heard)
        assert res.heard_words[: len(prev_heard)] == prev_heard
        prev_heard = res.heard_words


# =============================================================================
# Synthetic Ground Truth Tests (INV-T1, 200 Seeded Cases)
# =============================================================================


def test_synthetic_ground_truth_inv_t1_200_cases() -> None:
    """Synthetic ground truth tests: truncation error <= 1 word (INV-T1) across 200 seeded cases."""
    test_corpora = [
        ("en", "The quick brown fox jumps over the lazy dog near the river bank"),
        ("hi", "नमस्ते आप कैसे हैं आज का मौसम बहुत अच्छा है आपका दिन शुभ हो"),
        ("ta", "வணக்கம் நீங்கள் எப்படி இருக்கிறீர்கள் இன்று வானிலை மிகவும் நன்றாக உள்ளது"),
        ("te", "నమస్కారం మీరు ఎలా ఉన్నారు ఈ రోజు వాతావరణం చాలా బాగుంది శుభదినం"),
        ("en", "Welcome to full duplex voice conversations with live streaming audio"),
    ]

    rng = random.Random(1337)
    errors: list[int] = []

    for i in range(200):
        lang, text = test_corpora[i % len(test_corpora)]
        words = text.split()
        ms_per_unit = rng.uniform(70.0, 95.0)

        # Synthesize word timings with natural variations
        t = 0
        timings: list[WordTiming] = []
        for w in words:
            dur = int(count_grapheme_clusters(w) * ms_per_unit + rng.uniform(-8.0, 8.0))
            dur = max(dur, 40)
            timings.append(WordTiming(word=w, start_ms=t, end_ms=t + dur))
            t += dur + rng.choice([20, 30, 40])
        total_dur = t

        # Ground truth: midpoint rule against actual word timings
        cut_ms = rng.randint(0, total_dur + 100)
        gt_res = truncate_to_heard(
            [ClauseAudio(text=text, duration_ms=total_dur, words=tuple(timings), lang=lang)],
            played_ms=cut_ms,
        )

        # Fallback estimation without word timings:
        est = SpeakingRateEstimator()
        # Calibrate with the speech
        est.update(lang, audio_ms=total_dur, text=text)
        est_res = truncate_to_heard(
            [ClauseAudio(text=text, duration_ms=total_dur, words=(), lang=lang)],
            played_ms=cut_ms,
            rate_est=est,
        )

        error = abs(len(est_res.words) - len(gt_res.words))
        errors.append(error)
        assert error <= 1, (
            f"Case {i} ({lang}) exceeded INV-T1 error bound: "
            f"cut_ms={cut_ms}, gt_words={len(gt_res.words)}, est_words={len(est_res.words)}"
        )

    assert len(errors) == 200
    assert max(errors) <= 1
