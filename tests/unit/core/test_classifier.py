"""Unit tests and property tests for OverlapClassifier (§7, §10, INV-L4, INV-L11).

Tests every failure mode, rule order, early-exit latency, acoustic priors,
revisability, and property invariants.
"""

from __future__ import annotations

import copy

from hypothesis import given, settings
from hypothesis import strategies as st

from overlap.core.classifier import classify
from overlap.core.types import Label, OverlapFeatures, Tuning

# =============================================================================
# Failure Mode 1 & INV-L4: Never PENDING at or after classify_cap_ms
# =============================================================================


@given(
    overlap_ms=st.integers(min_value=400, max_value=2000),
    user_speaking=st.booleans(),
    rel_loudness_db=st.floats(min_value=-30.0, max_value=30.0),
    text=st.one_of(
        st.none(),
        st.sampled_from(["", "haan", "ruko", "lekin", "kya", "random content", "foo bar baz"]),
    ),
    conf=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    lang=st.one_of(st.none(), st.sampled_from(["hi", "en", "ta", "te", "bn"])),
    failsafe_bias=st.sampled_from([Label.INTERRUPT, Label.BACKCHANNEL]),
)
@settings(max_examples=50, deadline=None, derandomize=True)
def test_property_never_pending_at_or_after_cap(
    overlap_ms: int,
    user_speaking: bool,
    rel_loudness_db: float,
    text: str | None,
    conf: float | None,
    lang: str | None,
    failsafe_bias: Label,
) -> None:
    """At overlap_ms >= classify_cap_ms, classify() must NEVER return PENDING (INV-L4)."""
    tuning = Tuning(failsafe_bias=failsafe_bias)
    features = OverlapFeatures(
        overlap_ms=overlap_ms,
        user_speaking=user_speaking,
        rel_loudness_db=rel_loudness_db,
        partial_text=text,
        stt_conf=conf,
        lang=lang,
    )
    verdict = classify(features, tuning)
    assert verdict.label != Label.PENDING
    assert verdict.label in (Label.INTERRUPT, Label.BACKCHANNEL, Label.SIDE_TALK)


def test_hard_cap_failsafe_bias_respected() -> None:
    """When hard cap forces fail-safe on still-speaking speech with no text, bias is respected."""
    features = OverlapFeatures(
        overlap_ms=400,
        user_speaking=True,
        rel_loudness_db=0.0,
        partial_text=None,
    )

    # Default bias: INTERRUPT
    v_default = classify(features, Tuning(failsafe_bias=Label.INTERRUPT))
    assert v_default.label == Label.INTERRUPT
    assert "cap" in v_default.reason or "failsafe" in v_default.reason

    # Flipped bias: BACKCHANNEL
    v_flipped = classify(features, Tuning(failsafe_bias=Label.BACKCHANNEL))
    assert v_flipped.label == Label.BACKCHANNEL


def test_hard_cap_speech_ended_defaults_to_backchannel() -> None:
    """At hard cap, if user stopped speaking with no content/stop, defaults to BACKCHANNEL."""
    features = OverlapFeatures(
        overlap_ms=400,
        user_speaking=False,
        rel_loudness_db=0.0,
        partial_text=None,
    )
    verdict = classify(features, Tuning())
    assert verdict.label == Label.BACKCHANNEL


# =============================================================================
# Failure Mode 2: Early exit before classify_min_ms (250 ms)
# =============================================================================


def test_early_exit_stop_word() -> None:
    """Stop words decide INTERRUPT immediately well before 250 ms."""
    for stop_word in ["ruko", "wait", "stop", "nahi", "bas", "ek minute"]:
        features = OverlapFeatures(
            overlap_ms=60,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text=stop_word,
        )
        verdict = classify(features, Tuning())
        assert verdict.label == Label.INTERRUPT, f"Expected early INTERRUPT for '{stop_word}'"
        assert "stop" in verdict.reason


def test_early_exit_contrast_word() -> None:
    """Contrast words decide INTERRUPT immediately well before 250 ms."""
    for contrast_word in ["lekin", "par", "magar", "but", "aanaal"]:
        features = OverlapFeatures(
            overlap_ms=80,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text=contrast_word,
        )
        verdict = classify(features, Tuning())
        assert verdict.label == Label.INTERRUPT, f"Expected early INTERRUPT for '{contrast_word}'"
        assert "contrast" in verdict.reason


def test_early_exit_question() -> None:
    """Question words or punctuation decide INTERRUPT immediately before 250 ms."""
    for q in ["kya", "why?", "where is it", "kab aayega?"]:
        features = OverlapFeatures(
            overlap_ms=100,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text=q,
        )
        verdict = classify(features, Tuning())
        assert verdict.label == Label.INTERRUPT, f"Expected early INTERRUPT for '{q}'"


def test_early_exit_multiple_content_tokens() -> None:
    """Two or more content tokens trigger early INTERRUPT before 250 ms."""
    features = OverlapFeatures(
        overlap_ms=100,
        user_speaking=True,
        rel_loudness_db=0.0,
        partial_text="flight ticket",
    )
    verdict = classify(features, Tuning())
    assert verdict.label == Label.INTERRUPT
    assert "content" in verdict.reason


def test_single_content_token_waits_until_min_overlap() -> None:
    """Single content token returns PENDING before content_min_overlap_ms (250 ms)."""
    tuning = Tuning(content_min_overlap_ms=250)
    feat_early = OverlapFeatures(
        overlap_ms=120,
        user_speaking=True,
        rel_loudness_db=0.0,
        partial_text="ticket",
    )
    v_early = classify(feat_early, tuning)
    assert v_early.label == Label.PENDING

    feat_late = OverlapFeatures(
        overlap_ms=260,
        user_speaking=True,
        rel_loudness_db=0.0,
        partial_text="ticket",
    )
    v_late = classify(feat_late, tuning)
    assert v_late.label == Label.INTERRUPT


# =============================================================================
# Failure Mode 3: Pure backchannel escalation
# =============================================================================


def test_pure_backchannel_escalates_on_long_overlap() -> None:
    """Pure backchannel exceeding bc_max_overlap_ms escalates to INTERRUPT."""
    tuning = Tuning(bc_max_overlap_ms=600)
    features = OverlapFeatures(
        overlap_ms=650,
        user_speaking=True,
        rel_loudness_db=0.0,
        partial_text="theek hai ji",
    )
    verdict = classify(features, tuning)
    assert verdict.label == Label.INTERRUPT
    assert "overlap" in verdict.reason


def test_pure_backchannel_escalates_on_token_count_exceeded() -> None:
    """Pure backchannel exceeding bc_max_tokens escalates to INTERRUPT."""
    tuning = Tuning(bc_max_tokens=3)
    # 4 backchannel tokens
    features = OverlapFeatures(
        overlap_ms=350,
        user_speaking=True,
        rel_loudness_db=0.0,
        partial_text="haan haan haan haan",
    )
    verdict = classify(features, tuning)
    assert verdict.label == Label.INTERRUPT
    assert "token" in verdict.reason


def test_pure_backchannel_pending_before_bc_min_ms_while_speaking() -> None:
    """Pure backchannel while still speaking returns PENDING before bc_min_ms (200 ms)."""
    tuning = Tuning(bc_min_ms=200)
    features = OverlapFeatures(
        overlap_ms=80,
        user_speaking=True,
        rel_loudness_db=0.0,
        partial_text="haan",
    )
    verdict = classify(features, tuning)
    assert verdict.label == Label.PENDING


def test_pure_backchannel_decided_when_speech_ended_early() -> None:
    """Pure backchannel is decided immediately if speech ended even before bc_min_ms."""
    tuning = Tuning(bc_min_ms=200)
    features = OverlapFeatures(
        overlap_ms=120,
        user_speaking=False,
        rel_loudness_db=0.0,
        partial_text="haan",
    )
    verdict = classify(features, tuning)
    assert verdict.label == Label.BACKCHANNEL


# =============================================================================
# Failure Mode 4: No partial yet (STT latency) -> acoustic prior
# =============================================================================


def test_no_partial_pending_before_classify_min_ms() -> None:
    """With no partial yet, returns PENDING before classify_min_ms."""
    features = OverlapFeatures(
        overlap_ms=100,
        user_speaking=True,
        rel_loudness_db=0.0,
        partial_text=None,
    )
    verdict = classify(features, Tuning())
    assert verdict.label == Label.PENDING


def test_no_partial_short_overlap_speech_ended_is_backchannel() -> None:
    """Acoustic prior: after classify_min_ms, short overlap + speech ended -> BACKCHANNEL."""
    features = OverlapFeatures(
        overlap_ms=260,
        user_speaking=False,
        rel_loudness_db=0.0,
        partial_text=None,
    )
    verdict = classify(features, Tuning())
    assert verdict.label == Label.BACKCHANNEL
    assert "acoustic" in verdict.reason


def test_no_partial_loud_user_speaking_is_interrupt() -> None:
    """Acoustic prior: after classify_min_ms, rel_loudness_db >= assertive_db -> INTERRUPT."""
    tuning = Tuning(assertive_db=3.0)
    features = OverlapFeatures(
        overlap_ms=260,
        user_speaking=True,
        rel_loudness_db=4.5,
        partial_text=None,
    )
    verdict = classify(features, tuning)
    assert verdict.label == Label.INTERRUPT
    assert "assertive" in verdict.reason or "acoustic" in verdict.reason


# =============================================================================
# Failure Mode 5: Loud sustained user speech with empty/garbage partial
# =============================================================================


def test_loud_sustained_speech_with_garbage_text_is_interrupt() -> None:
    """Loud sustained user speech with unknown tokens must not be treated as backchannel."""
    tuning = Tuning(assertive_db=3.0, classify_cap_ms=400)
    features = OverlapFeatures(
        overlap_ms=400,
        user_speaking=True,
        rel_loudness_db=5.0,
        partial_text="xgqwb kjashd",
    )
    verdict = classify(features, tuning)
    assert verdict.label == Label.INTERRUPT


def test_sustained_speech_with_empty_text_reaches_interrupt_at_cap() -> None:
    """Sustained user speech with empty partial text escalates to INTERRUPT at hard cap."""
    features = OverlapFeatures(
        overlap_ms=400,
        user_speaking=True,
        rel_loudness_db=0.0,
        partial_text="",
    )
    verdict = classify(features, Tuning())
    assert verdict.label == Label.INTERRUPT


# =============================================================================
# Failure Mode 6: Side-talk rule safety
# =============================================================================


def test_side_talk_requires_low_loudness_and_low_confidence() -> None:
    """Side-talk triggers only when quiet (<= sidetalk_db) AND low stt_conf AND no lexicon hit."""
    tuning = Tuning(sidetalk_db=-9.0)
    features = OverlapFeatures(
        overlap_ms=280,
        user_speaking=True,
        rel_loudness_db=-12.0,
        stt_conf=0.25,
        partial_text="ek glass paani",
    )
    verdict = classify(features, tuning)
    assert verdict.label == Label.SIDE_TALK
    assert "sidetalk" in verdict.reason


def test_quiet_genuine_interruption_not_swallowed_as_sidetalk() -> None:
    """Quiet interruption with high STT confidence is INTERRUPT, not SIDE_TALK."""
    tuning = Tuning(sidetalk_db=-9.0)
    features = OverlapFeatures(
        overlap_ms=280,
        user_speaking=True,
        rel_loudness_db=-11.0,
        stt_conf=0.92,
        partial_text="flight cancel kar do",
    )
    verdict = classify(features, tuning)
    assert verdict.label == Label.INTERRUPT


def test_quiet_stop_word_not_swallowed_as_sidetalk() -> None:
    """Stop word at low volume and low confidence is still INTERRUPT."""
    tuning = Tuning(sidetalk_db=-9.0)
    features = OverlapFeatures(
        overlap_ms=100,
        user_speaking=True,
        rel_loudness_db=-12.0,
        stt_conf=0.3,
        partial_text="ruko",
    )
    verdict = classify(features, tuning)
    assert verdict.label == Label.INTERRUPT


# =============================================================================
# Failure Mode 7: Determinism
# =============================================================================


def test_deterministic_output_for_identical_features() -> None:
    """Classify returns the exact same Verdict for identical features."""
    features = OverlapFeatures(
        overlap_ms=300,
        user_speaking=True,
        rel_loudness_db=1.5,
        partial_text="haan ji",
        stt_conf=0.88,
        lang="hi",
    )
    tuning = Tuning()
    v1 = classify(features, tuning, decided_at_ns=1000)
    v2 = classify(copy.deepcopy(features), copy.deepcopy(tuning), decided_at_ns=1000)
    assert v1 == v2


# =============================================================================
# Revisability: Verdict updates as new partials arrive
# =============================================================================


def test_verdict_revisability() -> None:
    """Verdict transitions from PENDING -> BACKCHANNEL -> INTERRUPT as partial evolves."""
    tuning = Tuning()

    # t1: 100 ms, 'haan' arrived but user still speaking -> PENDING (accumulating)
    f1 = OverlapFeatures(
        overlap_ms=100,
        user_speaking=True,
        rel_loudness_db=0.0,
        partial_text="haan",
    )
    v1 = classify(f1, tuning)
    assert v1.label == Label.PENDING

    # t2: 220 ms, user spoke 'haan' (pure bc >= bc_min_ms) -> BACKCHANNEL
    f2 = OverlapFeatures(
        overlap_ms=220,
        user_speaking=True,
        rel_loudness_db=0.0,
        partial_text="haan",
    )
    v2 = classify(f2, tuning)
    assert v2.label == Label.BACKCHANNEL

    # t3: 320 ms, user continued with contrast word 'haan lekin ruko' -> INTERRUPT (escalation)
    f3 = OverlapFeatures(
        overlap_ms=320,
        user_speaking=True,
        rel_loudness_db=0.0,
        partial_text="haan lekin ruko",
    )
    v3 = classify(f3, tuning)
    assert v3.label == Label.INTERRUPT
