"""Unit and timeline tests for BackchannelPredictor (§9, §10, INV-B1).

Covers:
- Turn duration threshold (bc_min_user_turn_ms = 3000)
- Pause window gating (bc_pause_min_ms = 200 .. bc_pause_max_ms = 400)
- Pause measurement from VAD offset (never partial arrival)
- Clause-completion heuristics (punctuation, danda, terminal particles per language)
- Conjunction blacklist (trailing conjunction/comma mid-clause rejection)
- Acoustic energy falling heuristic (bc_energy_fall_frames = 5)
- Pacing invariants (INV-B1): cooldown gap >= 4 s, rate cap <= 6/min
- Clip-in-flight and FSM state gating (LISTENING only)
- Language-aware clip selection and seeded deterministic variation
- No repeat of the same clip within the last 2 triggers
- Table-driven timeline test suite (>= 20 scenarios)
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from overlap.core.predictor import (
    DEFAULT_CLIPS_BY_LANG,
    BackchannelPredictor,
    is_clause_complete,
    is_mid_clause,
)
from overlap.core.types import FsmState, SttPartial, Tuning

# =============================================================================
# Table-Driven Timeline Test Harness
# =============================================================================


@dataclass(frozen=True)
class TimelineStep:
    t_ms: int
    vad: bool
    partial: str | SttPartial | None = None
    energy_ema: float | None = None
    fsm_state: FsmState = FsmState.LISTENING
    clip_in_flight: bool | None = None


@dataclass(frozen=True)
class TimelineCase:
    id: str
    steps: tuple[TimelineStep, ...]
    expected_fires_at_ms: tuple[int, ...]
    expected_lang: str | None = None


def _make_steady_speech(
    start_ms: int, end_ms: int, step_ms: int = 50, partial: str | SttPartial | None = None
) -> list[TimelineStep]:
    """Helper to generate continuous user speech steps and transition to silence."""
    steps: list[TimelineStep] = []
    for t in range(start_ms, end_ms + 1, step_ms):
        steps.append(TimelineStep(t_ms=t, vad=True, partial=partial if t == end_ms else None))
    # Speech ends, silence onset at end_ms + 20 ms
    steps.append(TimelineStep(t_ms=end_ms + 20, vad=False))
    return steps


TIMELINE_CASES: list[TimelineCase] = [
    # 1. Happy path: Clause completion via sentence-ending punctuation (period)
    TimelineCase(
        id="happy_clause_punct_period",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="I have finished the document.")
            + [
                TimelineStep(t_ms=3320, vad=False),  # pause 100 ms (< 200 ms)
                TimelineStep(t_ms=3450, vad=False),  # pause 230 ms -> FIRE
                TimelineStep(t_ms=3500, vad=False),  # pause 280 ms -> no repeat in same pause
            ]
        ),
        expected_fires_at_ms=(3450,),
        expected_lang="en",
    ),
    # 2. Happy path: Hindi clause completion via Danda (।)
    TimelineCase(
        id="happy_hindi_clause_danda",
        steps=tuple(
            _make_steady_speech(0, 3200, partial=SttPartial(text="हमे सब कुछ मिल गया है।", lang="hi"))
            + [
                TimelineStep(t_ms=3450, vad=False),  # pause 230 ms -> FIRE
            ]
        ),
        expected_fires_at_ms=(3450,),
        expected_lang="hi",
    ),
    # 3. Happy path: Hindi clause completion via terminal particle (na)
    TimelineCase(
        id="happy_hindi_terminal_particle_na",
        steps=tuple(
            _make_steady_speech(0, 3200, partial=SttPartial(text="aap kal aaoge na", lang="hi"))
            + [
                TimelineStep(t_ms=3450, vad=False),  # pause 230 ms -> FIRE
            ]
        ),
        expected_fires_at_ms=(3450,),
        expected_lang="hi",
    ),
    # 4. Happy path: English clause completion via terminal particle (right)
    TimelineCase(
        id="happy_english_terminal_particle_right",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="we should deploy to production right")
            + [
                TimelineStep(t_ms=3450, vad=False),  # pause 230 ms -> FIRE
            ]
        ),
        expected_fires_at_ms=(3450,),
        expected_lang="en",
    ),
    # 5. Happy path: Tamil clause completion via terminal particle (nga)
    TimelineCase(
        id="happy_tamil_terminal_particle_nga",
        steps=tuple(
            _make_steady_speech(0, 3200, partial=SttPartial(text="naan solren nga", lang="ta"))
            + [
                TimelineStep(t_ms=3450, vad=False),  # pause 230 ms -> FIRE
            ]
        ),
        expected_fires_at_ms=(3450,),
        expected_lang="ta",
    ),
    # 6. Happy path: Energy falling for >= 5 frames without punctuation/particle
    TimelineCase(
        id="happy_energy_fall_heuristic",
        steps=tuple(
            _make_steady_speech(0, 3100, partial="the whole team was working together")
            + [
                TimelineStep(t_ms=3150, vad=False, energy_ema=-20.0),
                TimelineStep(t_ms=3200, vad=False, energy_ema=-25.0),
                TimelineStep(t_ms=3250, vad=False, energy_ema=-30.0),
                TimelineStep(t_ms=3300, vad=False, energy_ema=-35.0),
                TimelineStep(t_ms=3350, vad=False, energy_ema=-40.0),  # 5 falling frames
                TimelineStep(t_ms=3450, vad=False, energy_ema=-45.0),  # pause 330 ms -> FIRE
            ]
        ),
        expected_fires_at_ms=(3450,),
        expected_lang="en",
    ),
    # 7. Fail: Turn too short (< 3000 ms) - 1 second turn
    TimelineCase(
        id="fail_turn_too_short_1s",
        steps=tuple(
            _make_steady_speech(0, 1000, partial="yes that is correct.")
            + [
                TimelineStep(t_ms=1250, vad=False),  # pause 230 ms, but turn 1250 ms < 3000 ms
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 8. Fail: Turn too short (< 3000 ms) - 2800 ms turn
    TimelineCase(
        id="fail_turn_too_short_2800ms",
        steps=tuple(
            _make_steady_speech(0, 2500, partial="I am almost finished with this explanation.")
            + [
                TimelineStep(t_ms=2750, vad=False),  # pause 230 ms, turn 2750 ms < 3000 ms
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 9. Fail: Pause too short (< 200 ms)
    TimelineCase(
        id="fail_pause_too_short_100ms",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="all tasks are complete.")
            + [
                TimelineStep(t_ms=3320, vad=False),  # pause 100 ms (< 200 ms) -> NO FIRE
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 10. Fail: Pause too long (> 400 ms)
    TimelineCase(
        id="fail_pause_too_long_450ms",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="all tasks are complete.")
            + [
                TimelineStep(t_ms=3670, vad=False),  # pause 450 ms (> 400 ms) -> NO FIRE
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 11. Fail: Mid-clause trailing comma
    TimelineCase(
        id="fail_mid_clause_trailing_comma",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="we were reviewing the results,")
            + [
                TimelineStep(t_ms=3450, vad=False),  # pause 230 ms -> NO FIRE (mid-clause)
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 12. Fail: Mid-clause English conjunction 'and'
    TimelineCase(
        id="fail_mid_clause_conjunction_and",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="we went to the office and")
            + [
                TimelineStep(t_ms=3450, vad=False),  # pause 230 ms -> NO FIRE
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 13. Fail: Mid-clause English conjunction 'but'
    TimelineCase(
        id="fail_mid_clause_conjunction_but",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="we tried to fix it but")
            + [
                TimelineStep(t_ms=3450, vad=False),  # pause 230 ms -> NO FIRE
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 14. Fail: Mid-clause Hindi conjunction 'aur'
    TimelineCase(
        id="fail_mid_clause_conjunction_aur",
        steps=tuple(
            _make_steady_speech(
                0, 3200, partial=SttPartial(text="hum market gaye the aur", lang="hi")
            )
            + [
                TimelineStep(t_ms=3450, vad=False),  # pause 230 ms -> NO FIRE
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 15. Fail: Mid-clause Hindi conjunction 'lekin'
    TimelineCase(
        id="fail_mid_clause_conjunction_lekin",
        steps=tuple(
            _make_steady_speech(
                0, 3200, partial=SttPartial(text="maine call kiya lekin", lang="hi")
            )
            + [
                TimelineStep(t_ms=3450, vad=False),  # pause 230 ms -> NO FIRE
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 16. Fail: Trailing conjunction overrides energy fall (must not cut off mid-clause)
    TimelineCase(
        id="fail_conjunction_overrides_energy_fall",
        steps=tuple(
            _make_steady_speech(0, 3100, partial="I could not submit because")
            + [
                TimelineStep(t_ms=3150, vad=False, energy_ema=-20.0),
                TimelineStep(t_ms=3200, vad=False, energy_ema=-25.0),
                TimelineStep(t_ms=3250, vad=False, energy_ema=-30.0),
                TimelineStep(t_ms=3300, vad=False, energy_ema=-35.0),
                TimelineStep(t_ms=3350, vad=False, energy_ema=-40.0),  # energy fell
                TimelineStep(t_ms=3450, vad=False, energy_ema=-45.0),  # "because" -> NO FIRE
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 17. Fail: Cooldown violation (gap < 4000 ms)
    TimelineCase(
        id="fail_cooldown_gap_under_4s",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="the first topic is done.")
            + [
                TimelineStep(t_ms=3450, vad=False),  # FIRE 1 at 3450 ms
            ]
            + _make_steady_speech(3700, 5800, partial="now onto the second topic.")
            + [
                TimelineStep(t_ms=6050, vad=False),  # gap 2600 ms < 4000 ms -> NO FIRE
            ]
        ),
        expected_fires_at_ms=(3450,),
    ),
    # 18. Happy path: Cooldown respected (gap >= 4000 ms)
    TimelineCase(
        id="happy_cooldown_gap_over_4s",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="the first topic is done.")
            + [
                TimelineStep(t_ms=3450, vad=False),  # FIRE 1 at 3450 ms
            ]
            + _make_steady_speech(3700, 7500, partial="now onto the second topic.")
            + [
                TimelineStep(t_ms=7750, vad=False),  # gap 4300 ms >= 4000 ms -> FIRE 2
            ]
        ),
        expected_fires_at_ms=(3450, 7750),
    ),
    # 19. Fail: Clip in flight flag blocks trigger
    TimelineCase(
        id="fail_clip_in_flight",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="everything is ready.")
            + [
                TimelineStep(t_ms=3450, vad=False, clip_in_flight=True),  # clip playing -> NO FIRE
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 20. Fail: FSM in SPEAKING state
    TimelineCase(
        id="fail_fsm_speaking",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="everything is ready.")
            + [
                TimelineStep(t_ms=3450, vad=False, fsm_state=FsmState.SPEAKING),  # NO FIRE
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 21. Fail: FSM in THINKING state
    TimelineCase(
        id="fail_fsm_thinking",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="everything is ready.")
            + [
                TimelineStep(t_ms=3450, vad=False, fsm_state=FsmState.THINKING),  # NO FIRE
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 22. Fail: FSM in DUCKED state
    TimelineCase(
        id="fail_fsm_ducked",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="everything is ready.")
            + [
                TimelineStep(t_ms=3450, vad=False, fsm_state=FsmState.DUCKED),  # NO FIRE
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 23. Fail: FSM in BACKCHANNELING state
    TimelineCase(
        id="fail_fsm_backchanneling",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="everything is ready.")
            + [
                TimelineStep(t_ms=3450, vad=False, fsm_state=FsmState.BACKCHANNELING),  # NO FIRE
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 24. Fail: Incomplete sentence with NO falling energy
    TimelineCase(
        id="fail_no_clause_and_no_energy_fall",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="we were looking at the quarterly")
            + [
                TimelineStep(t_ms=3450, vad=False, energy_ema=-20.0),  # no fall -> NO FIRE
            ]
        ),
        expected_fires_at_ms=(),
    ),
    # 25. Timing correctness: Pause measured from VAD offset, NOT partial arrival
    TimelineCase(
        id="timing_vad_offset_vs_partial_arrival",
        steps=(
            TimelineStep(t_ms=0, vad=True),
            TimelineStep(t_ms=3200, vad=True),
            TimelineStep(t_ms=3220, vad=False),  # VAD offset at 3220 ms
            TimelineStep(t_ms=3380, vad=False, partial="I have sent the email."),
            TimelineStep(t_ms=3470, vad=False),  # pause since VAD = 250 ms -> FIRE
        ),
        expected_fires_at_ms=(3470,),
        expected_lang="en",
    ),
    # 26. Timing correctness: Partial arrives early while user is still speaking
    TimelineCase(
        id="timing_partial_arrives_before_vad_offset",
        steps=(
            TimelineStep(t_ms=0, vad=True),
            TimelineStep(t_ms=3000, vad=True, partial="I have sent the email."),
            TimelineStep(t_ms=3300, vad=True),
            TimelineStep(t_ms=3320, vad=False),  # VAD offset at 3320 ms
            TimelineStep(t_ms=3350, vad=False),  # pause since VAD is 30 ms < 200 ms -> NO FIRE
        ),
        expected_fires_at_ms=(),
    ),
    # 27. Telugu clause completion via terminal particle (andi)
    TimelineCase(
        id="happy_telugu_terminal_particle_andi",
        steps=tuple(
            _make_steady_speech(0, 3200, partial=SttPartial(text="nenu cheppanu andi", lang="te"))
            + [
                TimelineStep(t_ms=3450, vad=False),  # pause 230 ms -> FIRE
            ]
        ),
        expected_fires_at_ms=(3450,),
        expected_lang="te",
    ),
    # 28. Bengali clause completion via terminal particle (tai na)
    TimelineCase(
        id="happy_bengali_terminal_particle_tai_na",
        steps=tuple(
            _make_steady_speech(
                0, 3200, partial=SttPartial(text="shob thik ache tai na", lang="bn")
            )
            + [
                TimelineStep(t_ms=3450, vad=False),  # pause 230 ms -> FIRE
            ]
        ),
        expected_fires_at_ms=(3450,),
        expected_lang="bn",
    ),
    # 29. Fire only once per pause window
    TimelineCase(
        id="fire_only_once_per_pause_window",
        steps=tuple(
            _make_steady_speech(0, 3200, partial="this is the completed statement.")
            + [
                TimelineStep(t_ms=3440, vad=False),  # pause 220 ms -> FIRE 1
                TimelineStep(t_ms=3460, vad=False),  # pause 240 ms -> NO FIRE
                TimelineStep(t_ms=3500, vad=False),  # pause 280 ms -> NO FIRE
                TimelineStep(t_ms=3570, vad=False),  # pause 350 ms -> NO FIRE
            ]
        ),
        expected_fires_at_ms=(3440,),
    ),
]


@pytest.mark.parametrize("case", TIMELINE_CASES, ids=lambda c: c.id)
def test_timeline_cases(case: TimelineCase) -> None:
    """Execute table-driven timeline cases and assert exact fire timestamps."""
    predictor = BackchannelPredictor(tuning=Tuning(), seed=42)
    fired_times: list[int] = []

    for step in case.steps:
        now_ns = step.t_ms * 1_000_000
        due = predictor.update(
            now_ns=now_ns,
            vad_state=step.vad,
            partial=step.partial,
            energy_ema=step.energy_ema,
            fsm_state=step.fsm_state,
            clip_in_flight=step.clip_in_flight,
        )
        if due is not None:
            fired_times.append(step.t_ms)
            if case.expected_lang is not None:
                assert due.lang == case.expected_lang

    assert tuple(fired_times) == case.expected_fires_at_ms, (
        f"Case {case.id} failed: expected fires at {case.expected_fires_at_ms}, got {fired_times}"
    )


# =============================================================================
# Specific Invariant and Unit Tests
# =============================================================================


def test_rate_cap_6_per_minute() -> None:
    """Assert INV-B1: maximum 6 backchannels in any 60-second sliding window."""
    predictor = BackchannelPredictor(tuning=Tuning(bc_min_gap_ms=4000, bc_max_per_min=6), seed=42)

    fired_count = 0
    current_ms = 0
    for _ in range(12):
        # 4500 ms speaking
        for _ in range(9):
            current_ms += 500
            predictor.update(
                now_ns=current_ms * 1_000_000,
                vad_state=True,
                partial="we are explaining the full architecture.",
            )
        # Speech stops, silence onset
        current_ms += 20
        predictor.update(now_ns=current_ms * 1_000_000, vad_state=False)

        # 230 ms pause
        current_ms += 230
        due = predictor.update(
            now_ns=current_ms * 1_000_000,
            vad_state=False,
        )
        if due is not None:
            fired_count += 1
            predictor.notify_clip_done()

        current_ms += 250  # finish pause

    # Even though 12 opportunities occurred with gap > 4s, at most 6 can fire in 60s
    assert fired_count == 6


def test_language_aware_clip_selection_and_no_repeat() -> None:
    """Assert language-aware clip selection and no repeat of same clip in last 2."""
    langs = ("en", "hi", "ta", "te", "kn", "ml", "bn", "mr", "gu", "pa", "or")

    for lang in langs:
        predictor = BackchannelPredictor(seed=123)
        clips_seen: list[str] = []

        now_ms = 0
        for _ in range(20):
            # Speech onset at now_ms
            predictor.update(now_ns=now_ms * 1_000_000, vad_state=True)
            # Speech ongoing for 3500 ms until now_ms + 3500
            now_ms += 3500
            predictor.update(
                now_ns=now_ms * 1_000_000,
                vad_state=True,
                partial=SttPartial(text="valid completed sentence.", lang=lang),
            )
            # Silence onset at +20 ms
            now_ms += 20
            predictor.update(now_ns=now_ms * 1_000_000, vad_state=False)

            # 230 ms into pause
            now_ms += 230
            due = predictor.update(
                now_ns=now_ms * 1_000_000,
                vad_state=False,
            )
            assert due is not None, f"Expected fire for lang {lang}"
            assert due.lang == lang
            assert due.clip in DEFAULT_CLIPS_BY_LANG[lang]

            clips_seen.append(due.clip)
            predictor.notify_clip_done()

            # Advance 7000 ms to satisfy cooldown and sliding rate cap (<= 6/min)
            now_ms += 7000

        # Assert no clip repeated within the last 2 triggers
        for i in range(2, len(clips_seen)):
            assert clips_seen[i] != clips_seen[i - 1], (
                f"Lang {lang}: Clip repeated immediately at index {i}: {clips_seen[i]}"
            )
            assert clips_seen[i] != clips_seen[i - 2], (
                f"Lang {lang}: Clip repeated within 2 at index {i}: {clips_seen[i]}"
            )


def test_seeded_determinism() -> None:
    """Assert identical seeds yield identical clip sequences across predictors."""
    pred1 = BackchannelPredictor(seed=42)
    pred2 = BackchannelPredictor(seed=42)
    pred3 = BackchannelPredictor(seed=9999)

    seq1: list[str] = []
    seq2: list[str] = []
    seq3: list[str] = []

    now_ms = 0
    for _ in range(10):
        # Speech onset at now_ms
        for p in (pred1, pred2, pred3):
            p.update(now_ns=now_ms * 1_000_000, vad_state=True)

        now_ms += 3500
        for p in (pred1, pred2, pred3):
            p.update(now_ns=now_ms * 1_000_000, vad_state=True, partial="test clause.")
            p.update(now_ns=(now_ms + 20) * 1_000_000, vad_state=False)

        now_ms += 250
        d1 = pred1.update(now_ns=now_ms * 1_000_000, vad_state=False)
        d2 = pred2.update(now_ns=now_ms * 1_000_000, vad_state=False)
        d3 = pred3.update(now_ns=now_ms * 1_000_000, vad_state=False)

        assert d1 is not None and d2 is not None and d3 is not None
        seq1.append(d1.clip)
        seq2.append(d2.clip)
        seq3.append(d3.clip)

        for p in (pred1, pred2, pred3):
            p.notify_clip_done()
        now_ms += 7000

    assert seq1 == seq2, "Identical seeds must generate identical clip sequences"
    assert seq1 != seq3, "Different seeds should generate different sequences"


def test_conjunction_blacklist_unit() -> None:
    """Verify is_mid_clause detects trailing conjunctions and commas across languages."""
    # Mid-clause cases
    assert is_mid_clause("I went home, and")
    assert is_mid_clause("I went home and")
    assert is_mid_clause("we wanted to go but")
    assert is_mid_clause("because")
    assert is_mid_clause("let's see,")
    assert is_mid_clause("hum gaye the aur")
    assert is_mid_clause("usne bola lekin")
    assert is_mid_clause("aanaal", lang="ta")
    assert is_mid_clause("mariyu", lang="te")

    # Complete clause cases
    assert not is_mid_clause("I went home.")
    assert not is_mid_clause("hum gaye the।")
    assert not is_mid_clause("aap kal aaoge na")
    assert not is_mid_clause("done")


def test_clause_complete_heuristics_unit() -> None:
    """Verify is_clause_complete detects punctuation, dandas, and terminal particles."""
    assert is_clause_complete("This is finished.")
    assert is_clause_complete("Is this ready?")
    assert is_clause_complete("Awesome!")
    assert is_clause_complete("सब कुछ ठीक है।")
    assert is_clause_complete("आप आएंगे ना", lang="hi")
    assert is_clause_complete("you understand right", lang="en")
    assert is_clause_complete("naan varen nga", lang="ta")

    # Incomplete
    assert not is_clause_complete("we were discussing")
    assert not is_clause_complete("I went to the store and")
    assert not is_clause_complete("just a moment,")
