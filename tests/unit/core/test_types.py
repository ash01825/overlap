"""Unit tests for core dataclasses, enums, events, effects, and Tuning."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from overlap.core.types import (
    AbortLlm,
    AgentAudioDrained,
    ArmTimer,
    AudioFrame,
    BackchannelDue,
    CancelResponse,
    CancelTimer,
    CancelTts,
    ClipDone,
    CloseOverlapWindow,
    EndOfTurn,
    FadeOutClip,
    FirstAudioReady,
    FlushPlayout,
    FsmState,
    Label,
    MergeUserTurn,
    OpenOverlapWindow,
    OverlapFeatures,
    PlayClip,
    SetGain,
    StartPlayout,
    StartResponse,
    SttEvent,
    SttPartial,
    Tag,
    TelemetryEffect,
    TimerFired,
    TransportError,
    TruncateHistory,
    Tuning,
    VadOffset,
    VadOnset,
    Verdict,
    WordTiming,
    YieldCompleted,
)


def _instantiate_dummy(cls: type[Any]) -> Any:
    """Instantiate a class with dummy arguments based on its fields."""
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            continue
        if field.default_factory is not dataclasses.MISSING:
            continue
        # Provide type-compatible dummies
        if "pcm" in field.name:
            kwargs[field.name] = b"\x00" * 320
        elif field.type in (int, "int"):
            kwargs[field.name] = 0
        elif field.type in (float, "float"):
            kwargs[field.name] = 0.0
        elif field.type in (str, "str"):
            kwargs[field.name] = "dummy"
        elif field.type in (bool, "bool"):
            kwargs[field.name] = False
        elif field.name == "final":
            kwargs[field.name] = SttEvent(kind="final", text="test", lang=None, conf=None, ts_ns=0)
        elif field.name == "label":
            kwargs[field.name] = Label.INTERRUPT
        elif field.name == "kind":
            kwargs[field.name] = "final"
        elif field.name == "tags":
            kwargs[field.name] = Tag.NONE
        elif field.name == "words":
            kwargs[field.name] = ()
        else:
            kwargs[field.name] = None
    return cls(**kwargs)


ALL_DATACLASSES: list[type[Any]] = [
    AudioFrame,
    WordTiming,
    SttEvent,
    Verdict,
    OverlapFeatures,
    Tuning,
    # Events
    EndOfTurn,
    BackchannelDue,
    ClipDone,
    FirstAudioReady,
    VadOnset,
    VadOffset,
    AgentAudioDrained,
    SttPartial,
    TimerFired,
    YieldCompleted,
    TransportError,
    # Effects
    SetGain,
    FlushPlayout,
    CancelTts,
    AbortLlm,
    TruncateHistory,
    ArmTimer,
    CancelTimer,
    PlayClip,
    StartResponse,
    OpenOverlapWindow,
    CloseOverlapWindow,
    FadeOutClip,
    StartPlayout,
    CancelResponse,
    MergeUserTurn,
    TelemetryEffect,
]


@pytest.mark.parametrize("cls", ALL_DATACLASSES)
def test_dataclasses_are_frozen_and_have_slots(cls: type[Any]) -> None:
    """Ensure every core dataclass is frozen and has slots (failure mode: mutable leaks)."""
    assert dataclasses.is_dataclass(cls), f"{cls.__name__} must be a dataclass"
    assert hasattr(cls, "__slots__"), f"{cls.__name__} must have __slots__"

    instance = _instantiate_dummy(cls)
    assert not hasattr(instance, "__dict__"), f"{cls.__name__} instance must not have __dict__"

    # Mutating an existing or new field must raise FrozenInstanceError, AttributeError, or TypeError
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        setattr(instance, "some_new_attr_xyz", 123)  # noqa: B010

    fields = dataclasses.fields(cls)
    if fields:
        first_field = fields[0].name
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            setattr(instance, first_field, getattr(instance, first_field))


def test_tag_flags() -> None:
    """Test Tag bit flags."""
    assert not bool(Tag.NONE)
    assert bool(Tag.SELF_BACKCHANNEL)
    assert bool(Tag.SELF_TTS)
    combined = Tag.SELF_BACKCHANNEL | Tag.SELF_TTS
    assert Tag.SELF_BACKCHANNEL in combined
    assert Tag.SELF_TTS in combined


def test_label_enum() -> None:
    """Test Label enum values."""
    assert {lbl.name for lbl in Label} == {
        "BACKCHANNEL",
        "INTERRUPT",
        "SIDE_TALK",
        "PENDING",
    }


def test_fsm_state_enum() -> None:
    """Test FsmState enum values per ARCHITECTURE §6."""
    assert {s.name for s in FsmState} == {
        "LISTENING",
        "THINKING",
        "SPEAKING",
        "DUCKED",
        "YIELDED",
        "BACKCHANNELING",
    }


def test_tuning_defaults_match_architecture_spec() -> None:
    """Test that all Tuning defaults match ARCHITECTURE §10 exactly."""
    t = Tuning()

    # Latency & performance invariants (§10)
    assert t.frame_ms == 20  # INV-L1
    assert t.duck_gain == 0.3  # INV-L3
    assert t.duck_attack_ms == 20  # INV-L3
    assert t.duck_release_ms == 150  # INV-L8
    assert t.duck_latency_cap_ms == 40  # INV-L2
    assert t.classify_min_ms == 250  # INV-L4
    assert t.classify_cap_ms == 400  # INV-L4
    assert t.max_lookahead_ms == 60  # INV-L9
    assert t.interrupt_stop_cap_ms == 40  # INV-L5
    assert t.cancel_dispatch_cap_ms == 5  # INV-L6
    assert t.post_cancel_leak_frames == 0  # INV-L7
    assert t.bc_clip_start_ms == 50  # INV-L10
    assert t.bc_min_gap_ms == 4000  # INV-B1
    assert t.bc_max_per_min == 6  # INV-B1

    # Audio rates (§4)
    assert t.in_rate_hz == 16000
    assert t.out_rate_hz == 16000

    # Playout and truncation (§8.2, §8.3)
    assert t.assumed_downstream_ms == 120
    assert t.truncation_marker == " …"

    # Agent backchannel heuristics (§9)
    assert t.bc_min_user_turn_ms == 3000
    assert t.bc_pause_min_ms == 200
    assert t.bc_pause_max_ms == 400
    assert t.bc_energy_fall_frames == 5

    # Classifier specifics (§7.1)
    assert t.content_min_overlap_ms == 250
    assert t.recheck_window_ms == 1500
    assert t.bc_max_tokens == 3
    assert t.bc_max_overlap_ms == 600
    assert t.bc_min_ms == 200
    assert t.assertive_db == 3.0
    assert t.sidetalk_db == -9.0
    assert t.failsafe_bias == Label.INTERRUPT
    assert t.barge_in_thinking_ms == 100


def test_tuning_is_frozen() -> None:
    """Verify Tuning instances cannot be mutated at runtime."""
    t = Tuning()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        t.duck_gain = 0.5  # type: ignore[misc]


def test_audio_frame() -> None:
    """Test AudioFrame creation and defaults."""
    pcm = b"\x00" * 640
    frame = AudioFrame(pcm=pcm, sample_rate=16000, ts_ns=1_000_000)
    assert frame.pcm == pcm
    assert frame.sample_rate == 16000
    assert frame.ts_ns == 1_000_000
    assert frame.tags == Tag.NONE


def test_word_timing_and_stt_event() -> None:
    """Test WordTiming and SttEvent structures."""
    wt = WordTiming(word="theek", start_ms=0, end_ms=200)
    event = SttEvent(
        kind="partial",
        text="theek hai",
        lang="hi",
        conf=0.92,
        ts_ns=2_000_000,
        words=(wt,),
    )
    assert event.kind == "partial"
    assert event.text == "theek hai"
    assert event.lang == "hi"
    assert event.conf == 0.92
    assert len(event.words) == 1
    assert event.words[0].word == "theek"


def test_verdict_and_overlap_features() -> None:
    """Test Verdict and OverlapFeatures structures."""
    v = Verdict(
        label=Label.BACKCHANNEL,
        confidence=0.88,
        reason="lexicon_backchannel",
        overlap_ms=220,
        decided_at_ns=5_000_000,
    )
    assert v.label == Label.BACKCHANNEL
    assert v.overlap_ms == 220

    f = OverlapFeatures(
        overlap_ms=180,
        user_speaking=True,
        rel_loudness_db=-2.0,
        partial_text="haan",
        partial_age_ms=40,
        stt_conf=0.9,
        lang="hi",
        is_final=False,
    )
    assert f.overlap_ms == 180
    assert f.user_speaking is True
    assert f.partial_text == "haan"
