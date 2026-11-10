"""Core types, enums, events, effects, and Tuning constants for OVERLAP.

Pure Python stdlib only. No asyncio, no I/O, no time.time(), no network.
Every dataclass is frozen with slots to prevent accidental mutation or leaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, Flag
from typing import Any, Literal


class Tag(Flag):
    """Audio frame ownership and metadata markers."""

    NONE = 0
    SELF_BACKCHANNEL = 1
    SELF_TTS = 2


class Label(Enum):
    """Overlap classification verdict labels."""

    BACKCHANNEL = "BACKCHANNEL"
    INTERRUPT = "INTERRUPT"
    SIDE_TALK = "SIDE_TALK"
    PENDING = "PENDING"


class FsmState(Enum):
    """Finite State Machine states per ARCHITECTURE §6."""

    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    DUCKED = "DUCKED"
    YIELDED = "YIELDED"
    BACKCHANNELING = "BACKCHANNELING"


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """20 ms int16 mono audio frame."""

    pcm: bytes
    sample_rate: int
    ts_ns: int
    tags: Tag = Tag.NONE


@dataclass(frozen=True, slots=True)
class WordTiming:
    """Word boundary relative to the start of clause audio."""

    word: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class SttEvent:
    """Speech-to-text event (partial or final transcript)."""

    kind: Literal["partial", "final"]
    text: str
    lang: str | None = None
    conf: float | None = None
    ts_ns: int = 0
    words: tuple[WordTiming, ...] = ()


@dataclass(frozen=True, slots=True)
class Verdict:
    """Classification verdict for overlapping speech."""

    label: Label
    confidence: float
    reason: str
    overlap_ms: int
    decided_at_ns: int


@dataclass(frozen=True, slots=True)
class OverlapFeatures:
    """Acoustic, lexical, and timing features feeding OverlapClassifier."""

    overlap_ms: int
    user_speaking: bool
    rel_loudness_db: float
    partial_text: str | None = None
    partial_age_ms: int | None = None
    stt_conf: float | None = None
    lang: str | None = None
    is_final: bool = False


# =============================================================================
# FSM Events (Inputs)
# =============================================================================


@dataclass(frozen=True, slots=True)
class EndOfTurn:
    """Final user transcript received indicating turn completion."""

    final: SttEvent


@dataclass(frozen=True, slots=True)
class BackchannelDue:
    """Agent backchannel predictor decided a backchannel is due."""

    clip: str
    lang: str | None = None


@dataclass(frozen=True, slots=True)
class ClipDone:
    """Playout of backchannel audio clip finished."""


@dataclass(frozen=True, slots=True)
class FirstAudioReady:
    """First synthesized TTS audio frame ready for playout."""


@dataclass(frozen=True, slots=True)
class VadOnset:
    """Voice activity detected on user input."""

    ts_ns: int = 0
    is_echo: bool = False
    tag: Tag = Tag.NONE


@dataclass(frozen=True, slots=True)
class VadOffset:
    """Voice activity ceased on user input."""

    ts_ns: int = 0


@dataclass(frozen=True, slots=True)
class AgentAudioDrained:
    """Playout mixer drained all agent audio."""


@dataclass(frozen=True, slots=True)
class SttPartial:
    """Incremental partial transcript from STT stream."""

    text: str = ""
    conf: float | None = None
    lang: str | None = None
    words: tuple[WordTiming, ...] = ()
    ts_ns: int = 0


@dataclass(frozen=True, slots=True)
class TimerFired:
    """Named virtual or system timer fired."""

    name: str


@dataclass(frozen=True, slots=True)
class YieldCompleted:
    """All cancellations and flushes acknowledged after yield."""


@dataclass(frozen=True, slots=True)
class TransportError:
    """Fatal or non-fatal transport error."""

    message: str = ""


Event = (
    EndOfTurn
    | BackchannelDue
    | ClipDone
    | FirstAudioReady
    | VadOnset
    | VadOffset
    | AgentAudioDrained
    | SttPartial
    | Verdict
    | TimerFired
    | YieldCompleted
    | TransportError
)


# =============================================================================
# FSM Effects (Outputs)
# =============================================================================


@dataclass(frozen=True, slots=True)
class SetGain:
    """Adjust playout mixer gain with optional linear ramp."""

    gain: float
    ramp_ms: int = 0


@dataclass(frozen=True, slots=True)
class FlushPlayout:
    """Drop all queued playout samples and flush downstream transport."""


@dataclass(frozen=True, slots=True)
class CancelTts:
    """Cancel in-flight TTS generation stream."""


@dataclass(frozen=True, slots=True)
class AbortLlm:
    """Abort in-flight LLM inference stream."""


@dataclass(frozen=True, slots=True)
class TruncateHistory:
    """Truncate assistant conversation history to words heard by user."""

    played_pos_ms: int = 0


@dataclass(frozen=True, slots=True)
class ArmTimer:
    """Arm a named deadline timer via clock.call_at."""

    name: str
    deadline_ns: int = 0


@dataclass(frozen=True, slots=True)
class CancelTimer:
    """Disarm an active named timer."""

    name: str


@dataclass(frozen=True, slots=True)
class PlayClip:
    """Inject a pre-synthesized audio clip into playout mixer."""

    clip: str
    tag: Tag = Tag.SELF_BACKCHANNEL


@dataclass(frozen=True, slots=True)
class StartResponse:
    """Start generating a new agent response turn."""

    turn_id: str = ""
    epoch: int = 0


@dataclass(frozen=True, slots=True)
class OpenOverlapWindow:
    """Open an overlap classification window."""


@dataclass(frozen=True, slots=True)
class CloseOverlapWindow:
    """Close an active overlap classification window."""


@dataclass(frozen=True, slots=True)
class FadeOutClip:
    """Quickly fade out playing clip."""

    dur_ms: int = 40


@dataclass(frozen=True, slots=True)
class StartPlayout:
    """Begin un-ducked audio playout to transport."""


@dataclass(frozen=True, slots=True)
class CancelResponse:
    """Cancel pending response while in THINKING state."""


@dataclass(frozen=True, slots=True)
class MergeUserTurn:
    """Prepend cancelled turn text to next final transcript."""

    text: str = ""


@dataclass(frozen=True, slots=True)
class TelemetryEffect:
    """Emit telemetry metric or event."""

    name: str
    payload: tuple[tuple[str, Any], ...] = ()


Effect = (
    SetGain
    | FlushPlayout
    | CancelTts
    | AbortLlm
    | TruncateHistory
    | ArmTimer
    | CancelTimer
    | PlayClip
    | StartResponse
    | OpenOverlapWindow
    | CloseOverlapWindow
    | FadeOutClip
    | StartPlayout
    | CancelResponse
    | MergeUserTurn
    | TelemetryEffect
)


# =============================================================================
# Tuning Constants (§10)
# =============================================================================


@dataclass(frozen=True, slots=True)
class Tuning:
    """Single source of truth for all thresholds, latencies, and parameters.

    Every constant in ARCHITECTURE §10 lives here. Tests import Tuning rather
    than hardcoding literal values.
    """

    # Audio & frames (INV-L1, §4)
    frame_ms: int = 20
    in_rate_hz: int = 16000
    out_rate_hz: int = 16000

    # Ducking & Mixer (INV-L2, INV-L3, INV-L8, INV-L9)
    duck_gain: float = 0.3
    duck_attack_ms: int = 20
    duck_release_ms: int = 150
    max_lookahead_ms: int = 60
    duck_latency_cap_ms: int = 40

    # Classification & Decision (INV-L4, §7.1)
    classify_min_ms: int = 250
    classify_cap_ms: int = 400
    content_min_overlap_ms: int = 250
    recheck_window_ms: int = 1500
    bc_max_tokens: int = 3
    bc_max_overlap_ms: int = 600
    bc_min_ms: int = 200
    assertive_db: float = 3.0
    sidetalk_db: float = -9.0
    failsafe_bias: Label = Label.INTERRUPT
    barge_in_thinking_ms: int = 100
    stock_stop_ms: int = 250

    # Interruption & Cancellation (INV-L5, INV-L6, INV-L7)
    interrupt_stop_cap_ms: int = 40
    cancel_dispatch_cap_ms: int = 5
    post_cancel_leak_frames: int = 0

    # Playout Ledger & Truncation (INV-T1, §8.2, §8.3)
    assumed_downstream_ms: int = 120
    truncation_marker: str = " …"

    # Agent Backchannels (INV-B1, INV-L10, §9)
    bc_min_user_turn_ms: int = 3000
    bc_pause_min_ms: int = 200
    bc_pause_max_ms: int = 400
    bc_min_gap_ms: int = 4000
    bc_max_per_min: int = 6
    bc_energy_fall_frames: int = 5
    bc_clip_start_ms: int = 50
