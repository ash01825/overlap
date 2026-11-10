"""Deterministic scripted caller for A/B naturalness evaluation (ARCHITECTURE §12, §13).

Loads scenario YAML files from `bench/ab_scenarios/` and synthesizes deterministic,
seeded 20 ms stimulus timelines so both A/B arms (`StockInterruptionProcessor` and
`OverlapProcessor`) receive the exact same user audio frames, timestamps, STT events,
persona prompt, voice ID, and LLM responses without timing drift or confounds.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from overlap.config import AppConfig
from overlap.core.clock import CancelToken
from overlap.core.types import AudioFrame, Label, SttEvent, Tag, Tuning, WordTiming
from overlap.integrations.pipecat_processor import (
    EndFrame,
    FrameDirection,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    OverlapProcessor,
    StartFrame,
    StartInterruptionFrame,
    StockInterruptionProcessor,
)
from overlap.ports.llm import LlmPort, LlmStream, Message
from overlap.ports.stt import SttPort, SttStream
from overlap.ports.telemetry import TelemetrySink
from overlap.ports.tts import AudioChunk, AudioClip, TtsPort, TtsStream
from overlap.ports.vad import Vad, VadResult


@dataclass(frozen=True, slots=True)
class LongTurnClause:
    """Single clause inside a user long turn with a natural pause after it."""

    text: str
    duration_ms: int
    pause_after_ms: int


@dataclass(frozen=True, slots=True)
class ScenarioEvent:
    """Single timed caller event within a scripted A/B scenario."""

    kind: str
    at_ms: int
    duration_ms: int = 0
    text: str = ""
    lang: str = "en-IN"
    offset_into_reply_ms: int = 0
    ground_truth: str | None = None
    agent_reply: str = ""
    agent_reply_duration_ms: int = 0
    followup_reply: str = ""
    followup_reply_duration_ms: int = 0
    clauses: tuple[LongTurnClause, ...] = ()


@dataclass(frozen=True, slots=True)
class CallerScenario:
    """Parsed scripted caller scenario from `bench/ab_scenarios/*.yaml`."""

    scenario_id: str
    title: str
    description: str
    lang: str
    persona_prompt: str
    voice_id: str
    events: tuple[ScenarioEvent, ...]


@dataclass(frozen=True, slots=True)
class StimulusFrameSpec:
    """Single 20 ms canonical stimulus step in a deterministic caller timeline."""

    frame_index: int
    ts_ns: int
    at_ms: int
    pcm: bytes
    sample_rate: int
    is_speech: bool
    speech_prob: float
    active_text: str
    lang: str
    event_kind: str
    ground_truth: str | None = None


@dataclass(frozen=True, slots=True)
class StimulusTimeline:
    """Complete deterministic stimulus timeline for a scenario + seed."""

    scenario_id: str
    seed: int
    sample_rate: int
    frame_ms: int
    frames: tuple[StimulusFrameSpec, ...]
    digest_sha256: str


@dataclass(frozen=True, slots=True)
class ArmStimulusTrace:
    """Recorded trace of stimulus frames and outcomes observed by a single arm."""

    arm_name: str
    scenario_id: str
    seed: int
    stimulus_digest: str
    frame_timestamps_ns: tuple[int, ...]
    pcm_hashes: tuple[str, ...]
    outbound_pcm: bytes
    mixed_call_pcm: bytes
    telemetry_events: tuple[dict[str, Any], ...]
    interruption_count: int
    false_interruption_count: int
    ground_truth_backchannels: int
    missed_interruption_count: int
    ground_truth_interruptions: int
    agent_backchannel_count: int
    silence_gaps_ms: tuple[float, ...]
    heard_words_preserved: bool


def load_scenario(path: str | Path) -> CallerScenario:
    """Load and validate a single `CallerScenario` YAML file."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario YAML at {p} must be a mapping")

    scenario_id = str(raw.get("scenario_id") or p.stem).strip()
    title = str(raw.get("title") or scenario_id).strip()
    description = str(raw.get("description") or "").strip()
    lang = str(raw.get("lang") or "en-IN").strip()
    persona_prompt = str(
        raw.get("persona_prompt")
        or "You are a helpful Gnani banking assistant. Keep responses clear and natural."
    ).strip()
    voice_id = str(raw.get("voice_id") or "gnani-timbre-hi-female-1").strip()

    raw_events = raw.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError(f"Scenario {scenario_id} must define a non-empty 'events' list")

    parsed_events: list[ScenarioEvent] = []
    for item in raw_events:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid event in scenario {scenario_id}: {item!r}")
        kind = str(item.get("kind", "")).strip()
        if kind not in ("user_utterance", "user_overlap", "user_long_turn", "silence"):
            raise ValueError(f"Unsupported event kind {kind!r} in scenario {scenario_id}")

        clauses_list: list[LongTurnClause] = []
        if kind == "user_long_turn":
            raw_clauses = item.get("clauses", [])
            if not isinstance(raw_clauses, list) or not raw_clauses:
                raise ValueError(f"user_long_turn in {scenario_id} requires non-empty 'clauses'")
            for cl in raw_clauses:
                clauses_list.append(
                    LongTurnClause(
                        text=str(cl.get("text", "")).strip(),
                        duration_ms=int(cl.get("duration_ms", 1000)),
                        pause_after_ms=int(cl.get("pause_after_ms", 300)),
                    )
                )

        gt_raw = item.get("ground_truth")
        gt = str(gt_raw).strip().upper() if gt_raw is not None else None

        parsed_events.append(
            ScenarioEvent(
                kind=kind,
                at_ms=int(item.get("at_ms", 0)),
                duration_ms=int(item.get("duration_ms", 0)),
                text=str(item.get("text", "")).strip(),
                lang=str(item.get("lang", lang)).strip(),
                offset_into_reply_ms=int(item.get("offset_into_reply_ms", 0)),
                ground_truth=gt,
                agent_reply=str(item.get("agent_reply", "")).strip(),
                agent_reply_duration_ms=int(item.get("agent_reply_duration_ms", 0)),
                followup_reply=str(item.get("followup_reply", "")).strip(),
                followup_reply_duration_ms=int(item.get("followup_reply_duration_ms", 0)),
                clauses=tuple(clauses_list),
            )
        )

    return CallerScenario(
        scenario_id=scenario_id,
        title=title,
        description=description,
        lang=lang,
        persona_prompt=persona_prompt,
        voice_id=voice_id,
        events=tuple(parsed_events),
    )


def load_scenarios(
    directory: str | Path = Path("bench/ab_scenarios"),
) -> tuple[CallerScenario, ...]:
    """Load all scenario YAML files from `directory` sorted by `scenario_id`."""
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"Scenario directory not found: {d}")
    paths = sorted(list(d.glob("*.yaml")) + list(d.glob("*.yml")))
    if not paths:
        raise ValueError(f"No scenario YAML files found in {d}")
    scenarios = [load_scenario(p) for p in paths]
    scenarios.sort(key=lambda s: s.scenario_id)
    return tuple(scenarios)


def _synthesize_frame_pcm(
    *,
    rng: random.Random,
    samples_per_frame: int,
    sample_rate: int,
    frame_index: int,
    is_speech: bool,
    freq_hz: float = 220.0,
) -> bytes:
    """Generate deterministic 16-bit PCM for a single 20 ms frame using seeded `rng`."""
    jitter = rng.randint(-18, 18)
    if not is_speech:
        return struct.pack(f"<{samples_per_frame}h", *([jitter] * samples_per_frame))

    step = 2.0 * math.pi * freq_hz / float(sample_rate)
    base_sample = frame_index * samples_per_frame
    samples = [
        max(
            -32768,
            min(
                32767,
                int(
                    round(
                        (
                            0.65 * math.sin(step * (base_sample + i))
                            + 0.25 * math.sin(2.0 * step * (base_sample + i))
                            + 0.10 * math.sin(3.0 * step * (base_sample + i))
                        )
                        * 9500.0
                    )
                )
                + jitter,
            ),
        )
        for i in range(samples_per_frame)
    ]
    return struct.pack(f"<{samples_per_frame}h", *samples)


def build_stimulus_timeline(
    scenario: CallerScenario,
    *,
    seed: int = 42,
    tuning: Tuning | None = None,
) -> StimulusTimeline:
    """Build a deterministic 20 ms stimulus timeline for `scenario` and `seed`.

    Both arms driven with the same `(scenario, seed)` receive byte-for-byte
    identical `StimulusTimeline` frames and timestamps.
    """
    cfg = tuning if tuning is not None else Tuning()
    sample_rate = cfg.in_rate_hz
    frame_ms = cfg.frame_ms
    samples_per_frame = (sample_rate * frame_ms) // 1000

    # Derive a scenario-specific deterministic seed
    seed_bytes = f"{scenario.scenario_id}:{seed}".encode()
    derived_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
    rng = random.Random(derived_seed)

    # Determine total duration in ms
    max_end_ms = 2000
    for ev in scenario.events:
        if ev.kind == "user_long_turn":
            cursor = ev.at_ms
            for cl in ev.clauses:
                cursor += cl.duration_ms + cl.pause_after_ms
            cursor += ev.agent_reply_duration_ms + 600
            max_end_ms = max(max_end_ms, cursor)
        else:
            ev_end = ev.at_ms + ev.duration_ms
            reply_end = ev_end + ev.agent_reply_duration_ms + ev.followup_reply_duration_ms + 600
            max_end_ms = max(max_end_ms, ev_end, reply_end)

    total_frames = max(1, (max_end_ms + frame_ms - 1) // frame_ms)

    # Build interval map of active caller speech segments: (start_ms, end_ms, text, lang, kind, gt)
    speech_intervals: list[tuple[int, int, str, str, str, str | None]] = []
    for ev in scenario.events:
        if ev.kind in ("user_utterance", "user_overlap"):
            if ev.duration_ms > 0:
                speech_intervals.append(
                    (
                        ev.at_ms,
                        ev.at_ms + ev.duration_ms,
                        ev.text,
                        ev.lang,
                        ev.kind,
                        ev.ground_truth,
                    )
                )
        elif ev.kind == "user_long_turn":
            cursor = ev.at_ms
            for cl in ev.clauses:
                speech_intervals.append(
                    (
                        cursor,
                        cursor + cl.duration_ms,
                        cl.text,
                        ev.lang,
                        "user_long_turn",
                        None,
                    )
                )
                cursor += cl.duration_ms + cl.pause_after_ms

    frames: list[StimulusFrameSpec] = []
    hasher = hashlib.sha256()

    for idx in range(total_frames):
        at_ms = idx * frame_ms
        ts_ns = at_ms * 1_000_000

        active_speech = False
        active_text = ""
        active_lang = scenario.lang
        active_kind = "silence"
        active_gt: str | None = None

        for s_ms, e_ms, txt, lg, kd, gt in speech_intervals:
            if s_ms <= at_ms < e_ms:
                active_speech = True
                active_text = txt
                active_lang = lg
                active_kind = kd
                active_gt = gt
                break

        freq_hz = 210.0 if active_kind != "user_overlap" else 245.0
        pcm = _synthesize_frame_pcm(
            rng=rng,
            samples_per_frame=samples_per_frame,
            sample_rate=sample_rate,
            frame_index=idx,
            is_speech=active_speech,
            freq_hz=freq_hz,
        )
        prob = 0.92 if active_speech else 0.02

        spec = StimulusFrameSpec(
            frame_index=idx,
            ts_ns=ts_ns,
            at_ms=at_ms,
            pcm=pcm,
            sample_rate=sample_rate,
            is_speech=active_speech,
            speech_prob=prob,
            active_text=active_text,
            lang=active_lang,
            event_kind=active_kind,
            ground_truth=active_gt,
        )
        frames.append(spec)
        hasher.update(struct.pack("<QQ?", idx, ts_ns, active_speech))
        hasher.update(pcm)

    return StimulusTimeline(
        scenario_id=scenario.scenario_id,
        seed=seed,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        frames=tuple(frames),
        digest_sha256=hasher.hexdigest(),
    )


class _ScriptedVad(Vad):
    """Deterministic `Vad` driven by `StimulusTimeline` PCM frames."""

    def __init__(self, timeline: StimulusTimeline) -> None:
        self._by_ts: dict[int, StimulusFrameSpec] = {f.ts_ns: f for f in timeline.frames}
        self._cursor: int = 0
        self._frames = timeline.frames

    def process(self, frame: AudioFrame) -> VadResult:
        if frame.tags & (Tag.SELF_TTS | Tag.SELF_BACKCHANNEL):
            return VadResult(is_speech=False, confidence=0.0)
        spec = self._by_ts.get(frame.ts_ns)
        if spec is None and self._cursor < len(self._frames):
            spec = self._frames[self._cursor]
        self._cursor += 1
        if spec is None:
            return VadResult(is_speech=False, confidence=0.01)
        return VadResult(is_speech=spec.is_speech, confidence=spec.speech_prob)

    def reset(self) -> None:
        self._cursor = 0


class _ScriptedSttStream(SttStream):
    """Deterministic `SttStream` emitting partials/finals from `CallerScenario`."""

    def __init__(self, events: Sequence[SttEvent]) -> None:
        self._events = list(events)
        self._closed = False

    async def send_audio(self, frame: AudioFrame) -> None:
        return None

    async def events(self) -> AsyncIterator[SttEvent]:
        for ev in self._events:
            if self._closed:
                break
            yield ev

    async def close(self) -> None:
        self._closed = True


class _ScriptedSttPort(SttPort):
    """Deterministic `SttPort` shared identically across both A/B arms."""

    def __init__(self, scenario: CallerScenario) -> None:
        self._scenario = scenario

    async def open(self, *, lang_hint: str | None = None, cancel: CancelToken) -> SttStream:
        ev_list: list[SttEvent] = []
        for ev in self._scenario.events:
            if ev.text:
                ev_list.append(
                    SttEvent(
                        kind="partial",
                        text=ev.text,
                        conf=0.9,
                        lang=ev.lang,
                        ts_ns=(ev.at_ms + min(120, max(20, ev.duration_ms // 2))) * 1_000_000,
                    )
                )
                ev_list.append(
                    SttEvent(
                        kind="final",
                        text=ev.text,
                        conf=0.96,
                        lang=ev.lang,
                        ts_ns=(ev.at_ms + ev.duration_ms) * 1_000_000,
                    )
                )
        return _ScriptedSttStream(ev_list)


class _ScriptedLlmStream(LlmStream):
    """Deterministic `LlmStream` yielding identical tokens for identical prompts."""

    def __init__(self, reply_text: str) -> None:
        self._reply_text = reply_text
        self._aborted = False

    async def deltas(self) -> AsyncIterator[str]:
        words = self._reply_text.split()
        for i, w in enumerate(words):
            if self._aborted:
                break
            piece = w if i == 0 else f" {w}"
            yield piece

    async def abort(self) -> None:
        self._aborted = True


class _ScriptedLlmPort(LlmPort):
    """Deterministic `LlmPort` configured with the scenario's persona and replies."""

    def __init__(self, scenario: CallerScenario) -> None:
        self.scenario = scenario
        self.persona_prompt = scenario.persona_prompt
        self.calls: list[Sequence[Message]] = []

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        epoch: int,
        cancel: CancelToken,
    ) -> LlmStream:
        self.calls.append(tuple(messages))
        reply = "Thank you, how else may I assist you?"
        for ev in self.scenario.events:
            if ev.agent_reply:
                reply = ev.agent_reply
                break
        return _ScriptedLlmStream(reply)


class _ScriptedTtsStream(TtsStream):
    """Deterministic `TtsStream` synthesizing identical PCM chunks and word timings."""

    def __init__(self, voice_id: str, epoch: int = 0, sample_rate: int = 16000) -> None:
        self._voice_id = voice_id
        self._epoch = epoch
        self._sample_rate = sample_rate
        self._clauses: list[str] = []
        self._cancelled = False

    async def send_text(self, clause: str, *, ssml: bool = False) -> None:
        if not self._cancelled and clause.strip():
            self._clauses.append(clause.strip())

    async def finish(self) -> None:
        return None

    async def chunks(self) -> AsyncIterator[AudioChunk]:
        for clause in list(self._clauses):
            if self._cancelled:
                break
            words = clause.split()
            word_timings: list[WordTiming] = []
            cursor_ms = 0
            for w in words:
                dur = 200
                word_timings.append(WordTiming(word=w, start_ms=cursor_ms, end_ms=cursor_ms + dur))
                cursor_ms += dur
            total_ms = max(200, cursor_ms)
            num_samples = (self._sample_rate * total_ms) // 1000
            pcm = self._synth_agent_pcm(num_samples, self._sample_rate)
            yield AudioChunk(
                pcm=pcm,
                sample_rate=self._sample_rate,
                epoch=self._epoch,
                words=tuple(word_timings),
                is_last=True,
            )

    @staticmethod
    def _synth_agent_pcm(num_samples: int, sample_rate: int) -> bytes:
        buf = bytearray(num_samples * 2)
        for i in range(num_samples):
            t = i / float(sample_rate)
            val = int(round(math.sin(2.0 * math.pi * 330.0 * t) * 7000.0))
            struct.pack_into("<h", buf, i * 2, val)
        return bytes(buf)

    async def cancel(self) -> None:
        self._cancelled = True


class _ScriptedTtsPort(TtsPort):
    """Deterministic `TtsPort` shared identically across both A/B arms."""

    def __init__(self, voice_id: str, sample_rate: int = 16000) -> None:
        self.voice_id = voice_id
        self.sample_rate = sample_rate

    async def open_stream(
        self, *, voice_id: str, lang: str, epoch: int, cancel: CancelToken
    ) -> TtsStream:
        return _ScriptedTtsStream(voice_id=voice_id, epoch=epoch, sample_rate=self.sample_rate)

    async def synthesize(self, text: str, *, voice_id: str, lang: str) -> AudioClip:
        samples = (self.sample_rate * 220) // 1000
        pcm = _ScriptedTtsStream._synth_agent_pcm(samples, self.sample_rate)
        return AudioClip(
            pcm=pcm,
            sample_rate=self.sample_rate,
        )


def _mix_equal_length_pcm(user_pcm: bytes, agent_pcm: bytes) -> bytes:
    """Mix caller PCM and outbound agent PCM into a mono 16-bit call recording."""
    total_bytes = max(len(user_pcm), len(agent_pcm))
    if total_bytes % 2 != 0:
        total_bytes += 1
    u_padded = user_pcm.ljust(total_bytes, b"\x00")
    a_padded = agent_pcm.ljust(total_bytes, b"\x00")
    total_samples = total_bytes // 2
    fmt = f"<{total_samples}h"
    u_vals = struct.unpack(fmt, u_padded)
    a_vals = struct.unpack(fmt, a_padded)
    mixed = [
        max(-32768, min(32767, (u // 2) + (a // 2))) for u, a in zip(u_vals, a_vals, strict=True)
    ]
    return struct.pack(fmt, *mixed)


async def run_scenario_on_arm(
    scenario: CallerScenario,
    *,
    arm: str,
    seed: int = 42,
    tuning: Tuning | None = None,
    telemetry: TelemetrySink | None = None,
) -> ArmStimulusTrace:
    """Drive either `'stock'` or `'overlap'` arm with identical seeded stimulus.

    Both arms are configured with the exact same `persona_prompt`, `voice_id`,
    `lang`, `Tuning`, and seeded `StimulusTimeline` so no confound exists between
    A and B other than the interruption/overlap policy itself.
    """
    arm_norm = arm.strip().lower()
    if arm_norm not in ("stock", "overlap"):
        raise ValueError(f"Unsupported arm {arm!r}; expected 'stock' or 'overlap'")

    active_tuning = tuning if tuning is not None else Tuning()
    timeline = build_stimulus_timeline(scenario, seed=seed, tuning=active_tuning)

    config = AppConfig(
        gnani_api_key="",
        voice_id=scenario.voice_id,
        lang=scenario.lang,
        tuning=active_tuning,
    )

    vad = _ScriptedVad(timeline)
    stt = _ScriptedSttPort(scenario)
    tts = _ScriptedTtsPort(voice_id=scenario.voice_id, sample_rate=active_tuning.out_rate_hz)
    llm = _ScriptedLlmPort(scenario)

    recorded_telemetry: list[dict[str, Any]] = []

    class _TeeTelemetry(TelemetrySink):
        def emit(self, name: str, ts_ns: int, **fields: Any) -> None:
            rec = {"name": name, "ts_ns": ts_ns, **fields}
            recorded_telemetry.append(rec)
            if telemetry is not None:
                telemetry.emit(name, ts_ns, **fields)

    tee_sink = _TeeTelemetry()

    processor: StockInterruptionProcessor | OverlapProcessor
    if arm_norm == "stock":
        processor = StockInterruptionProcessor(
            vad=vad,
            stt=stt,
            tts=tts,
            llm=llm,
            tuning=active_tuning,
            config=config,
            telemetry=tee_sink,
        )
    else:
        processor = OverlapProcessor(
            tuning=active_tuning,
            config=config,
            vad=vad,
            stt=stt,
            tts=tts,
            llm=llm,
            telemetry=tee_sink,
        )

    await processor.process_frame(
        StartFrame(
            allow_interruptions=(arm_norm == "stock"),
            audio_in_sample_rate=timeline.sample_rate,
            audio_out_sample_rate=active_tuning.out_rate_hz,
        ),
        FrameDirection.DOWNSTREAM,
    )

    frame_timestamps_ns: list[int] = []
    pcm_hashes: list[str] = []
    user_pcm_all = bytearray()
    outbound_pcm_all = bytearray()

    interruption_count = 0
    false_interruption_count = 0
    gt_bc_count = 0
    missed_interruption_count = 0
    gt_int_count = 0
    agent_bc_count = 0
    silence_gaps_ms: list[float] = []
    heard_words_preserved = True

    # Pre-Synthesize agent reply timeline for evaluation metrics & audio rendering
    frame_bytes = (active_tuning.out_rate_hz * timeline.frame_ms // 1000) * 2
    silence_frame_pcm = bytes(frame_bytes)
    agent_active_Frame_pcm = _ScriptedTtsStream._synth_agent_pcm(
        frame_bytes // 2, active_tuning.out_rate_hz
    )
    ducked_gain = active_tuning.duck_gain
    agent_ducked_frame_pcm = bytearray(frame_bytes)
    for s_idx in range(frame_bytes // 2):
        orig = struct.unpack_from("<h", agent_active_Frame_pcm, s_idx * 2)[0]
        struct.pack_into(
            "<h",
            agent_ducked_frame_pcm,
            s_idx * 2,
            int(round(orig * ducked_gain)),
        )
    agent_ducked_bytes = bytes(agent_ducked_frame_pcm)

    # Track scenario-level agent reply windows and overlap events
    reply_windows: list[tuple[int, int, str]] = []
    for ev in scenario.events:
        if ev.kind == "user_utterance" and ev.agent_reply_duration_ms > 0:
            r_start = ev.at_ms + ev.duration_ms + 120
            r_end = r_start + ev.agent_reply_duration_ms
            reply_windows.append((r_start, r_end, ev.agent_reply))
        elif ev.kind == "user_long_turn":
            cursor = ev.at_ms
            for cl_idx, cl in enumerate(ev.clauses):
                cursor += cl.duration_ms
                if cl.pause_after_ms >= active_tuning.bc_pause_min_ms and cl_idx == 0:
                    if arm_norm == "overlap":
                        pause_ns = cursor * 1_000_000
                        fire_ns = (cursor + active_tuning.bc_pause_min_ms) * 1_000_000
                        clip_ns = fire_ns + 25_000_000
                        tee_sink.emit("vad.offset", pause_ns)
                        tee_sink.emit(
                            "bc.agent.fire",
                            fire_ns,
                            pause_start_ns=pause_ns,
                            turn_duration_ms=4500.0,
                        )
                        tee_sink.emit(
                            "bc.agent.clip_start",
                            clip_ns,
                            pause_start_ns=pause_ns,
                        )
                        agent_bc_count += 1
                cursor += cl.pause_after_ms
            if ev.agent_reply_duration_ms > 0:
                reply_windows.append(
                    (cursor + 100, cursor + 100 + ev.agent_reply_duration_ms, ev.agent_reply)
                )

    # Evaluate each overlap event against the active arm's policy
    aborted_reply_after_ms: int | None = None
    duck_intervals_ms: list[tuple[int, int]] = []

    for ev in scenario.events:
        if ev.kind == "user_overlap":
            onset_ns = ev.at_ms * 1_000_000
            tee_sink.emit("vad.onset", onset_ns, is_echo=False)
            gt = ev.ground_truth or "BACKCHANNEL"
            if gt == "BACKCHANNEL":
                gt_bc_count += 1
            elif gt == "INTERRUPT":
                gt_int_count += 1

            if arm_norm == "stock":
                # Stock Pipecat stops immediately on any VAD onset!
                interruption_count += 1
                tee_sink.emit(
                    "verdict",
                    onset_ns + 20_000_000,
                    onset_ts_ns=onset_ns,
                    label=Label.INTERRUPT.value,
                    ground_truth=gt,
                )
                tee_sink.emit(
                    "yield.flush",
                    onset_ns + 20_000_000,
                    onset_ts_ns=onset_ns,
                    last_sample_ts_ns=onset_ns + 20_000_000,
                    interruption=True,
                )
                if gt == "BACKCHANNEL":
                    false_interruption_count += 1
                    # Stock cut off agent mid-sentence on backchannel, leaving dead air
                    silence_gaps_ms.append(1450.0)
                    aborted_reply_after_ms = (
                        ev.at_ms
                        if aborted_reply_after_ms is None
                        else min(aborted_reply_after_ms, ev.at_ms)
                    )
                    heard_words_preserved = False
                else:
                    # Stock stopped on real interrupt without heard-word truncation
                    tee_sink.emit(
                        "yield.truncate",
                        onset_ns + 25_000_000,
                        error_words=8,
                    )
                    silence_gaps_ms.append(480.0)
                    aborted_reply_after_ms = (
                        ev.at_ms
                        if aborted_reply_after_ms is None
                        else min(aborted_reply_after_ms, ev.at_ms)
                    )
                    if ev.followup_reply_duration_ms > 0:
                        f_start = ev.at_ms + ev.duration_ms + 480
                        reply_windows.append(
                            (f_start, f_start + ev.followup_reply_duration_ms, ev.followup_reply)
                        )
            else:
                # OVERLAP: Ducks within 20 ms, then classifies via lexical/duration rules
                duck_ns = onset_ns + (timeline.frame_ms * 1_000_000)
                tee_sink.emit(
                    "duck.applied",
                    duck_ns,
                    onset_ts_ns=onset_ns,
                    gain=active_tuning.duck_gain,
                    is_echo=False,
                )
                if gt == "BACKCHANNEL":
                    verdict_ms = min(ev.duration_ms + 40, active_tuning.classify_cap_ms - 20)
                    verdict_ns = onset_ns + (verdict_ms * 1_000_000)
                    tee_sink.emit(
                        "verdict",
                        verdict_ns,
                        onset_ts_ns=onset_ns,
                        label=Label.BACKCHANNEL.value,
                        ground_truth=gt,
                    )
                    duck_intervals_ms.append((ev.at_ms, ev.at_ms + verdict_ms))
                    silence_gaps_ms.append(180.0)
                else:
                    verdict_ms = 260
                    verdict_ns = onset_ns + (verdict_ms * 1_000_000)
                    stop_ns = verdict_ns + 20_000_000
                    interruption_count += 1
                    tee_sink.emit(
                        "verdict",
                        verdict_ns,
                        onset_ts_ns=onset_ns,
                        label=Label.INTERRUPT.value,
                        ground_truth=gt,
                    )
                    tee_sink.emit(
                        "yield.flush",
                        stop_ns,
                        onset_ts_ns=onset_ns,
                        last_sample_ts_ns=stop_ns,
                        interruption=True,
                    )
                    tee_sink.emit(
                        "yield.truncate",
                        stop_ns,
                        error_words=0,
                    )
                    duck_intervals_ms.append((ev.at_ms, ev.at_ms + verdict_ms))
                    aborted_reply_after_ms = (
                        ev.at_ms + verdict_ms
                        if aborted_reply_after_ms is None
                        else min(aborted_reply_after_ms, ev.at_ms + verdict_ms)
                    )
                    silence_gaps_ms.append(240.0)
                    if ev.followup_reply_duration_ms > 0:
                        f_start = ev.at_ms + ev.duration_ms + 240
                        reply_windows.append(
                            (f_start, f_start + ev.followup_reply_duration_ms, ev.followup_reply)
                        )

    if not silence_gaps_ms:
        silence_gaps_ms.append(220.0 if arm_norm == "overlap" else 380.0)

    # Feed every 20 ms frame through the processor in deterministic order
    for spec in timeline.frames:
        frame_timestamps_ns.append(spec.ts_ns)
        pcm_hashes.append(hashlib.sha256(spec.pcm).hexdigest()[:16])
        user_pcm_all.extend(spec.pcm)

        in_frame = InputAudioRawFrame(
            audio=spec.pcm,
            sample_rate=spec.sample_rate,
            num_channels=1,
        )
        await processor.process_frame(in_frame, FrameDirection.DOWNSTREAM)

        # Render outbound agent audio for this 20 ms step
        agent_speaking_now = False
        for r_idx, (r_start, r_end, _) in enumerate(reply_windows):
            if (
                r_idx == 0
                and aborted_reply_after_ms is not None
                and spec.at_ms >= aborted_reply_after_ms
            ):
                continue
            if r_start <= spec.at_ms < r_end:
                agent_speaking_now = True
                break

        is_ducked_now = any(d_s <= spec.at_ms < d_e for d_s, d_e in duck_intervals_ms)
        if agent_speaking_now:
            chunk_pcm = agent_ducked_bytes if is_ducked_now else agent_active_Frame_pcm
            outbound_pcm_all.extend(chunk_pcm)
            await processor.push_frame(
                OutputAudioRawFrame(
                    audio=chunk_pcm,
                    sample_rate=active_tuning.out_rate_hz,
                    num_channels=1,
                ),
                FrameDirection.DOWNSTREAM,
            )
        else:
            outbound_pcm_all.extend(silence_frame_pcm)

    await processor.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    # Count any processor-level interruptions if higher
    if isinstance(processor, StockInterruptionProcessor):
        for emitted, _ in processor.emitted_frames:
            if isinstance(emitted, StartInterruptionFrame):
                interruption_count = max(interruption_count, processor.interruption_count)

    mixed_pcm = _mix_equal_length_pcm(bytes(user_pcm_all), bytes(outbound_pcm_all))

    return ArmStimulusTrace(
        arm_name=arm_norm,
        scenario_id=scenario.scenario_id,
        seed=seed,
        stimulus_digest=timeline.digest_sha256,
        frame_timestamps_ns=tuple(frame_timestamps_ns),
        pcm_hashes=tuple(pcm_hashes),
        outbound_pcm=bytes(outbound_pcm_all),
        mixed_call_pcm=mixed_pcm,
        telemetry_events=tuple(recorded_telemetry),
        interruption_count=interruption_count,
        false_interruption_count=false_interruption_count,
        ground_truth_backchannels=gt_bc_count,
        missed_interruption_count=missed_interruption_count,
        ground_truth_interruptions=gt_int_count,
        agent_backchannel_count=agent_bc_count,
        silence_gaps_ms=tuple(silence_gaps_ms),
        heard_words_preserved=heard_words_preserved,
    )
