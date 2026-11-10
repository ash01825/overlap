# ruff: noqa: E402
"""Invariant test for stop latency (INV-L5).

ARCHITECTURE §10:
INV-L5: Verdict INTERRUPT -> last agent sample emitted <= 40 ms (one frame).

Enforced in virtual time using VirtualClock, PlayoutMixer, and StreamSupervisor.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

_fakes_dir = str(Path(__file__).resolve().parents[1] / "fakes")
if _fakes_dir not in sys.path:
    sys.path.insert(0, _fakes_dir)

from transport import FakeTransport

from overlap.core.clock import VirtualClock
from overlap.core.fsm import FSM
from overlap.core.types import (
    EndOfTurn,
    FirstAudioReady,
    FsmState,
    Label,
    SetGain,
    SttEvent,
    Tuning,
    VadOnset,
    Verdict,
)
from overlap.ports.tts import AudioChunk
from overlap.runtime.mixer import PlayoutMixer
from overlap.runtime.supervisor import StreamSupervisor


def _generate_tone_pcm(
    num_samples: int,
    freq_hz: float = 440.0,
    amplitude: float = 14000.0,
    sample_rate: int = 16000,
) -> bytes:
    """Generate int16 mono sine wave PCM."""
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    wave = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
    int16_samples = np.clip(np.rint(wave), -32768, 32767).astype(np.int16)
    return bytes(int16_samples.tobytes())


@pytest.mark.parametrize("verdict_offset_ms", [0, 5, 10, 15, 19])
def test_stop_latency_invariant_inv_l5(
    verdict_offset_ms: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verdict INTERRUPT -> last agent sample emitted <= 40 ms virtual (INV-L5)."""
    clock = VirtualClock(start_ns=0)
    transport = FakeTransport(clock=clock)
    tuning = Tuning()

    mixer = PlayoutMixer(transport=transport, clock=clock, tuning=tuning)
    supervisor = StreamSupervisor(clock=clock, mixer=mixer, transport=transport)
    fsm = FSM(tuning=tuning)

    # 1. Start agent turn and queue 1000 ms of audio
    turn_id = "test_turn_1"
    epoch = 1
    supervisor.start_turn(turn_id, epoch)
    mixer.start_turn(turn_id, epoch)

    agent_pcm = _generate_tone_pcm(320 * 50, amplitude=14000.0)
    mixer.enqueue_chunk(AudioChunk(pcm=agent_pcm, sample_rate=16000, epoch=epoch, is_last=True))

    # Advance FSM from LISTENING -> THINKING -> SPEAKING
    fsm.transition(EndOfTurn(final=SttEvent(kind="final", text="prompt")), clock.now_ns())
    assert fsm.state == FsmState.THINKING

    effects = fsm.transition(FirstAudioReady(), clock.now_ns())
    supervisor.execute_effects(effects)
    assert cast(FsmState, fsm.state) == FsmState.SPEAKING

    # Let agent speak for 100 ms (5 frames: [0..20], [20..40], [40..60], [60..80], [80..100])
    for _ in range(5):
        clock.advance(20)

    base_t_ms = 100

    # User speech onset ducks playout
    onset_t_ns = base_t_ms * 1_000_000
    duck_effects = fsm.transition(VadOnset(ts_ns=onset_t_ns), onset_t_ns)
    for eff in duck_effects:
        if isinstance(eff, SetGain):
            mixer.set_gain(eff.gain, eff.ramp_ms)
    supervisor.execute_effects(duck_effects)
    assert cast(FsmState, fsm.state) == FsmState.DUCKED

    # Advance to verdict time: base_t_ms + verdict_offset_ms
    if verdict_offset_ms > 0:
        clock.advance(verdict_offset_ms)

    verdict_t_ms = base_t_ms + verdict_offset_ms
    verdict_t_ns = int(verdict_t_ms * 1_000_000)

    # 2. Issue INTERRUPT verdict
    verdict = Verdict(
        label=Label.INTERRUPT,
        confidence=1.0,
        reason="test_interrupt",
        overlap_ms=verdict_offset_ms,
        decided_at_ns=verdict_t_ns,
    )
    interrupt_effects = fsm.transition(verdict, verdict_t_ns)
    supervisor.execute_effects(interrupt_effects)
    assert cast(FsmState, fsm.state) == FsmState.YIELDED

    # Advance clock by 100 ms to confirm no further agent audio is emitted
    for _ in range(5):
        clock.advance(20)

    # 3. Analyze frames sent to transport after verdict
    frames_after_verdict = [r for r in transport.sent_frames if r.ts > verdict_t_ns]

    if frames_after_verdict:
        last_frame_ts = frames_after_verdict[-1].ts
        stop_latency_ms = (last_frame_ts - verdict_t_ns) / 1_000_000.0
    else:
        # Mixer flush immediately dropped un-emitted frames -> latency is 0 ms
        stop_latency_ms = 0.0

    # INV-L5: Verdict INTERRUPT -> last agent sample emitted <= 40 ms
    assert stop_latency_ms <= tuning.interrupt_stop_cap_ms, (
        f"INV-L5 VIOLATION at offset {verdict_offset_ms}ms: "
        f"stop latency {stop_latency_ms:.2f} ms > {tuning.interrupt_stop_cap_ms} ms"
    )

    result_line = (
        f"INV-L5 PASSED: verdict at t={verdict_t_ms}ms (offset {verdict_offset_ms}ms) -> "
        f"virtual stop latency = {stop_latency_ms:.1f} ms <= {tuning.interrupt_stop_cap_ms} ms"
    )
    with capsys.disabled():
        print(result_line)
