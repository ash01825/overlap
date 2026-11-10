# ruff: noqa: E402
"""Invariant tests for post-cancel audio leakage (INV-L7) and quiescence hygiene (INV-S1).

ARCHITECTURE §10:
INV-L7: Post-cancel audio leakage (stale-epoch chunks reaching transport) = 0 frames.
INV-S1: Cancel/timer hygiene: after any scenario/cancellation, 0 live tasks/streams/timers.
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
    YieldCompleted,
)
from overlap.ports.tts import AudioChunk
from overlap.runtime.mixer import PlayoutMixer
from overlap.runtime.supervisor import StreamSupervisor


def _generate_tone_pcm(
    num_samples: int,
    amplitude: float = 12000.0,
    sample_rate: int = 16000,
) -> bytes:
    """Generate int16 mono sine wave PCM."""
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    wave = amplitude * np.sin(2.0 * np.pi * 440.0 * t)
    int16_samples = np.clip(np.rint(wave), -32768, 32767).astype(np.int16)
    return bytes(int16_samples.tobytes())


def test_no_leak_post_cancel_zero_frames(capsys: pytest.CaptureFixture[str]) -> None:
    """Post-cancel audio leakage: stale-epoch chunks never reach transport (INV-L7)."""
    clock = VirtualClock(start_ns=0)
    transport = FakeTransport(clock=clock)
    tuning = Tuning()

    mixer = PlayoutMixer(transport=transport, clock=clock, tuning=tuning)
    supervisor = StreamSupervisor(clock=clock, mixer=mixer, transport=transport)
    fsm = FSM(tuning=tuning)

    turn_id = "turn_1"
    epoch = 1
    supervisor.start_turn(turn_id, epoch)
    mixer.start_turn(turn_id, epoch)

    # 1. Agent starts speaking
    fsm.transition(EndOfTurn(final=SttEvent(kind="final", text="prompt")), clock.now_ns())
    effects = fsm.transition(FirstAudioReady(), clock.now_ns())
    supervisor.execute_effects(effects)
    assert fsm.state == FsmState.SPEAKING

    # Enqueue initial 400 ms of agent audio
    initial_pcm = _generate_tone_pcm(320 * 20)
    mixer.enqueue_chunk(AudioChunk(pcm=initial_pcm, sample_rate=16000, epoch=epoch, is_last=False))

    # Let agent speak for 60 ms (3 frames emitted)
    for _ in range(3):
        clock.advance(20)

    initial_frames_sent = len(transport.sent_frames)
    assert initial_frames_sent >= 2

    # 2. User overlap occurs and triggers INTERRUPT verdict
    duck_effects = fsm.transition(VadOnset(ts_ns=clock.now_ns()), clock.now_ns())
    for eff in duck_effects:
        if isinstance(eff, SetGain):
            mixer.set_gain(eff.gain, eff.ramp_ms)
    supervisor.execute_effects(duck_effects)
    assert cast(FsmState, fsm.state) == FsmState.DUCKED

    # Advance 20 ms
    clock.advance(20)

    # Issue INTERRUPT verdict -> transitions to YIELDED, bumps epoch to 2
    verdict = Verdict(
        label=Label.INTERRUPT,
        confidence=1.0,
        reason="user_interrupted",
        overlap_ms=20,
        decided_at_ns=clock.now_ns(),
    )
    yield_effects = fsm.transition(verdict, clock.now_ns())
    supervisor.execute_effects(yield_effects)
    assert cast(FsmState, fsm.state) == FsmState.YIELDED

    # Baseline frame count at the exact moment of flush/yield
    baseline_frames_count = len(transport.sent_frames)

    # 3. Simulate delayed/stale chunks from old epoch=1 arriving after cancellation
    stale_pcm = _generate_tone_pcm(320 * 10)  # 200 ms of stale audio
    for _ in range(5):
        mixer.enqueue_chunk(
            AudioChunk(pcm=stale_pcm, sample_rate=16000, epoch=epoch, is_last=False)
        )

    # Advance clock by 100 ms in 20 ms steps to allow any leaked frames to emerge
    for _ in range(5):
        clock.advance(20)

    # Verify zero stale-epoch frames reached transport
    final_frames_count = len(transport.sent_frames)
    leaked_frames = final_frames_count - baseline_frames_count

    assert leaked_frames == tuning.post_cancel_leak_frames, (
        f"INV-L7 VIOLATION: {leaked_frames} stale-epoch frames leaked to transport "
        f"(expected {tuning.post_cancel_leak_frames})"
    )

    # 4. Complete yield sequence and verify quiescence (INV-S1)
    fsm.transition(YieldCompleted(), clock.now_ns())
    assert cast(FsmState, fsm.state) == FsmState.LISTENING

    supervisor.assert_quiescent()

    result_line = (
        f"INV-L7 & INV-S1 PASSED: post-cancel leaked frames = {leaked_frames} == 0, "
        f"supervisor is quiescent (0 live tasks/streams/timers)"
    )
    with capsys.disabled():
        print(result_line)
