"""Unit tests for PlayoutLedger and PlayoutSnapshot.

Tests accounting of queued and emitted audio, playout position calculation
with downstream latency and transport queue compensation, clamping bounds,
and immutable flush snapshots per ARCHITECTURE §8.2.
"""

from __future__ import annotations

import dataclasses

import pytest

from overlap.core.playout import PlayoutLedger, PlayoutSnapshot
from overlap.core.types import Tuning


def test_ledger_initial_state() -> None:
    ledger = PlayoutLedger(turn_id="turn-1", epoch=1)
    assert ledger.turn_id == "turn-1"
    assert ledger.epoch == 1
    assert ledger.samples_queued == 0
    assert ledger.samples_emitted == 0
    assert ledger.queued_ms == 0
    assert ledger.emitted_ms == 0
    assert ledger.played_ms() == 0


def test_ledger_queue_and_emit_accounting() -> None:
    tuning = Tuning()
    rate = tuning.out_rate_hz  # 16000
    ledger = PlayoutLedger(turn_id="turn-1", epoch=1, sample_rate_hz=rate)

    # Queue 3200 samples = 200 ms
    ledger.record_queued(3200)
    assert ledger.samples_queued == 3200
    assert ledger.queued_ms == 200
    assert ledger.samples_emitted == 0
    assert ledger.emitted_ms == 0

    # Emit 1600 samples = 100 ms
    ledger.record_emitted(1600)
    assert ledger.samples_emitted == 1600
    assert ledger.emitted_ms == 100
    assert ledger.samples_queued == 3200
    assert ledger.queued_ms == 200


def test_ledger_queue_ms_and_emit_ms_helpers() -> None:
    ledger = PlayoutLedger(turn_id="turn-1", epoch=1)
    ledger.queue_ms(300)
    assert ledger.queued_ms == 300

    ledger.emit_ms(200)
    assert ledger.emitted_ms == 200


def test_played_ms_normal_calculation() -> None:
    tuning = Tuning()
    ledger = PlayoutLedger(turn_id="turn-1", epoch=1, tuning=tuning)
    ledger.emit_ms(500)

    # emitted = 500 ms, transport_queue = 40 ms, downstream_latency = 100 ms
    # played_ms = 500 - 40 - 100 = 360 ms
    played = ledger.played_ms(downstream_latency_ms=100, transport_queue_ms=40)
    assert played == 360


def test_played_ms_uses_tuning_default_when_downstream_latency_is_none() -> None:
    tuning = Tuning()
    ledger = PlayoutLedger(turn_id="turn-1", epoch=1, tuning=tuning)
    ledger.emit_ms(400)

    # When downstream_latency_ms is None, uses tuning.assumed_downstream_ms (120 ms)
    # played = 400 - 0 - 120 = 280 ms
    played = ledger.played_ms(downstream_latency_ms=None, transport_queue_ms=0)
    assert played == 400 - tuning.assumed_downstream_ms


def test_played_ms_clamped_at_zero_when_latency_exceeds_emitted() -> None:
    ledger = PlayoutLedger(turn_id="turn-1", epoch=1)
    ledger.emit_ms(50)

    # 50 - 100 = -50 => clamped to 0
    played = ledger.played_ms(downstream_latency_ms=100, transport_queue_ms=0)
    assert played == 0


def test_played_ms_clamped_at_emitted_when_downstream_latency_negative() -> None:
    """Failure mode: downstream latency negative/unknown breaks clamp (played > emitted)."""
    ledger = PlayoutLedger(turn_id="turn-1", epoch=1)
    ledger.emit_ms(200)

    # Negative downstream latency (-100 ms) must NEVER allow played_ms > emitted_ms
    played = ledger.played_ms(downstream_latency_ms=-100, transport_queue_ms=0)
    assert played <= ledger.emitted_ms
    assert played == 200


def test_played_ms_clamped_to_clause_audio_total() -> None:
    ledger = PlayoutLedger(turn_id="turn-1", epoch=1)
    ledger.emit_ms(800)

    # raw = 800 - 50 = 750 ms, but clause_audio_total_ms is 600 ms
    played = ledger.played_ms(
        downstream_latency_ms=50, transport_queue_ms=0, clause_audio_total_ms=600
    )
    assert played == 600


def test_unheard_audio_in_mixer_or_transport_not_counted() -> None:
    """Failure mode: played position counts audio still queued in mixer/transport/jitter buffer."""
    ledger = PlayoutLedger(turn_id="turn-1", epoch=1)
    # 1000 ms queued in mixer
    ledger.queue_ms(1000)
    # Only 400 ms emitted to transport
    ledger.emit_ms(400)

    # Transport still has 60 ms in its queue, network/downstream has 120 ms
    # Playout must only count audio that actually reached the listener
    played = ledger.played_ms(downstream_latency_ms=120, transport_queue_ms=60)
    assert played == 400 - 60 - 120
    assert played == 220
    # 600 ms remaining in mixer queue was NEVER counted
    assert played < ledger.queued_ms


def test_flush_returns_immutable_snapshot() -> None:
    ledger = PlayoutLedger(turn_id="turn-42", epoch=3)
    ledger.queue_ms(800)
    ledger.emit_ms(500)

    snapshot = ledger.flush(downstream_latency_ms=100, transport_queue_ms=50)

    assert isinstance(snapshot, PlayoutSnapshot)
    assert snapshot.turn_id == "turn-42"
    assert snapshot.epoch == 3
    assert snapshot.samples_queued == ledger.samples_queued
    assert snapshot.samples_emitted == ledger.samples_emitted
    assert snapshot.queued_ms == 800
    assert snapshot.emitted_ms == 500
    assert snapshot.played_ms == 350
    assert snapshot.dropped_ms == 300
    assert snapshot.dropped_samples == snapshot.samples_queued - snapshot.samples_emitted

    # Snapshot is frozen/immutable
    with pytest.raises(dataclasses.FrozenInstanceError):
        # Type ignore to test runtime immutability
        snapshot.played_ms = 999  # type: ignore[misc]


def test_per_turn_epoch_accounting() -> None:
    ledger = PlayoutLedger()

    ledger.start_turn("turn-1", epoch=1)
    ledger.queue_ms(400)
    ledger.emit_ms(300)
    snap1 = ledger.flush(downstream_latency_ms=100)
    assert snap1.turn_id == "turn-1"
    assert snap1.epoch == 1
    assert snap1.played_ms == 200

    ledger.start_turn("turn-2", epoch=2)
    assert ledger.turn_id == "turn-2"
    assert ledger.epoch == 2
    assert ledger.samples_queued == 0
    assert ledger.samples_emitted == 0
    ledger.queue_ms(600)
    ledger.emit_ms(500)
    snap2 = ledger.flush(downstream_latency_ms=50)
    assert snap2.turn_id == "turn-2"
    assert snap2.epoch == 2
    assert snap2.played_ms == 450

    # History preserved
    assert ledger.get_snapshot("turn-1", 1) == snap1
    assert ledger.get_snapshot("turn-2", 2) == snap2
