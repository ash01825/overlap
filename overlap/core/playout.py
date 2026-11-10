"""Playout accounting ledger and immutable snapshots.

Tracks audio sample queuing, emission to transport, and playout estimation
per (turn_id, epoch), accounting for mixer queues, transport queues, and
downstream transmission/speaker latency per ARCHITECTURE §8.2.
"""

from __future__ import annotations

from dataclasses import dataclass

from overlap.core.types import Tuning


@dataclass(frozen=True, slots=True)
class PlayoutSnapshot:
    """Immutable record of playout state at a point in time or on flush."""

    turn_id: str
    epoch: int
    samples_queued: int
    samples_emitted: int
    sample_rate_hz: int
    queued_ms: int
    emitted_ms: int
    played_ms: int

    @property
    def dropped_samples(self) -> int:
        """Samples queued in mixer that were never emitted downstream."""
        return max(0, self.samples_queued - self.samples_emitted)

    @property
    def dropped_ms(self) -> int:
        """Duration in milliseconds queued but dropped on flush."""
        return max(0, self.queued_ms - self.emitted_ms)


class PlayoutLedger:
    """Pure accounting ledger tracking audio queuing, emission, and playout.

    Enforces invariants from ARCHITECTURE §8.2:
    played_ms = min(emitted_ms - transport_queue_ms - downstream_latency_ms, clause_audio_total_ms)
    clamped to [0, emitted_ms].
    """

    def __init__(
        self,
        turn_id: str = "",
        epoch: int = 0,
        sample_rate_hz: int | None = None,
        tuning: Tuning | None = None,
    ) -> None:
        self._tuning = tuning or Tuning()
        self._sample_rate_hz = sample_rate_hz or self._tuning.out_rate_hz
        self._turn_id = turn_id
        self._epoch = epoch
        self._samples_queued = 0
        self._samples_emitted = 0
        self._turn_snapshots: dict[tuple[str, int], PlayoutSnapshot] = {}

    @property
    def turn_id(self) -> str:
        return self._turn_id

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def sample_rate_hz(self) -> int:
        return self._sample_rate_hz

    @property
    def samples_queued(self) -> int:
        return self._samples_queued

    @property
    def samples_emitted(self) -> int:
        return self._samples_emitted

    @property
    def queued_ms(self) -> int:
        return int(self._samples_queued * 1000 / self._sample_rate_hz)

    @property
    def emitted_ms(self) -> int:
        return int(self._samples_emitted * 1000 / self._sample_rate_hz)

    def start_turn(self, turn_id: str, epoch: int) -> None:
        """Begin accounting for a new turn and epoch."""
        self._turn_id = turn_id
        self._epoch = epoch
        self._samples_queued = 0
        self._samples_emitted = 0

    def record_queued(self, samples: int) -> None:
        """Record samples queued into the playout mixer."""
        if samples > 0:
            self._samples_queued += samples

    def record_emitted(self, samples: int) -> None:
        """Record samples emitted downstream to the transport."""
        if samples > 0:
            self._samples_emitted += samples

    def queue_ms(self, ms: int) -> None:
        """Helper to queue audio duration in milliseconds."""
        if ms > 0:
            samples = int(ms * self._sample_rate_hz / 1000)
            self.record_queued(samples)

    def emit_ms(self, ms: int) -> None:
        """Helper to emit audio duration in milliseconds."""
        if ms > 0:
            samples = int(ms * self._sample_rate_hz / 1000)
            self.record_emitted(samples)

    def played_ms(
        self,
        downstream_latency_ms: int | None = None,
        transport_queue_ms: int = 0,
        clause_audio_total_ms: int | None = None,
    ) -> int:
        """Calculate playout position actually heard by user per ARCHITECTURE §8.2.

        Clamped strictly to [0, emitted_ms].
        """
        downstream_ms = (
            self._tuning.assumed_downstream_ms
            if downstream_latency_ms is None
            else max(0, downstream_latency_ms)
        )
        transport_q = max(0, transport_queue_ms)

        raw = self.emitted_ms - transport_q - downstream_ms
        if clause_audio_total_ms is not None:
            raw = min(raw, max(0, clause_audio_total_ms))

        return max(0, min(self.emitted_ms, raw))

    def snapshot(
        self,
        downstream_latency_ms: int | None = None,
        transport_queue_ms: int = 0,
        clause_audio_total_ms: int | None = None,
    ) -> PlayoutSnapshot:
        """Generate an immutable snapshot of current playout state."""
        played = self.played_ms(
            downstream_latency_ms=downstream_latency_ms,
            transport_queue_ms=transport_queue_ms,
            clause_audio_total_ms=clause_audio_total_ms,
        )
        return PlayoutSnapshot(
            turn_id=self._turn_id,
            epoch=self._epoch,
            samples_queued=self._samples_queued,
            samples_emitted=self._samples_emitted,
            sample_rate_hz=self._sample_rate_hz,
            queued_ms=self.queued_ms,
            emitted_ms=self.emitted_ms,
            played_ms=played,
        )

    def flush(
        self,
        downstream_latency_ms: int | None = None,
        transport_queue_ms: int = 0,
        clause_audio_total_ms: int | None = None,
    ) -> PlayoutSnapshot:
        """Drop un-emitted audio and return an immutable snapshot."""
        snap = self.snapshot(
            downstream_latency_ms=downstream_latency_ms,
            transport_queue_ms=transport_queue_ms,
            clause_audio_total_ms=clause_audio_total_ms,
        )
        self._turn_snapshots[(self._turn_id, self._epoch)] = snap
        return snap

    def get_snapshot(self, turn_id: str, epoch: int) -> PlayoutSnapshot | None:
        """Retrieve historical snapshot for a (turn_id, epoch) pair."""
        return self._turn_snapshots.get((turn_id, epoch))
