"""Fake implementation of Vad for testing and scenarios."""

from __future__ import annotations

from collections import deque

from overlap.core.types import AudioFrame
from overlap.ports.vad import Vad, VadResult


class FakeVad(Vad):
    """Fake VAD processor implementing Vad port."""

    def __init__(
        self,
        default_is_speech: bool = False,
        default_confidence: float = 0.0,
        default_energy_db: float | None = None,
    ) -> None:
        self.default_is_speech = default_is_speech
        self.default_confidence = default_confidence
        self.default_energy_db = default_energy_db
        self._scripted: deque[VadResult] = deque()

    def set_speech(
        self,
        is_speech: bool,
        confidence: float = 1.0,
        energy_db: float | None = None,
    ) -> None:
        """Update default VAD response."""
        self.default_is_speech = is_speech
        self.default_confidence = confidence
        self.default_energy_db = energy_db

    def script_results(self, results: list[VadResult]) -> None:
        """Enqueue a sequence of canned VAD results."""
        self._scripted.extend(results)

    def process(self, frame: AudioFrame) -> VadResult:
        """Synchronously process an audio frame and return VadResult."""
        if self._scripted:
            return self._scripted.popleft()
        return VadResult(
            is_speech=self.default_is_speech,
            confidence=self.default_confidence,
            energy_db=self.default_energy_db,
        )
