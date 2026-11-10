"""Voice Activity Detection (VAD) port protocol definitions.

Pure port interface: typing.Protocol only, no implementation logic, no asyncio import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from overlap.core.types import AudioFrame


@dataclass(frozen=True, slots=True)
class VadResult:
    """Result of VAD classification on a single audio frame."""

    is_speech: bool
    confidence: float = 0.0
    energy_db: float | None = None


@runtime_checkable
class Vad(Protocol):
    """Voice Activity Detector interface."""

    def process(self, frame: AudioFrame) -> VadResult:
        """Process an audio frame and return speech detection result (sync; <= 1 frame latency)."""
        ...
