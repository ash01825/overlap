"""Fake recording sink for TelemetrySink port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from overlap.ports.telemetry import TelemetrySink


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """Recorded telemetry event."""

    event: str
    ts_ns: int
    fields: dict[str, Any]


class RecordingSink(TelemetrySink):
    """Telemetry sink that captures all emitted events in memory for test assertions."""

    def __init__(self) -> None:
        self.records: list[TelemetryRecord] = []

    def emit(self, name: str, ts_ns: int, **fields: Any) -> None:
        """Capture telemetry event synchronously."""
        rec = TelemetryRecord(event=name, ts_ns=ts_ns, fields=dict(fields))
        self.records.append(rec)

    def has_event(self, event: str) -> bool:
        """Check if an event with the given name was emitted."""
        return any(r.event == event for r in self.records)

    def events_named(self, event: str) -> list[TelemetryRecord]:
        """Return all records matching event name."""
        return [r for r in self.records if r.event == event]

    def clear(self) -> None:
        """Clear recorded events."""
        self.records.clear()
