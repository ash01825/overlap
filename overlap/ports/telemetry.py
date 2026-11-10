"""Telemetry sink port protocol definitions.

Pure port interface: typing.Protocol only, no implementation logic, no asyncio import.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TelemetrySink(Protocol):
    """Telemetry event and metric sink."""

    def emit(self, name: str, ts_ns: int, **fields: Any) -> None:
        """Synchronously emit a named telemetry event (non-blocking)."""
        ...


class NullSink:
    """Default no-op telemetry sink."""

    def emit(self, name: str, ts_ns: int, **fields: Any) -> None:
        """Discard emitted telemetry event."""
        pass
