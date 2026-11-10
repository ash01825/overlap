"""TelemetrySink implementation with PII redaction and non-blocking JSONL writing.

Per ARCHITECTURE §10 & §12:
- Implements `TelemetrySink` protocol (`emit(name, ts_ns, **fields) -> None`).
- Synchronous and strictly non-blocking on the fast path; never raises.
- Optional PII/transcript redaction replaces sensitive text fields with SHA-256 hash
  prefix and character/word length metadata while preserving numeric counts.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from overlap.core.metrics import DerivedMetrics, derive_metrics
from overlap.ports.telemetry import TelemetrySink
from overlap.telemetry.jsonl import JsonlWriter

DEFAULT_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "text",
        "partial_text",
        "transcript",
        "content",
        "turn_text",
        "user_text",
        "heard_text",
        "full_text",
        "utterance",
        "prompt",
    }
)


def redact_text(value: str) -> str:
    """Replace raw transcript/PII text with a deterministic SHA-256 prefix and length."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    words = len(value.split())
    return f"<redacted sha256={digest} len={len(value)} words={words}>"


def redact_fields(
    fields: Mapping[str, Any],
    sensitive_keys: frozenset[str] = DEFAULT_SENSITIVE_KEYS,
) -> dict[str, Any]:
    """Recursively redact sensitive text keys while preserving word counts for metrics."""
    out: dict[str, Any] = {}
    for k, v in fields.items():
        key_lower = k.lower()
        if k in sensitive_keys or key_lower in sensitive_keys:
            if isinstance(v, str):
                out[k] = redact_text(v)
            elif isinstance(v, (list, tuple)):
                out[k] = [redact_text(str(item)) for item in v]
            else:
                out[k] = v
        elif key_lower in ("heard_words", "truth_words") and isinstance(v, str):
            # Preserve numeric word count for truncation_error_words while redacting text
            out[k] = len(v.split())
            out[f"{k}_redacted"] = redact_text(v)
        elif key_lower == "words" and isinstance(v, (list, tuple)):
            out[k] = f"<redacted_words count={len(v)}>"
        elif isinstance(v, Mapping):
            out[k] = redact_fields(v, sensitive_keys=sensitive_keys)
        else:
            out[k] = v
    return out


class TelemetryCollector(TelemetrySink):
    """Production TelemetrySink with in-memory ring buffer, redaction, and JSONL writer."""

    def __init__(
        self,
        jsonl_path: str | Path | None = None,
        *,
        writer: JsonlWriter | None = None,
        redact: bool = False,
        max_queue_size: int = 1000,
        max_memory_events: int = 10_000,
    ) -> None:
        self._redact: bool = redact
        self._max_memory_events: int = max_memory_events
        self._events: deque[dict[str, Any]] = deque(maxlen=max_memory_events)
        self._emit_errors: int = 0

        if writer is not None:
            self._writer: JsonlWriter | None = writer
        elif jsonl_path is not None:
            self._writer = JsonlWriter(jsonl_path, max_queue_size=max_queue_size)
        else:
            self._writer = None

    @property
    def redact(self) -> bool:
        """True if transcript/PII redaction is enabled."""
        return self._redact

    @property
    def writer(self) -> JsonlWriter | None:
        """Attached non-blocking JsonlWriter, if any."""
        return self._writer

    @property
    def dropped_count(self) -> int:
        """Number of events dropped by the bounded JSONL queue."""
        return self._writer.dropped_count if self._writer is not None else 0

    @property
    def events(self) -> list[dict[str, Any]]:
        """Snapshot of recorded telemetry events in memory."""
        return list(self._events)

    def emit(self, name: str, ts_ns: int, **fields: Any) -> None:
        """Synchronously emit a named telemetry event without blocking or raising."""
        try:
            clean_fields = redact_fields(fields) if self._redact else dict(fields)
            record: dict[str, Any] = {
                "name": str(name),
                "ts_ns": int(ts_ns),
                **clean_fields,
            }
            self._events.append(record)
            if self._writer is not None:
                self._writer.enqueue(record)
        except Exception:
            self._emit_errors += 1

    def derive_metrics(self) -> DerivedMetrics:
        """Derive all ARCHITECTURE §12 metrics from collected events."""
        return derive_metrics(list(self._events))

    def flush(self, timeout_s: float = 2.0) -> None:
        """Flush pending events to the JSONL file if a writer is configured."""
        if self._writer is not None:
            self._writer.flush(timeout_s=timeout_s)

    def close(self, timeout_s: float = 2.0) -> None:
        """Flush and close the underlying JSONL writer."""
        if self._writer is not None:
            self._writer.close(timeout_s=timeout_s)
