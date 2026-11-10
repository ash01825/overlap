"""Telemetry collection, non-blocking JSONL persistence, and evaluation reporting."""

from __future__ import annotations

from overlap.telemetry.collector import TelemetryCollector, redact_fields, redact_text
from overlap.telemetry.jsonl import JsonlWriter, read_jsonl

__all__ = [
    "JsonlWriter",
    "TelemetryCollector",
    "read_jsonl",
    "redact_fields",
    "redact_text",
]
