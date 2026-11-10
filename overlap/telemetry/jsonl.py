"""Non-blocking background JSONL writer for telemetry events.

Per ARCHITECTURE §10 & §12:
- Bounded in-memory queue with O(1) drop-oldest policy when full.
- Background writer thread handles all disk I/O off the fast path (INV-L13).
- Tracks dropped_count and written_count counters; never raises into caller.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any


def _json_default(obj: Any) -> Any:
    """Serialize Enums, bytes, sets, and arbitrary objects safely for JSONL."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes len={len(obj)}>"
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    return str(obj)


class JsonlWriter:
    """Non-blocking bounded-queue JSONL writer with background thread flush."""

    def __init__(self, path: str | Path, max_queue_size: int = 1000) -> None:
        if max_queue_size <= 0:
            raise ValueError(f"max_queue_size must be > 0, got {max_queue_size}")

        self._path: Path = Path(path)
        self._max_queue_size: int = max_queue_size
        self._queue: deque[dict[str, Any]] = deque()
        self._cond: threading.Condition = threading.Condition(threading.Lock())

        self._dropped_count: int = 0
        self._written_count: int = 0
        self._write_errors: int = 0
        self._closed: bool = False

        self._worker: threading.Thread = threading.Thread(
            target=self._run_worker,
            name="overlap-jsonl-writer",
            daemon=True,
        )
        self._worker.start()

    @property
    def path(self) -> Path:
        """Target JSONL file path."""
        return self._path

    @property
    def max_queue_size(self) -> int:
        """Maximum buffered events before drop-oldest triggers."""
        return self._max_queue_size

    @property
    def dropped_count(self) -> int:
        """Total number of oldest records dropped due to queue overflow."""
        with self._cond:
            return self._dropped_count

    @property
    def written_count(self) -> int:
        """Total number of records flushed to disk."""
        with self._cond:
            return self._written_count

    @property
    def write_errors(self) -> int:
        """Total number of disk write errors encountered by background worker."""
        with self._cond:
            return self._write_errors

    @property
    def pending_count(self) -> int:
        """Current number of queued records awaiting background flush."""
        with self._cond:
            return len(self._queue)

    def enqueue(self, record: dict[str, Any]) -> bool:
        """Synchronously enqueue a record without blocking or performing disk I/O.

        If the queue is at capacity, drops the oldest record and increments dropped_count.
        Returns True if enqueued without drop, False if oldest was dropped or writer closed.
        """
        try:
            with self._cond:
                if self._closed:
                    return False
                dropped = False
                if len(self._queue) >= self._max_queue_size:
                    self._queue.popleft()
                    self._dropped_count += 1
                    dropped = True
                self._queue.append(record)
                self._cond.notify()
                return not dropped
        except Exception:
            return False

    def flush(self, timeout_s: float = 2.0) -> None:
        """Wait for the background thread to drain all currently queued records."""
        with self._cond:
            self._cond.notify()
            self._cond.wait_for(lambda: len(self._queue) == 0 or self._closed, timeout=timeout_s)

    def close(self, timeout_s: float = 2.0) -> None:
        """Flush remaining records and stop the background writer thread (idempotent)."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cond.notify_all()

        if self._worker.is_alive():
            self._worker.join(timeout=timeout_s)

    def _run_worker(self) -> None:
        """Background loop draining queued records to the JSONL file."""
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait()
                if not self._queue and self._closed:
                    self._cond.notify_all()
                    return

                batch: list[dict[str, Any]] = []
                while self._queue:
                    batch.append(self._queue.popleft())

            lines: list[str] = []
            for item in batch:
                try:
                    line = json.dumps(item, default=_json_default, ensure_ascii=False)
                    lines.append(line + "\n")
                except Exception:
                    with self._cond:
                        self._write_errors += 1

            if lines:
                try:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    with self._path.open("a", encoding="utf-8") as fh:
                        fh.writelines(lines)
                        fh.flush()
                    with self._cond:
                        self._written_count += len(lines)
                except Exception:
                    with self._cond:
                        self._write_errors += 1

            with self._cond:
                if not self._queue:
                    self._cond.notify_all()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read and parse all JSONL event records from disk."""
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            if isinstance(data, dict):
                records.append(data)
    return records
