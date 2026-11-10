"""In-memory duplex FakeWebSocket replaying JSONL wire transcripts."""

from __future__ import annotations

import asyncio
import base64
import json
import warnings
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from overlap.core.clock import VirtualClock


class AssumedFixtureWarning(UserWarning):
    """Warning emitted when a wire fixture with 'assumed: true' is loaded."""


@dataclass(frozen=True, slots=True)
class WireMessage:
    """Wire frame in JSONL transcript."""

    t_ms: int
    direction: str  # "in" | "out"
    kind: str  # "text" | "binary"
    payload: str | None = None
    data: bytes | None = None
    assumed: bool = False


class FakeWebSocket:
    """Fake WebSocket replaying JSONL wire transcripts in virtual time and capturing sent frames."""

    def __init__(self, clock: VirtualClock | None = None) -> None:
        self.clock = clock
        self.inbound_queue: deque[WireMessage] = deque()
        self.sent_messages: list[WireMessage] = []
        self.is_assumed: bool = False
        self._notify_event = asyncio.Event()
        self._closed: bool = False

    def load_jsonl(self, path_or_content: str | Path) -> None:
        """Parse JSONL wire transcript file or string content."""
        if isinstance(path_or_content, Path) or (
            isinstance(path_or_content, str)
            and "\n" not in path_or_content
            and Path(path_or_content).exists()
        ):
            path = Path(path_or_content)
            content = path.read_text(encoding="utf-8")
        else:
            content = str(path_or_content)

        inbound: list[WireMessage] = []
        has_assumed = False

        for line_num, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at line {line_num}: {err}") from err

            t_ms = int(item.get("t_ms", 0))
            direction = str(item.get("dir", "in"))
            kind = str(item.get("kind", "text"))
            payload = item.get("payload")
            b64_str = item.get("b64")
            data = base64.b64decode(b64_str) if b64_str else None
            assumed = bool(item.get("assumed", False))

            if assumed:
                has_assumed = True

            msg = WireMessage(
                t_ms=t_ms,
                direction=direction,
                kind=kind,
                payload=payload,
                data=data,
                assumed=assumed,
            )

            if direction == "in":
                inbound.append(msg)

        if has_assumed:
            self.is_assumed = True
            warnings.warn(
                "Using assumed wire fixture (hand-written fixture flagged 'assumed: true')",
                AssumedFixtureWarning,
                stacklevel=2,
            )

        # Stable sort preserving transcript order for identical t_ms
        inbound.sort(key=lambda m: m.t_ms)
        self.inbound_queue = deque(inbound)
        if self.inbound_queue:
            self._notify_event.set()

    async def send_text(self, text: str) -> None:
        """Record outbound text message."""
        now_ms = (self.clock.now_ns() // 1_000_000) if self.clock is not None else 0
        msg = WireMessage(
            t_ms=now_ms,
            direction="out",
            kind="text",
            payload=text,
            data=None,
        )
        self.sent_messages.append(msg)

    async def send_bytes(self, data: bytes) -> None:
        """Record outbound binary message."""
        now_ms = (self.clock.now_ns() // 1_000_000) if self.clock is not None else 0
        msg = WireMessage(
            t_ms=now_ms,
            direction="out",
            kind="binary",
            payload=None,
            data=data,
        )
        self.sent_messages.append(msg)

    async def send(self, data: str | bytes) -> None:
        """Record outbound message."""
        if isinstance(data, str):
            await self.send_text(data)
        else:
            await self.send_bytes(data)

    async def recv(self) -> str | bytes:
        """Receive the next inbound message honoring virtual time."""
        while not self._closed:
            if not self.inbound_queue:
                self._notify_event.clear()
                await self._notify_event.wait()
                if self._closed:
                    raise asyncio.IncompleteReadError(b"", None)
                continue

            msg = self.inbound_queue[0]
            if self.clock is not None:
                now_ms = self.clock.now_ns() // 1_000_000
                if now_ms < msg.t_ms:
                    # Delay until message t_ms
                    fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                    delay_ms = msg.t_ms - now_ms
                    self.clock.call_later(
                        delay_ms,
                        lambda f=fut: not f.done() and f.set_result(None),
                    )
                    await fut
                    continue

            self.inbound_queue.popleft()
            if msg.kind == "text":
                return msg.payload or ""
            return msg.data or b""

        raise asyncio.IncompleteReadError(b"", None)

    async def recv_text(self) -> str:
        """Receive next text frame."""
        res = await self.recv()
        if isinstance(res, str):
            return res
        return res.decode("utf-8")

    async def recv_bytes(self) -> bytes:
        """Receive next binary frame."""
        res = await self.recv()
        if isinstance(res, bytes):
            return res
        return res.encode("utf-8")

    async def __aiter__(self) -> AsyncIterator[str | bytes]:
        """Iterate over inbound messages until closed or empty."""
        while not self._closed and self.inbound_queue:
            try:
                yield await self.recv()
            except asyncio.IncompleteReadError:
                break

    async def close(self) -> None:
        """Close fake connection."""
        self._closed = True
        self._notify_event.set()
