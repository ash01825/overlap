"""LLM (Large Language Model) port and stream protocol definitions.

Pure port interface: typing.Protocol only, no implementation logic, no asyncio import.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from overlap.core.clock import CancelToken


@dataclass(frozen=True, slots=True)
class Message:
    """Chat message exchanged with the language model."""

    role: Literal["system", "user", "assistant"]
    content: str


@runtime_checkable
class LlmStream(Protocol):
    """Active streaming LLM inference session."""

    def deltas(self) -> AsyncIterator[str]:
        """Asynchronous iterator yielding generated text tokens/deltas."""
        ...

    def abort(self) -> Coroutine[Any, Any, None]:
        """Abort inference stream. Idempotent; closes HTTP/WS stream."""
        ...


@runtime_checkable
class LlmPort(Protocol):
    """Port for streaming LLM completions."""

    def stream(
        self, messages: Sequence[Message], *, epoch: int, cancel: CancelToken
    ) -> Coroutine[Any, Any, LlmStream]:
        """Open an LLM completion stream for the given messages."""
        ...
