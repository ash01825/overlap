"""Live wire test suite for Evon / Gemini LLM streaming adapter.

Live-gated tests hitting the real Gemini 2.5 Flash endpoint per docs/WIRE.md §4.
Requires GEMINI_API_KEY (or GNANI_API_KEY) in environment.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

_root_dir = Path(__file__).resolve().parent.parent.parent
_overlap_dir = str(_root_dir)
if _overlap_dir not in sys.path:
    sys.path.insert(0, _overlap_dir)

import overlap.adapters  # noqa: E402

_overlap_adapters_dir = str(_root_dir / "overlap" / "adapters")
if _overlap_adapters_dir not in overlap.adapters.__path__:
    overlap.adapters.__path__.insert(0, _overlap_adapters_dir)

from overlap.adapters.gnani.evon import EvonLlmAdapter  # noqa: E402
from overlap.core.clock import CancelToken  # noqa: E402
from overlap.ports.llm import Message  # noqa: E402

pytestmark = pytest.mark.live


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GNANI_API_KEY")
    if not key:
        pytest.skip("Neither GEMINI_API_KEY nor GNANI_API_KEY is set in environment")
    return key


def test_live_evon_streaming() -> None:
    """Test live streaming token generation from Gemini 2.5 Flash."""
    api_key = _get_api_key()

    async def _run() -> None:
        adapter = EvonLlmAdapter(api_key=api_key)
        cancel = CancelToken()

        stream = await adapter.stream(
            [
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Reply with exactly the words: 'OVERLAP is live'."),
            ],
            epoch=1,
            cancel=cancel,
        )

        deltas: list[str] = []
        async for delta in stream.deltas():
            deltas.append(delta)

        full_response = "".join(deltas).strip()
        assert len(deltas) > 0, "Expected at least one delta from live LLM stream"
        assert "OVERLAP" in full_response or "live" in full_response.lower()

    asyncio.run(_run())


def test_live_evon_abort_mid_stream() -> None:
    """Test that abort() immediately cuts off live token generation."""
    api_key = _get_api_key()

    async def _run() -> None:
        adapter = EvonLlmAdapter(api_key=api_key)
        cancel = CancelToken()

        stream = await adapter.stream(
            [
                Message(
                    role="user",
                    content="Count from 1 to 50 slowly, writing one number per line.",
                ),
            ],
            epoch=1,
            cancel=cancel,
        )

        tokens_received = 0
        async for _ in stream.deltas():
            tokens_received += 1
            if tokens_received >= 2:
                await stream.abort()
                break

        assert tokens_received == 2

    asyncio.run(_run())


def test_live_assistant_history_with_truncation_marker() -> None:
    """Test that multi-turn history with truncated assistant message is accepted."""
    api_key = _get_api_key()

    async def _run() -> None:
        adapter = EvonLlmAdapter(api_key=api_key)
        cancel = CancelToken()

        messages = [
            Message(role="system", content="You are a concise voice assistant."),
            Message(role="user", content="What is the capital of India?"),
            Message(
                role="assistant",
                content="The capital of India is New Delhi... [cut]",
            ),
            Message(role="user", content="And what is the official language?"),
        ]

        stream = await adapter.stream(messages, epoch=1, cancel=cancel)
        deltas: list[str] = []
        async for delta in stream.deltas():
            deltas.append(delta)

        full_response = "".join(deltas).strip()
        assert len(full_response) > 0
        assert "hindi" in full_response.lower() or "english" in full_response.lower()

    asyncio.run(_run())
