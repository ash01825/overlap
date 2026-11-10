# ruff: noqa: E402
"""Live integration smoke tests for Timbre TTS connecting to api.vachana.ai.

Gated behind pytest marker 'live'. Only runs when GNANI_API_KEY is present in env
and human initiates live tests. Never runs in fast or check gates.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pytest

from overlap.adapters.gnani.timbre import TimbreTtsAdapter, TimbreTtsConfig
from overlap.core.clock import CancelToken
from overlap.ports.tts import AudioChunk, AudioClip


@pytest.mark.live
def test_live_timbre_stream_smoke() -> None:
    """Smoke test: verify live Timbre TTS streaming synthesis over WebSocket."""
    api_key = os.environ.get("GNANI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("GNANI_API_KEY not set in environment")

    async def _run() -> None:
        adapter = TimbreTtsAdapter(TimbreTtsConfig(api_key=api_key))
        token = CancelToken()
        stream = await adapter.open_stream(
            voice_id="Nalini",
            lang="hi-IN",
            epoch=1,
            cancel=token,
        )

        await stream.send_text("नमस्ते, यह एक लाइव टेस्ट है।")
        await stream.finish()

        chunks: list[AudioChunk] = []
        async for chunk in stream.chunks():
            chunks.append(chunk)

        assert len(chunks) > 0
        total_pcm = sum(len(c.pcm) for c in chunks)
        assert total_pcm > 0
        for c in chunks:
            assert c.sample_rate == 16000
            assert len(c.pcm) % 2 == 0

    asyncio.run(_run())


@pytest.mark.live
def test_live_timbre_cancel_smoke() -> None:
    """Smoke test: verify live Timbre cancel mid-stream stops audio delivery."""
    api_key = os.environ.get("GNANI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("GNANI_API_KEY not set in environment")

    async def _run() -> None:
        adapter = TimbreTtsAdapter(TimbreTtsConfig(api_key=api_key))
        token = CancelToken()
        stream = await adapter.open_stream(
            voice_id="Nalini",
            lang="hi-IN",
            epoch=2,
            cancel=token,
        )

        await stream.send_text("यह एक बहुत लंबा वाक्य है जिसे बीच में रद्द करने का परीक्षण किया जा रहा है।")

        chunk_iter = stream.chunks()
        # Read first chunk
        first = await anext(chunk_iter)
        assert isinstance(first, AudioChunk)

        # Cancel mid-stream
        await stream.cancel()

        # Verify no further chunks delivered
        subsequent: list[AudioChunk] = []
        try:
            async for ch in chunk_iter:
                subsequent.append(ch)
        except StopAsyncIteration:
            pass

        assert len(subsequent) == 0

    asyncio.run(_run())


@pytest.mark.live
def test_live_timbre_synthesize_smoke() -> None:
    """Smoke test: verify live Timbre batch synthesis returns valid AudioClip."""
    api_key = os.environ.get("GNANI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("GNANI_API_KEY not set in environment")

    async def _run() -> None:
        adapter = TimbreTtsAdapter(TimbreTtsConfig(api_key=api_key))
        clip = await adapter.synthesize("हाँ", voice_id="Nalini", lang="hi-IN")
        assert isinstance(clip, AudioClip)
        assert clip.sample_rate == 16000
        assert len(clip.pcm) > 0
        assert len(clip.pcm) % 2 == 0

    asyncio.run(_run())
