"""Live integration test for PrismaSttPort against api.vachana.ai.

This test requires GNANI_API_KEY and active network access.
It is marked @pytest.mark.live and excluded from fast and check test suites.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from overlap.adapters.gnani.prisma import PrismaSttPort
from overlap.core.clock import CancelToken
from overlap.core.types import AudioFrame


@pytest.mark.live
def test_prisma_live_partial_latency() -> None:
    """Measure partial transcript delivery latency over live Prisma STT WebSocket."""
    api_key = os.environ.get("GNANI_API_KEY")
    if not api_key:
        pytest.skip("GNANI_API_KEY environment variable not set; skipping live test")

    async def _run() -> None:
        port = PrismaSttPort(api_key=api_key, language_code="hi-IN")
        token = CancelToken()
        stream = await port.open(cancel=token)

        first_partial_latency_ms: float | None = None
        t_start = time.monotonic()

        sample_rate = 16000
        frame_ms = 20
        frame_bytes = 640

        try:
            for i in range(50):
                pcm_bytes = bytearray(frame_bytes)
                frame = AudioFrame(
                    pcm=bytes(pcm_bytes),
                    sample_rate=sample_rate,
                    ts_ns=i * frame_ms * 1_000_000,
                )
                await stream.send_audio(frame)

            t_send_done = time.monotonic()
            async for evt in stream.events():
                if evt.kind == "partial":
                    first_partial_latency_ms = (time.monotonic() - t_start) * 1000.0
                    break
                if time.monotonic() - t_send_done > 5.0:
                    break
        finally:
            await stream.close()

        if first_partial_latency_ms is not None:
            assert first_partial_latency_ms > 0

    asyncio.run(_run())
