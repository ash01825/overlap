"""Live capture smoke tests connecting to upstream services.

Gated behind pytest marker 'live'. Only runs when credentials are present in env
and human initiates live tests. Never runs in fast or check gates.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import ssl
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.mark.live
def test_live_stt_handshake_smoke() -> None:
    """Smoke test: verify live STT WebSocket handshake succeeds on api.vachana.ai."""
    api_key = os.environ.get("GNANI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("GNANI_API_KEY not set in environment")

    ctx = ssl.create_default_context()
    s = socket.create_connection(("api.vachana.ai", 443), timeout=10)
    ss = ctx.wrap_socket(s, server_hostname="api.vachana.ai")

    sec_key = base64.b64encode(os.urandom(16)).decode("ascii")
    req = (
        f"GET /stt/v3/stream HTTP/1.1\r\n"
        f"Host: api.vachana.ai\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {sec_key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)\r\n"
        f"X-API-Key-ID: {api_key}\r\n"
        f"language_code: hi-IN\r\n"
        f"model: prisma-v3.0\r\n"
        f"x-partials: true\r\n"
        f"\r\n"
    )
    ss.sendall(req.encode("utf-8"))
    resp_header = ss.recv(2048).decode("utf-8", errors="replace")
    ss.close()

    assert "101 Switching Protocols" in resp_header
    assert api_key not in resp_header


@pytest.mark.live
def test_live_tts_handshake_smoke() -> None:
    """Smoke test: verify live TTS WebSocket handshake succeeds on api.vachana.ai."""
    api_key = os.environ.get("GNANI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("GNANI_API_KEY not set in environment")

    ctx = ssl.create_default_context()
    s = socket.create_connection(("api.vachana.ai", 443), timeout=10)
    ss = ctx.wrap_socket(s, server_hostname="api.vachana.ai")

    sec_key = base64.b64encode(os.urandom(16)).decode("ascii")
    req = (
        f"GET /api/v1/tts HTTP/1.1\r\n"
        f"Host: api.vachana.ai\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {sec_key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)\r\n"
        f"X-API-Key-ID: {api_key}\r\n"
        f"\r\n"
    )
    ss.sendall(req.encode("utf-8"))
    resp_header = ss.recv(2048).decode("utf-8", errors="replace")
    ss.close()

    assert "101 Switching Protocols" in resp_header
    assert api_key not in resp_header


@pytest.mark.live
def test_live_llm_stream_smoke() -> None:
    """Smoke test: verify live LLM SSE streaming succeeds."""
    import urllib.request

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set in environment")

    ctx = ssl.create_default_context()
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    payload = json.dumps(
        {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Ping"}],
            "stream": True,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
        assert resp.status == 200
        first_line = resp.readline().decode("utf-8")
        assert "data:" in first_line
        assert api_key not in first_line
