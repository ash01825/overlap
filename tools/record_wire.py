"""Wire traffic recorder and secret scrubber for OVERLAP.

Captures real WebSocket and HTTP/SSE traffic between OVERLAP adapters
and upstream services (Gnani Prisma STT, Gnani Timbre TTS, Gemini/Evon LLM),
scrubs all secrets/PII, and writes hermetic JSONL fixtures conforming to
tests/fixtures/wire/README.md.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import select
import socket
import ssl
import struct
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Secret replacement token
REDACTED = "[REDACTED]"

# Regex patterns for automatic secret scrubbing
HEADER_PATTERNS = [
    (
        re.compile(r"(?i)(X-API-Key-ID|X-API-Key|x-goog-api-key)\s*:\s*[^\r\n]+"),
        r"\1: [REDACTED]",
    ),
    (
        re.compile(r"(?i)(Authorization\s*:\s*Bearer\s+)[^\r\n]+"),
        r"\1[REDACTED]",
    ),
]

QUERY_PARAM_PATTERN = re.compile(
    r"(?i)([\?&](?:api_key|api-key|apiKey|key|token|secret|auth)=)[^&\s]+"
)

JSON_TOKEN_PATTERN = re.compile(
    r'(?i)("(?:\w*_)?(?:api_?key|token|secret|password|auth)"\s*:\s*)"(?:[^"\\]|\\.)*"'
)

BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_\-\.]{10,}\b")


def _get_known_secrets() -> list[str]:
    """Retrieve known environment secret values to ensure zero leak."""
    keys = ["GNANI_API_KEY", "GEMINI_API_KEY", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
    secrets: list[str] = []
    for k in keys:
        val = os.environ.get(k, "").strip()
        if val and len(val) >= 6:
            secrets.append(val)
    return secrets


def scrub_secrets(text: str, secret_values: Sequence[str] | None = None) -> str:
    """Scrub sensitive credentials from headers, query params, JSON payloads, and text."""
    if not text:
        return text

    scrubbed = text

    # 1. Exact match replacements from env or passed secrets
    targets = list(secret_values) if secret_values is not None else _get_known_secrets()
    for s in targets:
        if s and s != REDACTED and s in scrubbed:
            scrubbed = scrubbed.replace(s, REDACTED)

    # 2. Header scrubbing
    for pattern, repl in HEADER_PATTERNS:
        scrubbed = pattern.sub(repl, scrubbed)

    # 3. Query parameter scrubbing
    scrubbed = QUERY_PARAM_PATTERN.sub(r"\1" + REDACTED, scrubbed)

    # 4. JSON field scrubbing
    scrubbed = JSON_TOKEN_PATTERN.sub(r'\1"' + REDACTED + '"', scrubbed)

    # 5. Bearer token scrubbing
    scrubbed = BEARER_PATTERN.sub("Bearer " + REDACTED, scrubbed)

    return scrubbed


def scrub_wire_frame(
    frame: dict[str, Any], secret_values: Sequence[str] | None = None
) -> dict[str, Any]:
    """Return a copy of the wire frame with all secrets scrubbed."""
    out = dict(frame)
    if "payload" in out and isinstance(out["payload"], str):
        out["payload"] = scrub_secrets(out["payload"], secret_values)
    # Ensure assumed flag is False or not present
    if "assumed" in out:
        out["assumed"] = False
    return out


# =============================================================================
# Minimal RFC 6455 WebSocket client (stdlib only)
# =============================================================================


def _make_ws_frame(opcode: int, data: bytes, masked: bool = True) -> bytes:
    """Encode an RFC 6455 WebSocket frame."""
    b1 = 0x80 | (opcode & 0x0F)
    length = len(data)
    if length < 126:
        b2 = (0x80 if masked else 0) | length
        header = bytes([b1, b2])
    elif length < 65536:
        b2 = (0x80 if masked else 0) | 126
        header = bytes([b1, b2]) + struct.pack(">H", length)
    else:
        b2 = (0x80 if masked else 0) | 127
        header = bytes([b1, b2]) + struct.pack(">Q", length)

    if masked:
        mask = os.urandom(4)
        masked_data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        return header + mask + masked_data
    return header + data


def _recv_ws_frame(ss: ssl.SSLSocket) -> tuple[int, bytes] | None:
    """Decode a single inbound WebSocket frame."""
    header = ss.recv(2)
    if not header or len(header) < 2:
        return None
    b1, b2 = header[0], header[1]
    opcode = b1 & 0x0F
    masked = (b2 >> 7) & 1
    length = b2 & 0x7F
    if length == 126:
        ext = ss.recv(2)
        if len(ext) < 2:
            return None
        length = struct.unpack(">H", ext)[0]
    elif length == 127:
        ext = ss.recv(8)
        if len(ext) < 8:
            return None
        length = struct.unpack(">Q", ext)[0]

    mask = ss.recv(4) if masked else None
    data = bytearray()
    while len(data) < length:
        chunk = ss.recv(min(4096, length - len(data)))
        if not chunk:
            break
        data.extend(chunk)

    if masked and mask:
        data = bytearray(b ^ mask[i % 4] for i, b in enumerate(data))

    return opcode, bytes(data)


# =============================================================================
# Live Wire Recorders
# =============================================================================


def record_prisma_stt(
    audio_pcm: bytes,
    *,
    language_code: str = "hi-IN",
    model: str = "prisma-v3.0",
    format_style: str = "verbatim",
    api_key: str,
    silence_post_ms: int = 1500,
) -> list[dict[str, Any]]:
    """Record live Prisma STT WebSocket session exchanging audio frames."""
    ctx = ssl.create_default_context()
    s = socket.create_connection(("api.vachana.ai", 443), timeout=10)
    ss = ctx.wrap_socket(s, server_hostname="api.vachana.ai")

    sec_key = base64.b64encode(os.urandom(16)).decode("ascii")
    req_headers = (
        f"GET /stt/v3/stream HTTP/1.1\r\n"
        f"Host: api.vachana.ai\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {sec_key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)\r\n"
        f"X-API-Key-ID: {api_key}\r\n"
        f"language_code: {language_code}\r\n"
        f"model: {model}\r\n"
        f"x-partials: true\r\n"
        f"format: {format_style}\r\n"
        f"\r\n"
    )

    t0 = time.monotonic()
    ss.sendall(req_headers.encode("utf-8"))
    ss.recv(2048)  # Upgrade ack

    frames: list[dict[str, Any]] = []

    def log_inbound() -> None:
        while True:
            r, _, _ = select.select([ss], [], [], 0.0)
            if not r:
                break
            f = _recv_ws_frame(ss)
            if not f:
                break
            op, data = f
            t_ms = int((time.monotonic() - t0) * 1000)
            if op == 1:
                frames.append(
                    {
                        "t_ms": t_ms,
                        "dir": "in",
                        "kind": "text",
                        "payload": data.decode("utf-8", errors="replace"),
                    }
                )
            elif op == 2:
                frames.append(
                    {
                        "t_ms": t_ms,
                        "dir": "in",
                        "kind": "binary",
                        "b64": base64.b64encode(data).decode("ascii"),
                    }
                )

    # Initial frame (connected)
    f_conn = _recv_ws_frame(ss)
    if f_conn:
        t_ms = int((time.monotonic() - t0) * 1000)
        frames.append(
            {
                "t_ms": t_ms,
                "dir": "in",
                "kind": "text",
                "payload": f_conn[1].decode("utf-8", errors="replace"),
            }
        )

    # Stream audio in 640-byte chunks (20 ms @ 16 kHz)
    chunk_size = 640
    for i in range(0, len(audio_pcm), chunk_size):
        chunk = audio_pcm[i : i + chunk_size]
        t_ms = int((time.monotonic() - t0) * 1000)
        frames.append(
            {
                "t_ms": t_ms,
                "dir": "out",
                "kind": "binary",
                "b64": base64.b64encode(chunk).decode("ascii"),
            }
        )
        ss.sendall(_make_ws_frame(2, chunk, masked=True))
        time.sleep(0.02)
        log_inbound()

    # Stream post-speech silence so VAD confirms end of utterance
    silence_bytes = int(16000 * 2 * (silence_post_ms / 1000.0))
    for i in range(0, silence_bytes, chunk_size):
        chunk = b"\x00" * min(chunk_size, silence_bytes - i)
        t_ms = int((time.monotonic() - t0) * 1000)
        frames.append(
            {
                "t_ms": t_ms,
                "dir": "out",
                "kind": "binary",
                "b64": base64.b64encode(chunk).decode("ascii"),
            }
        )
        ss.sendall(_make_ws_frame(2, chunk, masked=True))
        time.sleep(0.02)
        log_inbound()

    # Drain remaining inbound frames up to timeout
    ss.setblocking(True)
    ss.settimeout(3.0)
    try:
        while True:
            f = _recv_ws_frame(ss)
            if not f:
                break
            op, data = f
            t_ms = int((time.monotonic() - t0) * 1000)
            if op == 1:
                frames.append(
                    {
                        "t_ms": t_ms,
                        "dir": "in",
                        "kind": "text",
                        "payload": data.decode("utf-8", errors="replace"),
                    }
                )
    except (TimeoutError, OSError):
        pass

    ss.close()
    return [scrub_wire_frame(fr, secret_values=[api_key]) for fr in frames]


def record_timbre_tts(
    text: str,
    *,
    voice: str = "Nalini",
    model: str = "timbre-v2.5",
    language: str = "hi-IN",
    api_key: str,
    cancel_after_chunks: int | None = None,
) -> list[dict[str, Any]]:
    """Record live Timbre TTS WebSocket session."""
    ctx = ssl.create_default_context()
    s = socket.create_connection(("api.vachana.ai", 443), timeout=10)
    ss = ctx.wrap_socket(s, server_hostname="api.vachana.ai")

    sec_key = base64.b64encode(os.urandom(16)).decode("ascii")
    req_headers = (
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

    t0 = time.monotonic()
    ss.sendall(req_headers.encode("utf-8"))
    ss.recv(2048)

    frames: list[dict[str, Any]] = []

    req_payload = {
        "text": text,
        "voice": voice,
        "model": model,
        "language": language,
        "speed": 1.0,
        "audio_config": {
            "sample_rate": 16000,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "wav",
        },
    }
    raw_json = json.dumps(req_payload)
    t_ms = int((time.monotonic() - t0) * 1000)
    frames.append(
        {
            "t_ms": t_ms,
            "dir": "out",
            "kind": "text",
            "payload": raw_json,
        }
    )
    ss.sendall(_make_ws_frame(1, raw_json.encode("utf-8"), masked=True))

    audio_chunks_seen = 0
    while True:
        f = _recv_ws_frame(ss)
        if not f:
            break
        op, data = f
        t_ms = int((time.monotonic() - t0) * 1000)
        if op == 1:
            txt = data.decode("utf-8", errors="replace")
            frames.append(
                {
                    "t_ms": t_ms,
                    "dir": "in",
                    "kind": "text",
                    "payload": txt,
                }
            )
            try:
                parsed = json.loads(txt)
                if parsed.get("type") == "audio":
                    audio_chunks_seen += 1
                if cancel_after_chunks is not None and audio_chunks_seen >= cancel_after_chunks:
                    # Send close frame
                    close_ms = int((time.monotonic() - t0) * 1000)
                    frames.append(
                        {
                            "t_ms": close_ms,
                            "dir": "out",
                            "kind": "text",
                            "payload": json.dumps({"action": "cancel", "code": 1000}),
                        }
                    )
                    ss.sendall(_make_ws_frame(8, struct.pack(">H", 1000), masked=True))
                    break
                if parsed.get("data", {}).get("is_final") or parsed.get("is_final"):
                    break
            except Exception:
                pass
        elif op == 8:
            frames.append(
                {
                    "t_ms": t_ms,
                    "dir": "in",
                    "kind": "text",
                    "payload": json.dumps({"type": "close", "code": 1000}),
                }
            )
            break

    ss.close()
    return [scrub_wire_frame(fr, secret_values=[api_key]) for fr in frames]


def record_llm_stream(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    abort_after_chunks: int | None = None,
) -> list[dict[str, Any]]:
    """Record live LLM SSE streaming session."""
    ctx = ssl.create_default_context()
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

    req_payload = {
        "model": "gemini-2.5-flash",
        "messages": messages,
        "stream": True,
    }
    raw_json = json.dumps(req_payload)

    t0 = time.monotonic()
    frames: list[dict[str, Any]] = [
        {
            "t_ms": 0,
            "dir": "out",
            "kind": "text",
            "payload": raw_json,
        }
    ]

    req = urllib.request.Request(
        url,
        data=raw_json.encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    chunks_seen = 0
    with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
        while True:
            line_b = resp.readline()
            if not line_b:
                break
            line = line_b.decode("utf-8")
            if not line.strip():
                continue
            t_ms = int((time.monotonic() - t0) * 1000)
            frames.append(
                {
                    "t_ms": t_ms,
                    "dir": "in",
                    "kind": "text",
                    "payload": line.rstrip("\r\n"),
                }
            )
            chunks_seen += 1
            if abort_after_chunks is not None and chunks_seen >= abort_after_chunks:
                # Abort client stream by closing connection
                abort_ms = int((time.monotonic() - t0) * 1000)
                frames.append(
                    {
                        "t_ms": abort_ms,
                        "dir": "out",
                        "kind": "text",
                        "payload": json.dumps({"action": "abort"}),
                    }
                )
                break

    return [scrub_wire_frame(fr, secret_values=[api_key]) for fr in frames]


def write_jsonl(path: Path, frames: list[dict[str, Any]]) -> None:
    """Write frames to a JSONL file ensuring parent directories exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr, ensure_ascii=False) + "\n")


def synthesize_clip_raw(text: str, api_key: str) -> bytes:
    """Helper to synthesize audio via Timbre REST for STT input tests."""
    ctx = ssl.create_default_context()
    req = urllib.request.Request(
        "https://api.vachana.ai/api/v1/tts/inference",
        data=json.dumps(
            {
                "text": text,
                "voice": "Nalini",
                "model": "timbre-v2.5",
                "language": "hi-IN",
                "speed": 1.0,
                "audio_config": {
                    "sample_rate": 16000,
                    "num_channels": 1,
                    "encoding": "linear_pcm",
                    "container": "raw",
                },
            }
        ).encode("utf-8"),
        headers={
            "X-API-Key-ID": api_key,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
        return bytes(resp.read())


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for recording live wire fixtures."""
    parser = argparse.ArgumentParser(description="Record scrubbed wire traffic fixtures.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/fixtures/wire"),
        help="Base directory for wire fixtures.",
    )
    parser.add_argument(
        "--service",
        choices=["all", "prisma", "timbre", "evon"],
        default="all",
        help="Which service to record.",
    )
    args = parser.parse_args(argv)

    gnani_key = os.environ.get("GNANI_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not gnani_key:
        sys.stderr.write("Error: GNANI_API_KEY must be set in env\n")
        return 1

    out_base = args.out_dir
    print(f"Recording wire fixtures to {out_base}...")

    if args.service in ("all", "timbre"):
        print("Recording Timbre TTS fixtures...")
        # 1. Full stream
        f_full = record_timbre_tts("नमस्ते, आप कैसे हैं?", api_key=gnani_key)
        write_jsonl(out_base / "timbre" / "full_stream.jsonl", f_full)
        print("  -> timbre/full_stream.jsonl")

        # 2. Cancel mid-stream
        f_cancel = record_timbre_tts(
            "यह एक बहुत लंबा वाक्य है जिसे बीच में रद्द करने का परीक्षण किया जा रहा है।",
            api_key=gnani_key,
            cancel_after_chunks=1,
        )
        write_jsonl(out_base / "timbre" / "cancel_mid_stream.jsonl", f_cancel)
        print("  -> timbre/cancel_mid_stream.jsonl")

    if args.service in ("all", "prisma"):
        print("Recording Prisma STT fixtures...")
        # Synthesize audio clips to stream into STT
        audio_short = synthesize_clip_raw("नमस्ते आप कैसे हैं", gnani_key)
        f_short = record_prisma_stt(audio_short, api_key=gnani_key)
        write_jsonl(out_base / "prisma" / "short_utterance.jsonl", f_short)
        print("  -> prisma/short_utterance.jsonl")

        audio_codeswitch = synthesize_clip_raw("haan meeting cancel ho gayi hai", gnani_key)
        f_cs = record_prisma_stt(
            audio_codeswitch,
            language_code="indic-auto",
            model="prisma-v3.0",
            api_key=gnani_key,
        )
        write_jsonl(out_base / "prisma" / "code_switch.jsonl", f_cs)
        print("  -> prisma/code_switch.jsonl")

        # Silence (1.5s pure silence)
        silence_pcm = b"\x00" * 48000
        f_silence = record_prisma_stt(silence_pcm, api_key=gnani_key, silence_post_ms=500)
        write_jsonl(out_base / "prisma" / "silence.jsonl", f_silence)
        print("  -> prisma/silence.jsonl")

    if args.service in ("all", "evon"):
        if not gemini_key:
            sys.stderr.write("Warning: GEMINI_API_KEY not set; skipping LLM recording\n")
        else:
            print("Recording Evon/Gemini LLM fixtures...")
            msgs = [{"role": "user", "content": "Say 'hello world' and one sentence."}]
            f_stream = record_llm_stream(msgs, api_key=gemini_key)
            write_jsonl(out_base / "evon" / "stream.jsonl", f_stream)
            print("  -> evon/stream.jsonl")

            f_abort = record_llm_stream(msgs, api_key=gemini_key, abort_after_chunks=2)
            write_jsonl(out_base / "evon" / "abort.jsonl", f_abort)
            print("  -> evon/abort.jsonl")

    print("Wire fixtures recording complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
