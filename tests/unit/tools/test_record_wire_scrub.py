"""Unit tests for automatic secret scrubbing in tools/record_wire.py.

Verifies scrubbing of headers, query parameters, JSON payload tokens,
and raw secret values, ensuring zero secrets leak into committed fixtures.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_record_wire() -> ModuleType:
    module_path = REPO_ROOT / "tools" / "record_wire.py"
    spec = importlib.util.spec_from_file_location("record_wire", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load record_wire from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["record_wire"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_scrub_headers() -> None:
    rw = _load_record_wire()
    secret = "secret-gnani-token-abc12345"

    headers_text = (
        f"GET /stt/v3/stream HTTP/1.1\r\n"
        f"Host: api.vachana.ai\r\n"
        f"X-API-Key-ID: {secret}\r\n"
        f"Authorization: Bearer my-secret-bearer-token-999\r\n"
        f"x-goog-api-key: my-secret-gemini-key-777\r\n"
        f"language_code: hi-IN\r\n"
    )

    scrubbed = rw.scrub_secrets(headers_text, secret_values=[secret])

    assert secret not in scrubbed
    assert "my-secret-bearer-token-999" not in scrubbed
    assert "my-secret-gemini-key-777" not in scrubbed
    assert "X-API-Key-ID: [REDACTED]" in scrubbed
    assert "Authorization: Bearer [REDACTED]" in scrubbed
    assert "x-goog-api-key: [REDACTED]" in scrubbed
    assert "language_code: hi-IN" in scrubbed


def test_scrub_query_parameters() -> None:
    rw = _load_record_wire()
    raw_url = "https://api.vachana.ai/stt?key=topsecret123&language=hi-IN&api_key=secondsecret456"
    scrubbed = rw.scrub_secrets(raw_url)

    assert "topsecret123" not in scrubbed
    assert "secondsecret456" not in scrubbed
    assert "key=[REDACTED]" in scrubbed
    assert "api_key=[REDACTED]" in scrubbed
    assert "language=hi-IN" in scrubbed


def test_scrub_json_payload_tokens() -> None:
    rw = _load_record_wire()
    payload = {
        "api_key": "raw-key-12345",
        "client_secret": "client-secret-999",
        "token": "bearer-tok-888",
        "model": "timbre-v2.5",
        "text": "नमस्ते आप कैसे हैं",
    }
    raw_json = json.dumps(payload)
    scrubbed = rw.scrub_secrets(raw_json)

    assert "raw-key-12345" not in scrubbed
    assert "client-secret-999" not in scrubbed
    assert "bearer-tok-888" not in scrubbed

    parsed = json.loads(scrubbed)
    assert parsed["api_key"] == "[REDACTED]"
    assert parsed["client_secret"] == "[REDACTED]"
    assert parsed["token"] == "[REDACTED]"
    assert parsed["model"] == "timbre-v2.5"
    assert parsed["text"] == "नमस्ते आप कैसे हैं"


def test_scrub_wire_frame_dictionary() -> None:
    rw = _load_record_wire()
    secret = "gnani_key_sensitive_xyz"
    frame = {
        "t_ms": 120,
        "dir": "out",
        "kind": "text",
        "payload": json.dumps({"apiKey": secret, "action": "start"}),
        "assumed": False,
    }

    scrubbed_frame = rw.scrub_wire_frame(frame, secret_values=[secret])

    assert secret not in str(scrubbed_frame)
    payload_dict = json.loads(str(scrubbed_frame["payload"]))
    assert payload_dict["apiKey"] == "[REDACTED]"
    assert payload_dict["action"] == "start"
    assert scrubbed_frame["assumed"] is False


def test_scrub_preserves_binary_frame() -> None:
    rw = _load_record_wire()
    frame = {
        "t_ms": 250,
        "dir": "in",
        "kind": "binary",
        "b64": "AAECAwQFBgcICQoLDA0ODw==",
    }
    scrubbed = rw.scrub_wire_frame(frame)
    assert scrubbed["b64"] == "AAECAwQFBgcICQoLDA0ODw=="
    assert scrubbed["kind"] == "binary"


def test_scrub_idempotent() -> None:
    rw = _load_record_wire()
    text = "Authorization: Bearer [REDACTED]\r\nX-API-Key-ID: [REDACTED]"
    once = rw.scrub_secrets(text)
    twice = rw.scrub_secrets(once)
    assert once == text
    assert twice == text
