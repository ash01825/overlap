# WIRE.md — OVERLAP Wire Protocol Specification

This document is the **single source of truth** for all wire interactions between OVERLAP adapters and upstream providers: **Gnani Prisma** (STT), **Gnani Timbre** (TTS), and **Evon / Gemini** (LLM).

All endpoints, headers, message schemas, and timing properties documented here are verified against live production traffic captured in task **T3.01** using `tools/record_wire.py`.

---

## 1. Authentication & Base Infrastructure

* **Base URL:** `https://api.vachana.ai`
* **WebSocket Base:** `wss://api.vachana.ai`
* **Authentication Header:**
  ```http
  X-API-Key-ID: <GNANI_API_KEY>
  ```
* **Cloudflare User-Agent Requirement:**
  Requests from the default Python `urllib` user agent are blocked by Cloudflare with HTTP 403 (`error code: 1010: The owner of this website has banned your access based on your browser's signature`). All adapters and HTTP/WebSocket clients **must** set an explicit client `User-Agent` (e.g. `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)` or `overlap-client/0.1.0`).

---

## 2. Gnani Prisma STT (Speech-to-Text)

### 2.1 Endpoint & Handshake

* **WebSocket URL:** `wss://api.vachana.ai/stt/v3/stream`
* **Connection Handshake:** Configuration is passed via HTTP upgrade headers during the initial WebSocket handshake:

| Header | Value | Purpose |
|---|---|---|
| `X-API-Key-ID` | `<GNANI_API_KEY>` | Authentication |
| `model` | `prisma-v3.0` | STT model (`prisma-v2.5` or `prisma-v3.0`) |
| `language_code` | `hi-IN` or `indic-auto` | Target language code or multilingual auto-detection |
| `x-partials` | `true` | **Enables live interim partial transcripts** |
| `format` | `verbatim` or `transcribe` | Output normalization style (`verbatim` raw, `transcribe` with ITN) |
| `User-Agent` | `Mozilla/5.0 ...` | Bypass Cloudflare bot protection |

### 2.2 Audio Ingestion Protocol

* **Frame Format:** Raw mono PCM 16-bit linear (`int16`), little-endian, sampled at **16,000 Hz**.
* **Transport:** Streamed as binary WebSocket frames (opcode 2, masked from client).
* **Frame Cadence:** 20 ms frames (320 samples = 640 bytes) or 32 ms frames (512 samples = 1024 bytes). OVERLAP's audio engine standardizes on 20 ms frames (640 bytes).
* **Buffer Policy:** Non-blocking bounded send queue. If socket backpressure occurs, the client drops oldest uncompressed frames to prevent audio pipeline latency.

### 2.3 Server Event Sequence

Once connected, the server emits text frames (opcode 1, JSON) matching the following lifecycle:

#### 1. `connected`
Sent immediately upon successful handshake:
```json
{
  "type": "connected",
  "message": "STT service ready — VAD service connected",
  "request_id": "01a1c514-9b79-7bc8-a3e4-e54e5ed94ef9",
  "config": {
    "sample_rate": 16000,
    "chunk_size": 512,
    "partials_enabled": true,
    "partial_interval_ms": 450,
    "partial_max_window_s": 15.0,
    "language_code": "hi-IN",
    "format": "verbatim"
  },
  "timestamp": "2026-11-10T10:50:24.579662+00:00"
}
```

#### 2. `speech_start`
Emitted by server VAD when speech onset is detected:
```json
{
  "type": "speech_start",
  "utterance_id": "u4eb6fd94",
  "t_ms": 0,
  "timestamp": "2026-11-10T10:50:24.941420+00:00"
}
```

#### 3. `partial_transcript`
Emitted incrementally at ~450 ms intervals (`partial_interval_ms`) while speech continues:
```json
{
  "type": "partial_transcript",
  "utterance_id": "u4eb6fd94",
  "seq": 1,
  "is_final": false,
  "text": "नमस्ते",
  "stable_text": "",
  "volatile_text": "नमस्ते",
  "words": [{"text": "नमस्ते", "stable": false}],
  "stable_word_count": 0,
  "new_stable_word_count": 0,
  "audio_ms": 780,
  "timestamp": "2026-11-10T10:50:25.589210+00:00"
}
```

#### 4. `speech_end`
Emitted by server VAD when speech offset / pause is detected:
```json
{
  "type": "speech_end",
  "utterance_id": "u4eb6fd94",
  "t_ms": 2120,
  "audio_ms": 2112,
  "timestamp": "2026-11-10T10:50:27.197871+00:00"
}
```

#### 5. `final_transcript`
Finalized, punctuated transcript for the utterance segment:
```json
{
  "type": "final_transcript",
  "text": "नमस्ते, आप कैसे हैं?",
  "audio_duration_ms": 2112.0,
  "segment_id": "01a1c514-9b79-7bc8-a3e4-e54e5ed94ef9-seg-1",
  "segment_index": 1,
  "utterance_id": "u4eb6fd94",
  "is_final": true,
  "latency": 72,
  "prefetched": false,
  "timestamp": "2026-11-10T10:50:27.270408+00:00",
  "detected_language_code": "hi-IN"
}
```

### 2.4 Code-Switching & Auto-Detect
When `model: "prisma-v3.0"` and `language_code: "indic-auto"` are passed, Prisma transparently detects mixed Indic-English utterances (e.g. `"हैन, मीटिंग कैंसिल हो गई है।"`) and returns the dominant script and `detected_language_code`.

---

## 3. Gnani Timbre TTS (Text-to-Speech)

### 3.1 Endpoints & Supported Models

* **Streaming WebSocket:** `wss://api.vachana.ai/api/v1/tts`
* **REST Batch Synthesis:** `POST https://api.vachana.ai/api/v1/tts/inference` (used by `ClipCache`)
* **SSE Streaming:** `POST https://api.vachana.ai/api/v1/tts/sse`
* **Models:**
  * `timbre-v2.5`: Production stable multilingual Indic model.
  * `timbre-v3.0`: Expressive Indic model supporting inline emotion tags (`<whispers>`, `<laugh>`, `<pauses>`).

### 3.2 Request Schema

Sent over the WebSocket as a single text frame (opcode 1, JSON) after connection:
```json
{
  "text": "नमस्ते, आप कैसे हैं?",
  "voice": "Nalini",
  "model": "timbre-v2.5",
  "language": "hi-IN",
  "speed": 1.0,
  "audio_config": {
    "sample_rate": 16000,
    "num_channels": 1,
    "sample_width": 2,
    "encoding": "linear_pcm",
    "container": "wav"
  }
}
```

### 3.3 Server Event Sequence

#### 1. `start`
```json
{
  "type": "start",
  "message": "Streaming started",
  "request_id": "01a1c514-980c-7714-8f5f-16aa6b5dcac8"
}
```

#### 2. `audio`
Chunks are delivered as base64-encoded audio. The container header (e.g. 44-byte RIFF WAV header) is present in chunk 1:
```json
{
  "type": "audio",
  "data": {
    "chunk_index": 1,
    "audio": "UklGRv////9XQVZFZm10...",
    "is_final": false
  }
}
```
The final chunk sets `is_final: true`:
```json
{
  "type": "audio",
  "data": {
    "chunk_index": 14,
    "audio": "...",
    "is_final": true
  }
}
```

### 3.4 Word Timings Determination

> **CRITICAL VERIFIED FINDING:**
> Empirical traffic analysis of `wss://api.vachana.ai/api/v1/tts` confirms that **Gnani Timbre DOES NOT provide word timing markers** in its audio chunk payloads.
> 
> Therefore, per `ARCHITECTURE.md §15 #3`, OVERLAP implements the **calibrated-rate fallback path**:
> * `AudioChunk.words` is left empty (`words = ()`).
> * `PlayoutLedger` tracks playtime by exact sample count (`samples_played / sample_rate`), truncating the heard transcript proportionally against the spoken text duration.

### 3.5 Interruption & Cancel Protocol (Barge-In)

When the FSM transitions to `DUCKED` or `YIELDED`:
1. Client sends an RFC 6455 close frame (opcode 8 with code 1000) or `{"action": "cancel", "code": 1000}`.
2. Server drains and discards remaining audio synthesis and returns a close acknowledgment frame.
3. The client-side `TtsStream.cancel()` method is idempotent; it ceases yielding chunks to the mixer within 1 event loop turn.

---

## 4. Evon / Gemini LLM Integration

### 4.1 Architecture Decision

Gnani's hosted `api.vachana.ai` gateway restricts API key scopes strictly to `/stt` and `/api/v1/tts` (returning `403 SCOPE_NOT_ALLOWED: Service 'None' not in allowed scopes` for text completions). The foundational Evon model (`gnani/gnani-evon-v3.3`) is published as open weights on Hugging Face for self-hosting.

For the live LLM completion stream, OVERLAP connects to **Gemini 2.5 Flash** using its OpenAI-compatible Server-Sent Events (SSE) streaming interface:

* **Endpoint:** `POST https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`
* **Authentication Header:** `Authorization: Bearer <GEMINI_API_KEY>`

### 4.2 Request Schema

```json
{
  "model": "gemini-2.5-flash",
  "messages": [
    {"role": "system", "content": "You are a concise, natural voice assistant."},
    {"role": "user", "content": "नमस्ते, क्या हाल है?"}
  ],
  "stream": true
}
```

### 4.3 Streaming SSE Event Format

Streamed line-by-line using standard SSE:
```http
data: {"choices":[{"delta":{"content":"नमस्ते! ","role":"assistant"},"finish_reason":null,"index":0}],"model":"gemini-2.5-flash"}
data: {"choices":[{"delta":{"content":"सब कुछ ठीक है।","role":"assistant"},"finish_reason":"stop","index":0}],"model":"gemini-2.5-flash"}
data: [DONE]
```

### 4.4 Stream Abort Semantics

* When an interruption occurs while in `THINKING` state, `LlmStream.abort()` closes the active HTTP/SSE connection immediately.
* Aborting halts further token generation and stops downstream TTS clause processing.

---

## 5. Port Protocol Compatibility Analysis

Reopened task **T0.04** was reviewed against real wire behavior:
1. `SttStream`: Protocol matches Prisma events (`partial` and `final` transcripts, language detection, non-blocking `send_audio`).
2. `TtsStream`: Protocol accommodates Timbre (`AudioChunk` carries `words=()` when word timings are unavailable; `cancel()` terminates stream).
3. `LlmStream`: Protocol matches SSE streaming (`deltas()` yields text tokens, `abort()` drops connection).

**Conclusion:** No breaking changes to `overlap/ports/*.py` are required. The ports defined in T0.04 correctly handle the real wire contracts.

---

## 6. Error Handling & Rate Limits

| Scenario | Server Response | Adapter Action |
|---|---|---|
| Invalid / Expired Key | HTTP 403 Forbidden | Raise `AuthenticationError`; fail fast |
| Cloudflare Block | HTTP 403 (Error 1010) | Supply valid `User-Agent` header |
| Rate Limit Exceeded | HTTP 429 Too Many Requests | Exponential backoff (initial 500 ms, max 3 retries) |
| Prisma WS Disconnect | Socket EOF / 1006 abnormal | Reconnect with ≤ 2 s audio replay buffer; fallback to acoustic VAD |
| Timbre Cancel Mid-Stream | Inbound chunks after close | Dropped by epoch check in audio mixer |
| LLM Stream Timeout | No token within 2.0 s | Abort stream; trigger canned apology clip |
