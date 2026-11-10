# Wire Transcripts Fixtures

This directory contains JSONL wire transcripts recorded or mocked for testing adapters via `FakeWebSocket`.

## Wire Format

Each line of a `.jsonl` transcript is a single JSON object representing one frame exchanged over the wire:

```json
{"t_ms": 120, "dir": "in", "kind": "text", "payload": "{\"type\":\"pong\"}"}
{"t_ms": 140, "dir": "out", "kind": "binary", "b64": "AAECAwQ="}
```

Fields:
- `t_ms` (integer): Millisecond timestamp relative to session start. Replayed deterministically in virtual time.
- `dir` (`"in"` | `"out"`): Direction of the frame.
  - `"in"`: Inbound frame from server to client (adapter receives via `ws.recv()`).
  - `"out"`: Outbound frame from client to server (adapter sends via `ws.send()`).
- `kind` (`"text"` | `"binary"`): Payload format.
- `payload` (string, optional): Text frame payload when `kind == "text"`.
- `b64` (string, optional): Base64-encoded bytes when `kind == "binary"`.
- `assumed` (boolean, optional): Flag indicating hand-written or assumed wire schemas before live verification.

## Assumed Fixtures Policy

Until live wire recording lands in **T3.01** (`tools/record_wire.py`, human-triggered):
- Hand-written test fixtures carry `"assumed": true`.
- When loaded, `FakeWebSocket` surfaces an `AssumedFixtureWarning`.
- **Per CLAUDE.md §3: Adapters cannot be marked done on assumed fixtures.**

## Capturing Real Transcripts

Real transcripts are captured once using `tools/record_wire.py` (live, human-triggered) and committed with all secrets and API keys scrubbed.
