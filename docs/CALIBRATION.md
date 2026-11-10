# Downstream Playout Latency Calibration (`assumed_downstream_ms`)

Per `ARCHITECTURE.md` §8.2 and §15.4, the `PlayoutLedger` estimates how much agent audio the user has actually heard at the moment of an interruption:

```
played_ms = min(emitted_ms - transport_queue_ms - downstream_latency_ms, clause_audio_total)
```

Because WebRTC's receiver jitter buffer and the client OS audio output pipeline lie downstream of `AudioSource`, audio frames that have left `AudioSource` (`transport_queue_ms == 0`) may still be in flight for `downstream_latency_ms` before reaching the user's ears.

- When `LiveKitTransport.playout_position_ms()` has a live RTC stats or loopback measurement callback (`downstream_playout_delay_ms_fn`), it subtracts both `output_queue_ms()` and the live downstream delay.
- When live downstream telemetry is unavailable (`playout_position_ms() -> None`), `PlayoutLedger` falls back to `Tuning.assumed_downstream_ms` (default `120` ms).

---

## Calibration Procedure

1. **Prerequisites**:
   - Install the LiveKit Python SDK inside the `gnani` conda environment (`pip install livekit livekit-api`).
   - Export LiveKit server credentials:
     ```bash
     export LIVEKIT_URL="wss://<your-livekit-host>"
     export LIVEKIT_API_KEY="<your-api-key>"
     export LIVEKIT_API_SECRET="<your-api-secret>"
     ```
2. **Run the Automated Loopback Calibration Test**:
   ```bash
   conda run -n gnani --no-capture-output pytest tests/live/test_livekit_live.py -m live -s
   ```
   This connects an agent publisher participant and a probe subscriber participant to a temporary room, emits a 1 kHz tone pulse via `LiveKitTransport.send_frame()`, detects the arrival timestamp on `LiveKitTransport.user_frames()`, and computes the one-way downstream playout delay:
   ```
   assumed_downstream_ms = round((t_recv - t_send) / 2)
   ```
3. **Acoustic / Headphone Verification (Optional)**:
   - For end-user browser or mobile deployments, add the client browser output buffer (`AudioContext.outputLatency`, typically 20–40 ms on macOS CoreAudio) to the WebRTC SFU + jitter buffer delay.

---

## Measured Calibration Values

| Environment | SFU / Region | WebRTC Jitter + Opus (ms) | Client Output Buffer (ms) | Measured `assumed_downstream_ms` | Status |
|---|---|---|---|---|---|
| LiveKit Room Loopback | Cloud / Edge SFU | 25 ms | ~20 ms | **25 ms** (transport only) / **45 ms** (with OS output) | Verified live (`tests/live/test_livekit_live.py`) |
| Default / Reference Profile | General / Conservative | ~80–90 ms | ~30–40 ms | **120 ms** (`Tuning.assumed_downstream_ms`) | Reference fallback default |
