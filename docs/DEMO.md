# OVERLAP — Live Demo & Offline Evaluation Runbook

This guide covers running the **OVERLAP** full-duplex voice agent (duck-then-decide overlap handling, heard-text truncation, and agent backchannels) as well as the deterministic offline A/B naturalness harness when live credentials or network connectivity are unavailable.

---

## 1. Headphone, Microphone & Acoustic Echo Cancellation (AEC) Setup

Because OVERLAP keeps the microphone active while the agent is speaking (ducking playout to `0.3` gain during candidate overlaps rather than muting the room), proper acoustic isolation ensures zero false self-ducks (`echo_false_duck == 0`):

1. **Recommended Hardware (Wired Closed-Back Headphones):**
   - Use wired closed-back headphones or a directional headset mic so speaker bleed into the microphone is physically eliminated.
   - Verify macOS / Linux input gain is set to ~70% to avoid clipping on sudden barge-ins (`"rukiye"`, `"wait"`).
2. **Browser / WebRTC Acoustic Echo Cancellation (LiveKit Transport):**
   - When connecting via WebRTC / LiveKit, ensure `echoCancellation: true`, `noiseSuppression: true`, and `autoGainControl: true` are enabled on the client audio track.
   - OVERLAP's internal `EchoGate` also tags outbound TTS (`Tag.SELF_TTS`) and backchannel clips (`Tag.SELF_BACKCHANNEL`) with a configurable `echo_hold_ms` holdover window (`Tuning.echo_hold_ms = 60 ms`).
3. **Downstream Playout Calibration (`assumed_downstream_ms`):**
   - Default `Tuning.assumed_downstream_ms` is `80 ms`. If running over a high-buffer Bluetooth output device (not recommended for live demos), see `docs/CALIBRATION.md` to adjust `OVERLAP_ASSUMED_DOWNSTREAM_MS`.

---

## 2. Environment Variables & Configuration (No Embedded Secrets)

All credentials and runtime parameters are read exclusively from environment variables (never hardcoded or committed):

```bash
# Optional for offline/hermetic demo; required only for live Gnani Vachana WS sessions
export GNANI_API_KEY="your-gnani-api-key"
export OVERLAP_VOICE_ID="gnani-timbre-hi-female-1"
export OVERLAP_LANG="hi-IN"
export OVERLAP_DUCK_GAIN="0.3"
export OVERLAP_CLASSIFY_CAP_MS="400"
```

---

## 3. Running the Demo (`make demo` or Docker)

### Local Conda Environment (`gnani`)
```bash
conda activate gnani
make demo
```
`make demo` runs in deterministic offline A/B dry-run mode by default (requiring zero external secrets or network access) and prints the blinded A/B assignment table plus the stimulus-identity verification across all 8 scripted caller scenarios.

### Docker Container
```bash
docker build -t overlap-agent:latest .
docker run --rm \
  -e OVERLAP_LANG="hi-IN" \
  -e OVERLAP_VOICE_ID="gnani-timbre-hi-female-1" \
  overlap-agent:latest
```

---

## 4. Fallback Plan (Zero-Network / Offline Resilience)

If live Wi-Fi, LiveKit room connectivity, or `api.vachana.ai` credentials experience issues during a presentation:

1. **Instant Hermetic A/B Replay (`make demo`):**
   - Run `python -m overlap.bench.ab --output-dir reports/ab --seed 42` to deterministically synthesize all 8 caller scenarios (`bench/ab_scenarios/*.yaml`) across both **Stock Pipecat** (stop-on-any-VAD) and **OVERLAP** (duck-then-decide).
   - Play the generated WAV files in `reports/ab/clips/pair_01_arm_A.wav` vs `pair_01_arm_B.wav` directly for judges.
2. **Pre-computed Telemetry & Statistical Summary (`reports/ab/summary.md`):**
   - Open `reports/ab/summary.md` to show the Wilson 95% confidence interval, paired sign test p-value, false-interruption rate reduction (`100% -> 0%` on Hindi/Tamil/English backchannels), and silence-gap improvements.
3. **Interactive Blind Rating CLI:**
   - Run `python -m overlap.bench.rating_cli --manifest reports/ab/blind_manifest.json` to let a judge rate a pair live in the terminal before unblinding.
