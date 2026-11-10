"""Unit tests for SileroVad adapter with EnergyVad fallback.

Covers all done_when criteria and failure modes specified in T3.05:
- Passes VadContract suite with stub session and fallback modes.
- Windowing strategy achieves <= 1 frame added latency (measured 0-frame algorithmic delay).
- Fail-safe fallback to EnergyVad with telemetry event on missing model or missing onnxruntime.
- Dual-threshold / hangover hysteresis prevents flapping on single dips and releases cleanly.
- Sample-rate mismatch handled gracefully via resampling.
- Real-model test skipped cleanly when model or onnxruntime is absent (marked slow).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from overlap.adapters.vad_silero import SileroVad, SileroVadConfig
from overlap.core.audio import float32_to_pcm16
from overlap.core.types import AudioFrame
from overlap.ports.vad import VadResult

_contracts_dir = str(Path(__file__).resolve().parent.parent.parent / "contracts")
if _contracts_dir not in sys.path:
    sys.path.insert(0, _contracts_dir)

from suites import VadContract  # noqa: E402


class MockTelemetrySink:
    """Mock telemetry sink for capturing emitted events in tests."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int, dict[str, Any]]] = []

    def emit(self, name: str, ts_ns: int, **fields: Any) -> None:
        self.events.append((name, ts_ns, fields))


class StubOnnxSessionV4:
    """Stub ONNX session mimicking Silero VAD v4 (inputs: input, sr, h, c)."""

    def __init__(
        self,
        prob_sequence: list[float] | None = None,
        default_prob: float = 0.8,
    ) -> None:
        self.prob_sequence = list(prob_sequence) if prob_sequence is not None else None
        self.default_prob = default_prob
        self.call_count = 0
        self.inputs_received: list[dict[str, Any]] = []

    def get_inputs(self) -> list[Any]:
        class Meta:
            def __init__(self, name: str) -> None:
                self.name = name

        return [Meta("input"), Meta("sr"), Meta("h"), Meta("c")]

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[np.ndarray]:
        self.inputs_received.append(input_feed)
        if self.prob_sequence is not None:
            p = self.prob_sequence[min(self.call_count, len(self.prob_sequence) - 1)]
        else:
            p = self.default_prob
        self.call_count += 1
        # Silero v4 returns output prob, hn, cn
        out_prob = np.array([[p]], dtype=np.float32)
        hn = np.zeros((2, 1, 64), dtype=np.float32)
        cn = np.zeros((2, 1, 64), dtype=np.float32)
        return [out_prob, hn, cn]


class StubOnnxSessionV5:
    """Stub ONNX session mimicking Silero VAD v5 (inputs: input, state, sr)."""

    def __init__(
        self,
        prob_sequence: list[float] | None = None,
        default_prob: float = 0.8,
    ) -> None:
        self.prob_sequence = list(prob_sequence) if prob_sequence is not None else None
        self.default_prob = default_prob
        self.call_count = 0
        self.inputs_received: list[dict[str, Any]] = []

    def get_inputs(self) -> list[Any]:
        class Meta:
            def __init__(self, name: str) -> None:
                self.name = name

        return [Meta("input"), Meta("state"), Meta("sr")]

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[np.ndarray]:
        self.inputs_received.append(input_feed)
        if self.prob_sequence is not None:
            p = self.prob_sequence[min(self.call_count, len(self.prob_sequence) - 1)]
        else:
            p = self.default_prob
        self.call_count += 1
        out_prob = np.array([[p]], dtype=np.float32)
        new_state = np.ones((2, 1, 128), dtype=np.float32)
        return [out_prob, new_state]


def _make_frame(
    samples: np.ndarray | None = None,
    sample_rate: int = 16000,
    ts_ns: int = 0,
) -> AudioFrame:
    """Helper to construct an AudioFrame from float32 samples or silence."""
    if samples is None:
        num_samples = int(sample_rate * 0.02)  # 20 ms
        samples = np.zeros(num_samples, dtype=np.float32)
    pcm = float32_to_pcm16(samples)
    return AudioFrame(pcm=pcm, sample_rate=sample_rate, ts_ns=ts_ns)


# =============================================================================
# 1. VadContract Compliance
# =============================================================================


def test_silero_vad_passes_vad_contract_with_stub_session() -> None:
    session = StubOnnxSessionV4(default_prob=0.8)
    vad = SileroVad(session=session)
    VadContract.verify(vad)


def test_silero_vad_passes_vad_contract_in_fallback_mode() -> None:
    # No session or model provided -> enters fallback to EnergyVad
    cfg = SileroVadConfig(threshold=0.6, neg_threshold=0.4, hangover_frames=3)
    vad = SileroVad(config=cfg, session=None, model_path=None)
    assert vad.is_fallback is True
    VadContract.verify(vad)


# =============================================================================
# 2. Startup Fail-Safe & Telemetry Fallback
# =============================================================================


def test_missing_model_file_falls_back_without_crashing() -> None:
    telemetry = MockTelemetrySink()
    vad = SileroVad(
        model_path="/path/to/nonexistent/silero_vad.onnx",
        telemetry=telemetry,
    )
    assert vad.is_fallback is True

    # Telemetry event must have been emitted
    fallback_events = [e for e in telemetry.events if e[0] == "vad.fallback_energy"]
    assert len(fallback_events) == 1
    _, ts_ns, fields = fallback_events[0]
    assert "reason" in fields
    assert "model_path" in fields

    # Processing frames must succeed via EnergyVad fallback
    frame = _make_frame()
    result = vad.process(frame)
    assert isinstance(result, VadResult)
    assert isinstance(result.is_speech, bool)


def test_fallback_handles_speech_and_silence() -> None:
    vad = SileroVad(session=None, model_path=None)
    assert vad.is_fallback is True

    # Silent frame
    silence = _make_frame(np.zeros(320, dtype=np.float32))
    res_silence = vad.process(silence)
    assert res_silence.is_speech is False

    # Loud speech frame
    t = np.linspace(0, 0.02, 320, endpoint=False)
    loud_tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    loud_frame = _make_frame(loud_tone)
    res_speech = vad.process(loud_frame)
    assert res_speech.is_speech is True


# =============================================================================
# 3. Windowing Strategy & Added Latency <= 1 Frame
# =============================================================================


def test_windowing_strategy_zero_added_latency() -> None:
    """Verify that model receives 512-sample window without waiting for future frames.

    Added latency must be <= 1 frame (INV-L1/L2: measured 0 frames algorithmic delay).
    When speech arrives at frame k, frame k must immediately register speech.
    """
    # 2 silent frames followed by 1 speech frame
    stub = StubOnnxSessionV4(prob_sequence=[0.1, 0.1, 0.95, 0.95])
    vad = SileroVad(session=stub, threshold=0.5)

    frame0 = _make_frame(ts_ns=0)
    frame1 = _make_frame(ts_ns=20_000_000)
    frame2 = _make_frame(ts_ns=40_000_000)

    res0 = vad.process(frame0)
    assert res0.is_speech is False

    res1 = vad.process(frame1)
    assert res1.is_speech is False

    # Frame 2 is speech -> must be detected immediately on frame 2 without waiting for frame 3
    res2 = vad.process(frame2)
    assert res2.is_speech is True
    assert res2.confidence == pytest.approx(0.95, abs=1e-3)

    # Verify that the session received exactly 512 samples on each call
    assert len(stub.inputs_received) == 3
    for feed in stub.inputs_received:
        inp = feed["input"]
        assert inp.shape == (1, 512), f"Expected input shape (1, 512), got {inp.shape}"


def test_window_content_uses_past_context() -> None:
    """Verify the 512-sample window consists of 192 past samples + 320 current samples."""
    stub = StubOnnxSessionV4(default_prob=0.5)
    vad = SileroVad(session=stub)

    # First frame: all 1.0s
    frame1_samples = np.ones(320, dtype=np.float32)
    frame1 = _make_frame(frame1_samples, ts_ns=0)
    vad.process(frame1)

    # First call: past 192 samples padded with zeros, followed by 320 ones
    feed1 = stub.inputs_received[0]["input"][0]
    np.testing.assert_array_equal(feed1[:192], np.zeros(192, dtype=np.float32))
    assert np.allclose(feed1[192:], 1.0, atol=1e-3)

    # Second frame: all 2.0s
    frame2_samples = np.full(320, 0.5, dtype=np.float32)
    frame2 = _make_frame(frame2_samples, ts_ns=20_000_000)
    vad.process(frame2)

    # Second call: past 192 samples come from trailing samples of frame 1 (ones),
    # followed by 320 samples of frame 2 (0.5s)
    feed2 = stub.inputs_received[1]["input"][0]
    assert np.allclose(feed2[:192], 1.0, atol=1e-3)
    assert np.allclose(feed2[192:], 0.5, atol=1e-3)


# =============================================================================
# 4. Hysteresis (No Flapping on Single Dips) and Clean Release
# =============================================================================


def test_hysteresis_prevents_flapping_on_transient_dip() -> None:
    # Sequence: High -> Low dip -> High
    probs = [0.8, 0.2, 0.8]
    stub = StubOnnxSessionV4(prob_sequence=probs)
    vad = SileroVad(
        session=stub,
        threshold=0.5,
        neg_threshold=0.35,
        hangover_frames=3,
    )

    r0 = vad.process(_make_frame())
    assert r0.is_speech is True

    # Transient dip should remain speech due to hangover
    r1 = vad.process(_make_frame())
    assert r1.is_speech is True

    r2 = vad.process(_make_frame())
    assert r2.is_speech is True


def test_hysteresis_cleanly_releases_after_hangover() -> None:
    # 1 speech frame, then 4 silent frames (with hangover_frames = 2)
    probs = [0.8, 0.1, 0.1, 0.1, 0.1]
    stub = StubOnnxSessionV4(prob_sequence=probs)
    vad = SileroVad(
        session=stub,
        threshold=0.5,
        neg_threshold=0.35,
        hangover_frames=2,
    )

    r0 = vad.process(_make_frame())
    assert r0.is_speech is True

    # Hangover frame 1
    r1 = vad.process(_make_frame())
    assert r1.is_speech is True

    # Hangover frame 2
    r2 = vad.process(_make_frame())
    assert r2.is_speech is True

    # Hangover expired -> clean release
    r3 = vad.process(_make_frame())
    assert r3.is_speech is False

    r4 = vad.process(_make_frame())
    assert r4.is_speech is False


def test_sub_threshold_prob_does_not_trigger_onset() -> None:
    # Prob is above neg_threshold (0.35) but below onset threshold (0.5) from silence
    probs = [0.4, 0.45, 0.4]
    stub = StubOnnxSessionV4(prob_sequence=probs)
    vad = SileroVad(
        session=stub,
        threshold=0.5,
        neg_threshold=0.35,
    )

    for _ in range(3):
        res = vad.process(_make_frame())
        assert res.is_speech is False


# =============================================================================
# 5. Sample Rate Mismatch Handling & Resampling
# =============================================================================


def test_handles_8khz_sample_rate() -> None:
    stub = StubOnnxSessionV4(default_prob=0.7)
    vad = SileroVad(session=stub)

    # 8 kHz frame: 20 ms = 160 samples
    frame_8k = _make_frame(sample_rate=8000)
    assert len(frame_8k.pcm) == 320  # 160 samples * 2 bytes

    res = vad.process(frame_8k)
    assert isinstance(res, VadResult)
    # The session should still have received a 512-sample 16 kHz window
    feed = stub.inputs_received[0]["input"]
    assert feed.shape == (1, 512)


def test_handles_48khz_sample_rate() -> None:
    stub = StubOnnxSessionV4(default_prob=0.7)
    vad = SileroVad(session=stub)

    # 48 kHz frame: 20 ms = 960 samples
    frame_48k = _make_frame(sample_rate=48000)
    assert len(frame_48k.pcm) == 1920  # 960 samples * 2 bytes

    res = vad.process(frame_48k)
    assert isinstance(res, VadResult)
    feed = stub.inputs_received[0]["input"]
    assert feed.shape == (1, 512)


def test_handles_empty_pcm_frame() -> None:
    stub = StubOnnxSessionV4(default_prob=0.7)
    vad = SileroVad(session=stub)

    empty_frame = AudioFrame(pcm=b"", sample_rate=16000, ts_ns=0)
    res = vad.process(empty_frame)
    assert res.is_speech is False
    assert res.confidence == 0.0


# =============================================================================
# 6. Silero V5 Model Compatibility & Reset
# =============================================================================


def test_silero_v5_model_session() -> None:
    stub = StubOnnxSessionV5(default_prob=0.85)
    vad = SileroVad(session=stub)

    res = vad.process(_make_frame())
    assert res.is_speech is True
    assert res.confidence == pytest.approx(0.85, abs=1e-3)
    assert "state" in stub.inputs_received[0]
    assert stub.inputs_received[0]["state"].shape == (2, 1, 128)


def test_reset_clears_history_and_state() -> None:
    stub = StubOnnxSessionV4(prob_sequence=[0.9, 0.9])
    vad = SileroVad(session=stub)

    vad.process(_make_frame())
    assert vad.process(_make_frame()).is_speech is True

    vad.reset()
    # History must be reset to zeros
    stub.prob_sequence = [0.1]
    stub.call_count = 0
    res = vad.process(_make_frame())
    assert res.is_speech is False
    first_input = stub.inputs_received[-1]["input"][0]
    np.testing.assert_array_equal(first_input[:192], np.zeros(192, dtype=np.float32))


# =============================================================================
# 7. Real-Model Slow Test (Skipped When Absent)
# =============================================================================


@pytest.mark.slow
def test_real_silero_model_if_present() -> None:
    """Run real ONNX inference if model and onnxruntime are present."""
    model_path = os.environ.get("SILERO_MODEL_PATH", "models/silero_vad.onnx")
    try:
        import onnxruntime  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pytest.skip("onnxruntime not installed")

    if not Path(model_path).exists():
        pytest.skip(f"Silero model not found at {model_path}")

    vad = SileroVad(model_path=model_path)
    assert vad.is_fallback is False

    frame = _make_frame()
    result = vad.process(frame)
    assert isinstance(result, VadResult)
    assert 0.0 <= result.confidence <= 1.0
