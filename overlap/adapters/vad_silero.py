"""Silero VAD adapter (ONNX runtime, 20 ms hop) with EnergyVad fallback.

=============================================================================
Windowing Strategy and Added Latency (INV-L1, INV-L2)
=============================================================================
Silero VAD models (v4 and v5) operate on a fixed inference window of 512 samples
(32 ms at 16 kHz), whereas OVERLAP streams audio in 20 ms frames (320 samples
at 16 kHz) with an invariant hop of 20 ms (INV-L1).

Naive buffering would wait for a second 20 ms frame before evaluating, adding
>= 1 frame (20 ms) of algorithmic latency and violating the ducking latency
guarantee (INV-L2: speech onset to gain-ducked frame <= 40 ms).

Windowing Strategy:
- Look-behind context: To construct the required 512-sample inference window
  for the current 20 ms frame (320 samples), the adapter prepends exactly
  192 samples (12 ms) of *past* audio history.
- Pre-roll padding: When the stream starts and fewer than 192 samples of history
  are available, the past context is padded with zeros (silence).
- Zero look-ahead delay: Because the extra 192 samples are drawn entirely from
  the past, inference on the current 20 ms frame occurs immediately upon receipt.
- Measured added algorithmic latency: 0 frames (0 ms), strictly fulfilling the
  requirement of added latency <= 1 frame.

=============================================================================
Fallback & Fail-Safe Operation
=============================================================================
If ONNX Runtime is not installed, or the model file is missing or invalid, the
adapter automatically falls back to EnergyVad without raising an exception or
crashing startup. A telemetry event ("vad.fallback_energy") is emitted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from overlap.core.audio import EnergyVad, pcm16_to_float32, rms_dbfs
from overlap.core.types import AudioFrame
from overlap.ports.telemetry import NullSink, TelemetrySink
from overlap.ports.vad import Vad, VadResult

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE: int = 16000
DEFAULT_WINDOW_SIZE: int = 512
FRAME_SAMPLES_20MS: int = 320  # 20 ms at 16 kHz
CONTEXT_SAMPLES: int = DEFAULT_WINDOW_SIZE - FRAME_SAMPLES_20MS  # 192 samples


@dataclass(frozen=True, slots=True)
class SileroVadConfig:
    """Configuration parameters for SileroVad adapter."""

    model_path: str | Path | None = None
    sample_rate: int = TARGET_SAMPLE_RATE
    window_size_samples: int = DEFAULT_WINDOW_SIZE
    threshold: float = 0.5
    neg_threshold: float = 0.35
    hangover_frames: int = 5


class SileroVad(Vad):
    """Voice Activity Detector using local Silero ONNX with EnergyVad fallback.

    Fulfills the Vad port protocol. Fast-path process() is strictly synchronous,
    evaluating incoming 20 ms frames with 0 frames added look-ahead latency.
    """

    def __init__(
        self,
        config: SileroVadConfig | None = None,
        session: Any | None = None,
        telemetry: TelemetrySink | None = None,
        model_path: str | Path | None = None,
        threshold: float = 0.5,
        neg_threshold: float = 0.35,
        hangover_frames: int = 5,
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> None:
        self._cfg = config or SileroVadConfig(
            model_path=model_path,
            sample_rate=sample_rate,
            threshold=threshold,
            neg_threshold=neg_threshold,
            hangover_frames=hangover_frames,
        )
        self._telemetry: TelemetrySink = telemetry or NullSink()
        self._threshold: float = self._cfg.threshold
        self._neg_threshold: float = self._cfg.neg_threshold
        self._hangover_frames: int = self._cfg.hangover_frames
        self._target_sr: int = self._cfg.sample_rate
        self._window_size: int = self._cfg.window_size_samples
        self._context_size: int = max(0, self._window_size - FRAME_SAMPLES_20MS)

        # Hysteresis state
        self._is_speech: bool = False
        self._hangover_remaining: int = 0

        # Audio history buffer for look-behind context
        self._history: np.ndarray = np.zeros(self._context_size, dtype=np.float32)

        # ONNX / Model state
        self._session: Any | None = session
        self._is_fallback: bool = False
        self._fallback_vad: EnergyVad | None = None

        # Model input format flags
        self._is_v5: bool = False
        self._has_sr_input: bool = True
        self._has_state_input: bool = False
        self._state: np.ndarray | None = None
        self._h: np.ndarray | None = None
        self._c: np.ndarray | None = None

        if self._session is not None:
            self._init_session_metadata()
        else:
            self._init_or_fallback()

    @property
    def is_fallback(self) -> bool:
        """True if the adapter is using EnergyVad fallback."""
        return self._is_fallback

    def _init_session_metadata(self) -> None:
        """Inspect input shapes and initialize state tensors for the session."""
        if self._session is None:
            return
        try:
            inputs = self._session.get_inputs()
            input_names = {inp.name for inp in inputs}
            self._has_sr_input = "sr" in input_names

            if "state" in input_names:
                # Silero VAD v5 format
                self._is_v5 = True
                self._has_state_input = True
                self._state = np.zeros((2, 1, 128), dtype=np.float32)
            elif "h" in input_names and "c" in input_names:
                # Silero VAD v4 format
                self._is_v5 = False
                self._has_state_input = True
                self._h = np.zeros((2, 1, 64), dtype=np.float32)
                self._c = np.zeros((2, 1, 64), dtype=np.float32)
            else:
                self._has_state_input = False
        except Exception:
            # Stub session or generic mock without get_inputs
            self._has_sr_input = False
            self._has_state_input = False

    def _init_or_fallback(self) -> None:
        """Attempt to load ONNX model; fail-safe fallback to EnergyVad on error."""
        model_path = self._cfg.model_path
        if model_path is None:
            self._trigger_fallback("no_model_path_provided")
            return

        resolved_path = Path(model_path)
        if not resolved_path.is_file():
            self._trigger_fallback(f"model_file_not_found: {model_path}")
            return

        try:
            import onnxruntime as ort  # type: ignore[import-not-found]

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                str(resolved_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._init_session_metadata()
            self._is_fallback = False
        except ImportError:
            self._trigger_fallback("onnxruntime_not_installed")
        except Exception as exc:
            self._trigger_fallback(f"onnx_load_error: {exc}")

    def _trigger_fallback(self, reason: str) -> None:
        """Activate EnergyVad fallback and emit telemetry event."""
        self._is_fallback = True
        self._fallback_vad = EnergyVad(
            speech_threshold_db=12.0,
            hangover_frames=self._hangover_frames,
        )
        self._telemetry.emit(
            "vad.fallback_energy",
            ts_ns=0,
            reason=reason,
            model_path=str(self._cfg.model_path) if self._cfg.model_path else None,
        )
        logger.info("SileroVad falling back to EnergyVad: %s", reason)

    def _resample_if_needed(self, samples: np.ndarray, src_sr: int) -> np.ndarray:
        """Resample audio samples to 16 kHz using linear interpolation if sample rate differs."""
        if src_sr == self._target_sr or len(samples) == 0:
            return samples
        target_len = int(round(len(samples) * self._target_sr / src_sr))
        if target_len <= 0:
            return np.empty(0, dtype=np.float32)
        orig_indices = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        target_indices = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        return np.asarray(np.interp(target_indices, orig_indices, samples), dtype=np.float32)

    def process(self, frame: AudioFrame) -> VadResult:
        """Process an audio frame and return speech detection result synchronously (INV-L1)."""
        if self._is_fallback and self._fallback_vad is not None:
            return self._fallback_vad.process(frame)  # type: ignore[no-any-return]

        if not frame.pcm:
            return VadResult(is_speech=False, confidence=0.0, energy_db=-90.0)

        # Convert PCM bytes to float32 samples
        samples = pcm16_to_float32(frame.pcm)
        energy_db = rms_dbfs(samples)

        # Handle sample rate mismatch via resampling to 16 kHz
        if frame.sample_rate != self._target_sr:
            samples = self._resample_if_needed(samples, frame.sample_rate)

        # Construct 512-sample inference window using past context + current samples
        n_samples = len(samples)
        if n_samples >= self._window_size:
            window = samples[-self._window_size :]
        else:
            needed_context = self._window_size - n_samples
            if len(self._history) < needed_context:
                pad = np.zeros(needed_context - len(self._history), dtype=np.float32)
                context = np.concatenate([pad, self._history])
            else:
                context = self._history[-needed_context:]
            window = np.concatenate([context, samples])

        # Update look-behind history with the trailing samples of the current frame
        if n_samples >= self._context_size:
            self._history = samples[-self._context_size :]
        elif self._context_size > 0:
            shift = self._context_size - n_samples
            self._history = np.concatenate([self._history[-shift:], samples])

        # Prepare model input tensor of shape (1, 512)
        input_tensor = np.expand_dims(window, axis=0).astype(np.float32)

        # Execute ONNX inference
        try:
            prob = self._run_inference(input_tensor)
        except Exception as exc:
            logger.warning("Silero inference error: %s; falling back to EnergyVad", exc)
            self._trigger_fallback(f"inference_runtime_error: {exc}")
            return self._fallback_vad.process(frame)  # type: ignore[union-attr, no-any-return]

        # Apply dual-threshold hysteresis and hangover logic
        if prob >= self._threshold:
            # 1-frame onset (INV-L1)
            self._is_speech = True
            self._hangover_remaining = self._hangover_frames
        elif prob >= self._neg_threshold and self._is_speech:
            # Maintain active speech while above release threshold
            self._is_speech = True
            self._hangover_remaining = self._hangover_frames
        else:
            # Probability below neg_threshold
            if self._is_speech:
                if self._hangover_remaining > 0:
                    self._hangover_remaining -= 1
                    self._is_speech = True
                else:
                    self._is_speech = False
            else:
                self._is_speech = False
                self._hangover_remaining = 0

        clamped_confidence = max(0.0, min(1.0, float(prob)))
        return VadResult(
            is_speech=self._is_speech,
            confidence=clamped_confidence,
            energy_db=float(energy_db),
        )

    def _run_inference(self, input_tensor: np.ndarray) -> float:
        """Run inference on the underlying session and return speech probability."""
        if self._session is None:
            return 0.0

        input_feed: dict[str, Any] = {"input": input_tensor}
        if self._has_sr_input:
            input_feed["sr"] = np.array(self._target_sr, dtype=np.int64)

        if self._is_v5 and self._has_state_input and self._state is not None:
            input_feed["state"] = self._state
        elif (
            not self._is_v5
            and self._has_state_input
            and self._h is not None
            and self._c is not None
        ):
            input_feed["h"] = self._h
            input_feed["c"] = self._c

        outputs = self._session.run(None, input_feed)
        out_prob = float(outputs[0].flatten()[0])

        # Update state tensors from output
        if self._is_v5 and len(outputs) > 1:
            self._state = outputs[1]
        elif not self._is_v5 and len(outputs) > 2:
            self._h = outputs[1]
            self._c = outputs[2]

        return out_prob

    def reset(self) -> None:
        """Reset internal history buffer, hysteresis, and model state tensors."""
        self._is_speech = False
        self._hangover_remaining = 0
        self._history = np.zeros(self._context_size, dtype=np.float32)
        if self._is_v5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        if self._is_fallback and self._fallback_vad is not None:
            # Recreate clean EnergyVad
            self._fallback_vad = EnergyVad(
                speech_threshold_db=12.0,
                hangover_frames=self._hangover_frames,
            )
