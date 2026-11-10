"""Pure audio math and DSP primitives for OVERLAP.

Per ARCHITECTURE §8, §9, §10:
- PCM 16-bit <-> float32 conversion and saturating addition/mixing.
- RMS dBFS calculation with floor at -90.0 dBFS.
- Click-free linear GainRamp with bounded per-sample delta (INV-L3, INV-L8).
- LoudnessTracker maintaining EMA of agent reference RMS for relative loudness.
- EchoGate adaptive echo suppression and self-audio window gating.
- EnergyVad 1-frame onset detector with hangover offset fulfilling Vad port (INV-L1).
"""

from __future__ import annotations

import importlib
import math
import sys
from typing import Any

import numpy as np

from overlap.core.types import AudioFrame, Tag

# =============================================================================
# PCM conversion & math utils
# =============================================================================


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Convert 16-bit signed little-endian PCM bytes to float32 in [-1.0, 1.0]."""
    if not pcm:
        return np.empty(0, dtype=np.float32)
    return (np.frombuffer(pcm, dtype=np.int16).astype(np.float32)) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 samples to 16-bit signed PCM bytes with saturating clamping."""
    if len(audio) == 0:
        return b""
    scaled = audio * 32768.0
    clipped = np.clip(np.rint(scaled), -32768.0, 32767.0).astype(np.int16)
    return bytes(clipped.tobytes())


def rms_dbfs(pcm_or_floats: bytes | np.ndarray) -> float:
    """Calculate root-mean-square level in dBFS, floored at -90.0 dBFS."""
    if isinstance(pcm_or_floats, (bytes, bytearray)):
        floats = pcm16_to_float32(bytes(pcm_or_floats))
    else:
        floats = pcm_or_floats

    if len(floats) == 0:
        return -90.0

    mean_sq = float(np.mean(np.square(floats)))
    if mean_sq <= 0.0:
        return -90.0

    rms = math.sqrt(mean_sq)
    if rms <= 1e-9:
        return -90.0

    dbfs = 20.0 * math.log10(rms)
    return max(-90.0, float(dbfs))


def mix_pcm16(*pcm_tracks: bytes) -> bytes:
    """Saturating element-wise addition of 16-bit PCM tracks to prevent integer overflow."""
    valid_tracks = [t for t in pcm_tracks if t]
    if not valid_tracks:
        return b""
    if len(valid_tracks) == 1:
        return valid_tracks[0]

    max_bytes = max(len(t) for t in valid_tracks)
    max_samples = max_bytes // 2
    summed = np.zeros(max_samples, dtype=np.int32)

    for track in valid_tracks:
        samples = np.frombuffer(track, dtype=np.int16).astype(np.int32)
        summed[: len(samples)] += samples

    clipped = np.clip(summed, -32768, 32767).astype(np.int16)
    return bytes(clipped.tobytes())


# =============================================================================
# Click-Free GainRamp (INV-L3, INV-L8)
# =============================================================================


class GainRamp:
    """Linear per-sample gain ramp across audio frames.

    Provides click-free ducking attack (<= 20 ms) and release (150 ms)
    with bounded delta-sample transitions between frames.
    """

    def __init__(
        self,
        current: float = 1.0,
        target: float = 1.0,
        ramp_ms: int = 0,
        sample_rate: int = 16000,
    ) -> None:
        self._current: float = float(current)
        self._target: float = float(target)
        self._sample_rate: int = sample_rate
        ramp_samples = max(0, int(sample_rate * (ramp_ms / 1000.0)))
        self._remaining_samples: int = ramp_samples
        self._step: float = (
            (self._target - self._current) / ramp_samples if ramp_samples > 0 else 0.0
        )
        if ramp_samples == 0:
            self._current = self._target

    @property
    def current_gain(self) -> float:
        """Current instantaneous gain factor."""
        return self._current

    @property
    def target_gain(self) -> float:
        """Target gain factor of the ramp."""
        return self._target

    @property
    def is_finished(self) -> bool:
        """True when the gain ramp has reached target gain."""
        return self._remaining_samples == 0

    def set_target(self, target: float, ramp_ms: int) -> None:
        """Begin ramping towards a new target gain over ramp_ms."""
        self._target = float(target)
        ramp_samples = max(0, int(self._sample_rate * (ramp_ms / 1000.0)))
        self._remaining_samples = ramp_samples
        self._step = (self._target - self._current) / ramp_samples if ramp_samples > 0 else 0.0
        if ramp_samples == 0:
            self._current = self._target

    def apply(self, pcm: bytes) -> bytes:
        """Apply the gain ramp per-sample across the provided PCM frame."""
        if not pcm:
            return b""

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        n = len(samples)

        if self._remaining_samples == 0:
            out = samples * self._current
        else:
            ramp_len = min(n, self._remaining_samples)
            indices = np.arange(1, ramp_len + 1, dtype=np.float32)
            curve = self._current + indices * self._step

            if self._step > 0:
                curve = np.minimum(curve, self._target)
            elif self._step < 0:
                curve = np.maximum(curve, self._target)

            if ramp_len < n:
                flat = np.full(n - ramp_len, self._target, dtype=np.float32)
                gain_factors = np.concatenate([curve, flat])
                self._current = self._target
                self._remaining_samples = 0
            else:
                gain_factors = curve
                self._current = float(curve[-1])
                self._remaining_samples -= ramp_len
                if self._remaining_samples == 0:
                    self._current = self._target

            out = samples * gain_factors

        clipped = np.clip(np.rint(out), -32768.0, 32767.0).astype(np.int16)
        return bytes(clipped.tobytes())


# =============================================================================
# LoudnessTracker
# =============================================================================


class LoudnessTracker:
    """Maintains an exponential moving average (EMA) of agent reference RMS.

    Used to compute the relative loudness of user speech in dBFS.
    Defined behaviour when the reference audio is silent (no div by zero or -inf).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        silence_floor_db: float = -90.0,
    ) -> None:
        self._alpha: float = alpha
        self._silence_floor_db: float = silence_floor_db
        self._agent_rms: float = 0.0
        self._has_agent_audio: bool = False

    def update_agent(self, pcm_or_rms: bytes | float) -> None:
        """Update the agent reference RMS EMA with newly sent frame audio."""
        if isinstance(pcm_or_rms, (bytes, bytearray)):
            floats = pcm16_to_float32(bytes(pcm_or_rms))
            rms = float(np.sqrt(np.mean(np.square(floats)))) if len(floats) > 0 else 0.0
        else:
            rms = float(pcm_or_rms)

        if not self._has_agent_audio:
            self._agent_rms = rms
            self._has_agent_audio = True
        else:
            self._agent_rms = self._alpha * rms + (1.0 - self._alpha) * self._agent_rms

    def rel_loudness_db(self, user_pcm_or_rms: bytes | float) -> float:
        """Compute relative loudness in dB: user_db - agent_ref_db.

        Guaranteed to return a finite, defined number even when the reference is silent.
        """
        if isinstance(user_pcm_or_rms, (bytes, bytearray)):
            user_db = rms_dbfs(bytes(user_pcm_or_rms))
        else:
            user_rms = float(user_pcm_or_rms)
            user_db = 20.0 * math.log10(user_rms) if user_rms > 1e-9 else self._silence_floor_db
            user_db = max(self._silence_floor_db, user_db)

        if not self._has_agent_audio or self._agent_rms <= 1e-9:
            if user_db <= self._silence_floor_db:
                return 0.0
            return float(user_db - self._silence_floor_db)

        ref_db = max(self._silence_floor_db, 20.0 * math.log10(self._agent_rms))
        if ref_db <= self._silence_floor_db:
            if user_db <= self._silence_floor_db:
                return 0.0
            return float(user_db - self._silence_floor_db)

        return float(user_db - ref_db)


# =============================================================================
# EchoGate (AEC Safety Net)
# =============================================================================


class EchoGate:
    """Acoustic echo and self-backchannel suppression gate.

    Compares mic RMS against reference RMS * adaptive ERL * margin.
    Suppresses audio during registered self-audio windows (+150 ms expiry).
    """

    def __init__(
        self,
        initial_erl: float = 0.25,
        margin_factor: float = 1.2,
        adaptation_rate: float = 0.05,
        silence_floor_db: float = -90.0,
    ) -> None:
        self._erl: float = initial_erl
        self._margin: float = margin_factor
        self._adapt_rate: float = adaptation_rate
        self._silence_floor_db: float = silence_floor_db
        self._self_windows: list[tuple[int, int, Tag]] = []

    def register_self_audio_window(
        self,
        start_ns: int,
        end_ns: int,
        tag: Tag = Tag.SELF_BACKCHANNEL,
    ) -> None:
        """Register a self-audio playout window active until end_ns + 150 ms."""
        expiry_ns = end_ns + 150_000_000
        self._self_windows.append((start_ns, expiry_ns, tag))

    def is_in_self_audio_window(self, now_ns: int) -> bool:
        """Check whether now_ns falls within any active self-audio window."""
        self._self_windows = [w for w in self._self_windows if now_ns <= w[1]]
        return any(w[0] <= now_ns <= w[1] for w in self._self_windows)

    def is_echo(
        self,
        mic: bytes | AudioFrame | float,
        ref_rms: float = 0.0,
        now_ns: int = 0,
    ) -> bool:
        """Return True if mic audio is classified as echo/self-talk (suppressed)."""
        if isinstance(mic, AudioFrame):
            now_ns = mic.ts_ns
            floats = pcm16_to_float32(mic.pcm)
            mic_rms = float(np.sqrt(np.mean(np.square(floats)))) if len(floats) > 0 else 0.0
        elif isinstance(mic, (bytes, bytearray)):
            floats = pcm16_to_float32(bytes(mic))
            mic_rms = float(np.sqrt(np.mean(np.square(floats)))) if len(floats) > 0 else 0.0
        else:
            mic_rms = float(mic)

        if now_ns > 0 and self.is_in_self_audio_window(now_ns):
            if mic_rms < 0.25:
                return True

        if ref_rms <= 1e-9:
            return False

        echo_threshold = ref_rms * self._erl * self._margin
        if mic_rms <= echo_threshold:
            if mic_rms > 1e-6:
                observed_erl = mic_rms / ref_rms
                self._erl = (1.0 - self._adapt_rate) * self._erl + self._adapt_rate * observed_erl
                self._erl = min(0.8, max(0.05, self._erl))
            return True

        return False


# =============================================================================
# EnergyVad (INV-L1, Vad Port Implementation)
# =============================================================================


class EnergyVad:
    """Voice Activity Detector with 1-frame onset and hangover offset.

    Satisfies Vad port protocol without violating core purity or layering contracts.
    """

    def __init__(
        self,
        speech_threshold_db: float = 12.0,
        min_noise_floor_db: float = -60.0,
        initial_noise_floor_db: float = -60.0,
        noise_adapt_rate: float = 0.05,
        hangover_frames: int = 5,
        silence_floor_db: float = -90.0,
    ) -> None:
        self._speech_threshold_db: float = speech_threshold_db
        self._min_noise_floor_db: float = min_noise_floor_db
        self._noise_floor_db: float = initial_noise_floor_db
        self._noise_adapt_rate: float = noise_adapt_rate
        self._hangover_frames: int = hangover_frames
        self._silence_floor_db: float = silence_floor_db
        self._hangover_remaining: int = 0
        self._vad_result_cls: Any = None

    def _get_result_class(self) -> Any:
        if self._vad_result_cls is None:
            if "overlap.ports.vad" in sys.modules:
                mod = sys.modules["overlap.ports.vad"]
            else:
                mod = importlib.import_module("overlap.ports.vad")
            self._vad_result_cls = mod.VadResult
        return self._vad_result_cls

    def process(self, frame: AudioFrame) -> Any:
        """Process an audio frame and return a VadResult synchronously (INV-L1)."""
        energy_db = rms_dbfs(frame.pcm)
        threshold = max(self._min_noise_floor_db, self._noise_floor_db) + self._speech_threshold_db

        if energy_db >= threshold:
            is_speech = True
            self._hangover_remaining = self._hangover_frames
        else:
            if self._hangover_remaining > 0:
                self._hangover_remaining -= 1
                is_speech = True
            else:
                is_speech = False
                if energy_db > self._silence_floor_db:
                    self._noise_floor_db = (
                        1.0 - self._noise_adapt_rate
                    ) * self._noise_floor_db + self._noise_adapt_rate * energy_db

        if is_speech:
            confidence = min(1.0, max(0.5, 0.5 + (energy_db - threshold) / 30.0))
        else:
            confidence = max(0.0, min(0.49, 0.5 - (threshold - energy_db) / 30.0))

        res_cls = self._get_result_class()
        return res_cls(
            is_speech=is_speech,
            confidence=float(confidence),
            energy_db=float(energy_db),
        )
