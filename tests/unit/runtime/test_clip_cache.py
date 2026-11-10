"""Unit tests for ClipCache runtime module (§9, §10, INV-B1).

Tests coverage:
- O(1) synchronous get(lang) returning None on miss and never awaiting.
- prepare(langs) pre-synthesizing via TtsPort.synthesize.
- Resampling clips to out_rate_hz when synthesized at different sample rate.
- Disk caching keyed by (voice_id, lang, text, ssml) and reuse on second prepare.
- Rotation when text is omitted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from virtual import run_virtual

from overlap.core.types import Tuning
from overlap.ports.tts import AudioClip, TtsPort, TtsStream
from overlap.runtime.clip_cache import ClipCache, resample_pcm


class RecordingTts(TtsPort):
    """Fake TTS engine recording synthesize calls and returning configurable rate audio."""

    def __init__(self, sample_rate: int = 16000, clip_duration_ms: int = 100) -> None:
        self.sample_rate = sample_rate
        self.clip_duration_ms = clip_duration_ms
        self.synthesize_calls: list[dict[str, Any]] = []

    async def open_stream(
        self,
        *,
        voice_id: str,
        lang: str,
        epoch: int,
        cancel: Any = None,
    ) -> TtsStream:
        raise NotImplementedError("Not used in clip cache tests")

    async def synthesize(self, text: str, *, voice_id: str = "v1", lang: str = "hi") -> AudioClip:
        self.synthesize_calls.append({"text": text, "voice_id": voice_id, "lang": lang})
        sample_count = int(self.sample_rate * (self.clip_duration_ms / 1000.0))
        # Deterministic non-zero 16-bit PCM (e.g. ramp)
        pcm = bytes([(i % 256) for i in range(sample_count * 2)])
        return AudioClip(pcm=pcm, sample_rate=self.sample_rate)


def test_clip_cache_miss_returns_none_synchronously() -> None:
    """Failure mode: ClipCache blocks the live path on a miss."""
    tts = RecordingTts()
    cache = ClipCache(tts=tts, voice_id="v1", out_rate_hz=16000)

    # get() must be synchronous, O(1), and never invoke TTS on a miss
    assert cache.get("hi") is None
    assert cache.get("hi", text="haan") is None
    assert cache.get("unknown_lang") is None
    assert len(tts.synthesize_calls) == 0


def test_resample_pcm_identity_and_rate_change() -> None:
    """Verify linear interpolation resampling for 16-bit PCM."""
    # Identity
    raw = b"\x00\x01\x00\x02\x00\x03\x00\x04"
    assert resample_pcm(raw, 16000, 16000) == raw

    # Empty
    assert resample_pcm(b"", 8000, 16000) == b""

    # 8000 Hz to 16000 Hz (sample count doubles)
    in_pcm = (1000).to_bytes(2, "little", signed=True) * 10
    out_pcm = resample_pcm(in_pcm, 8000, 16000)
    assert len(out_pcm) == 40  # 20 samples * 2 bytes


def test_clip_cache_prepare_presynthesizes_and_resamples(tmp_path: Path) -> None:
    """ClipCache.prepare(langs) pre-synthesizes and guarantees out_rate_hz."""

    async def _test() -> None:
        tts = RecordingTts(sample_rate=24000, clip_duration_ms=50)
        tuning = Tuning(out_rate_hz=16000)
        cache = ClipCache(
            tts=tts,
            voice_id="agent_voice",
            out_rate_hz=16000,
            cache_dir=tmp_path,
            clips_by_lang={"hi": ("haan", "theek hai")},
            tuning=tuning,
        )

        await cache.prepare(["hi"])

        # 2 clips synthesized
        assert len(tts.synthesize_calls) == 2
        assert tts.synthesize_calls[0] == {"text": "haan", "voice_id": "agent_voice", "lang": "hi"}

        # get(lang, text) returns resampled clip at out_rate_hz (16000)
        clip_haan = cache.get("hi", text="haan")
        assert clip_haan is not None
        assert clip_haan.sample_rate == 16000
        assert len(clip_haan.pcm) > 0

        # get(lang) returns in rotation
        c1 = cache.get("hi")
        c2 = cache.get("hi")
        assert c1 is not None and c2 is not None
        assert c1.sample_rate == 16000
        assert c2.sample_rate == 16000

    run_virtual(_test())


def test_clip_cache_disk_caching_and_reuse(tmp_path: Path) -> None:
    """Disk cache keyed by (voice_id, lang, text, ssml) avoids re-synthesizing."""

    async def _test() -> None:
        tts = RecordingTts(sample_rate=16000)
        cache1 = ClipCache(
            tts=tts,
            voice_id="v1",
            out_rate_hz=16000,
            cache_dir=tmp_path,
            clips_by_lang={"hi": ("haan",)},
        )

        await cache1.prepare(["hi"])
        assert len(tts.synthesize_calls) == 1

        # Verify disk files exist
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        meta = json.loads(files[0].read_text(encoding="utf-8"))
        assert meta["voice_id"] == "v1"
        assert meta["lang"] == "hi"
        assert meta["text"] == "haan"
        assert meta["sample_rate"] == 16000

        pcm_files = list(tmp_path.glob("*.pcm"))
        assert len(pcm_files) == 1

        # Second instance with fresh memory cache but same disk cache
        tts2 = RecordingTts(sample_rate=16000)
        cache2 = ClipCache(
            tts=tts2,
            voice_id="v1",
            out_rate_hz=16000,
            cache_dir=tmp_path,
            clips_by_lang={"hi": ("haan",)},
        )

        await cache2.prepare(["hi"])
        # Zero calls to tts2: loaded from disk!
        assert len(tts2.synthesize_calls) == 0

        clip = cache2.get("hi", text="haan")
        assert clip is not None
        assert clip.sample_rate == 16000

    run_virtual(_test())
