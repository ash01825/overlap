"""ClipCache: Pre-synthesized backchannel audio clip cache (§9, §10, INV-B1).

Pre-synthesizes clips per language via TtsPort.synthesize at session start,
resampling to out_rate_hz and caching on disk keyed by (voice_id, lang, text, ssml).
get(lang) is strictly synchronous, O(1), and never blocks or awaits on a miss.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from overlap.core.predictor import DEFAULT_CLIPS_BY_LANG
from overlap.core.types import Tuning
from overlap.ports.tts import AudioClip, TtsPort


def resample_pcm(pcm: bytes, in_rate: int, out_rate: int) -> bytes:
    """Resample 16-bit mono PCM audio between sample rates using numpy linear interpolation.

    Fast, hermetic, and guarantees output sample rate alignment.
    """
    if in_rate == out_rate or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16)
    in_len = len(samples)
    if in_len == 0:
        return b""
    out_len = int(round(in_len * out_rate / in_rate))
    if out_len == 0:
        return b""
    old_x = np.linspace(0.0, in_len - 1.0, in_len)
    new_x = np.linspace(0.0, in_len - 1.0, out_len)
    resampled = np.interp(new_x, old_x, samples).astype(np.int16)
    return resampled.tobytes()


class ClipCache:
    """In-memory and on-disk cache for pre-synthesized backchannel audio clips."""

    def __init__(
        self,
        tts: TtsPort,
        voice_id: str = "v1",
        out_rate_hz: int | None = None,
        cache_dir: Path | str | None = None,
        clips_by_lang: Mapping[str, Sequence[str]] | None = None,
        tuning: Tuning | None = None,
    ) -> None:
        self._tts: TtsPort = tts
        self._voice_id: str = voice_id
        t = tuning or Tuning()
        self._out_rate_hz: int = out_rate_hz if out_rate_hz is not None else t.out_rate_hz
        self._cache_dir: Path = Path(cache_dir) if cache_dir is not None else Path(".cache/clips")
        self._clips_by_lang: dict[str, tuple[str, ...]] = (
            {k: tuple(v) for k, v in clips_by_lang.items()}
            if clips_by_lang is not None
            else dict(DEFAULT_CLIPS_BY_LANG)
        )

        # In-memory fast lookups: (voice_id, lang, text, ssml) -> AudioClip
        self._memory_cache: dict[tuple[str, str, str, bool], AudioClip] = {}
        # Per-language rotation pool
        self._by_lang: dict[str, list[AudioClip]] = {}
        self._lang_rr: dict[str, int] = {}

    @property
    def voice_id(self) -> str:
        """Voice identifier for synthesis."""
        return self._voice_id

    @property
    def out_rate_hz(self) -> int:
        """Target audio output sample rate."""
        return self._out_rate_hz

    @property
    def cache_dir(self) -> Path:
        """Path to on-disk cache directory."""
        return self._cache_dir

    def _hash_key(self, voice_id: str, lang: str, text: str, ssml: bool) -> str:
        """Generate deterministic hex hash for disk cache file names."""
        payload = f"{voice_id}\x00{lang}\x00{text}\x00{ssml}".encode()
        return hashlib.sha256(payload).hexdigest()

    async def prepare(self, langs: Sequence[str]) -> None:
        """Pre-synthesize clips for requested languages via TtsPort.synthesize.

        Checks disk cache first; on miss synthesizes, resamples to out_rate_hz,
        and saves to disk cache.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        for lang in langs:
            clip_texts = self._clips_by_lang.get(lang, ())
            for text in clip_texts:
                cache_key = (self._voice_id, lang, text, False)
                if cache_key in self._memory_cache:
                    continue

                key_hash = self._hash_key(self._voice_id, lang, text, False)
                meta_path = self._cache_dir / f"{key_hash}.json"
                pcm_path = self._cache_dir / f"{key_hash}.pcm"

                if meta_path.is_file() and pcm_path.is_file():
                    # Load from disk cache
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        raw_pcm = pcm_path.read_bytes()
                        disk_rate = int(meta.get("sample_rate", self._out_rate_hz))
                        if disk_rate != self._out_rate_hz:
                            pcm = resample_pcm(raw_pcm, disk_rate, self._out_rate_hz)
                        else:
                            pcm = raw_pcm
                        clip = AudioClip(pcm=pcm, sample_rate=self._out_rate_hz)
                    except Exception:
                        # Fallback to synthesizing if disk load corrupted
                        clip = await self._synthesize_and_store(text, lang, meta_path, pcm_path)
                else:
                    clip = await self._synthesize_and_store(text, lang, meta_path, pcm_path)

                self._memory_cache[cache_key] = clip
                if lang not in self._by_lang:
                    self._by_lang[lang] = []
                self._by_lang[lang].append(clip)

    async def _synthesize_and_store(
        self, text: str, lang: str, meta_path: Path, pcm_path: Path
    ) -> AudioClip:
        """Synthesize a clip via TTS port, resample, and persist to disk."""
        raw_clip = await self._tts.synthesize(text, voice_id=self._voice_id, lang=lang)
        if raw_clip.sample_rate != self._out_rate_hz:
            pcm = resample_pcm(raw_clip.pcm, raw_clip.sample_rate, self._out_rate_hz)
        else:
            pcm = raw_clip.pcm

        clip = AudioClip(pcm=pcm, sample_rate=self._out_rate_hz)

        try:
            pcm_path.write_bytes(pcm)
            meta = {
                "voice_id": self._voice_id,
                "lang": lang,
                "text": text,
                "ssml": False,
                "sample_rate": self._out_rate_hz,
            }
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        except Exception:
            pass  # Best effort disk caching; never fail preparation

        return clip

    def get(self, lang: str, text: str | None = None, ssml: bool = False) -> AudioClip | None:
        """Retrieve pre-synthesized clip in O(1) synchronous time.

        Never awaits or performs I/O. A miss returns None immediately.
        """
        if text is not None:
            key = (self._voice_id, lang, text, ssml)
            return self._memory_cache.get(key)

        clips = self._by_lang.get(lang)
        if not clips:
            return None

        idx = self._lang_rr.get(lang, 0)
        clip = clips[idx % len(clips)]
        self._lang_rr[lang] = idx + 1
        return clip

    def get_by_key(
        self, voice_id: str, lang: str, text: str, ssml: bool = False
    ) -> AudioClip | None:
        """Direct key lookup in memory cache."""
        return self._memory_cache.get((voice_id, lang, text, ssml))
