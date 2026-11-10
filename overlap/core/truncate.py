"""Truncation of assistant speech to what the user heard.

Implements ARCHITECTURE §8.3:
- Midpoint rule for clauses with Timbre word timings.
- Calibrated speaking rate estimator (EMA over grapheme clusters via \\X)
  with snap-down to word boundaries.
- Multi-clause turn accounting and un-synthesized clause dropping.
- Suffixing with Tuning.truncation_marker when interrupted.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from overlap.core.types import Tuning, WordTiming


def _fallback_grapheme_clusters(text: str) -> list[str]:
    """Fallback grapheme cluster segmentation using Unicode combining categories.

    Combines non-spacing (Mn), spacing combining (Mc), and enclosing (Me) marks
    with the preceding base character to ensure matras and viramas are never orphaned.
    """
    clusters: list[str] = []
    curr = ""
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Mn", "Mc", "Me") and curr:
            curr += ch
        else:
            if curr:
                clusters.append(curr)
            curr = ch
    if curr:
        clusters.append(curr)
    return clusters


def split_grapheme_clusters(text: str) -> list[str]:
    """Split text into extended grapheme clusters (\\X).

    Prefers the `regex` library (Unicode TR29 extended grapheme clusters)
    and falls back to Unicode combining mark grouping.
    """
    if not text:
        return []
    try:
        import regex  # type: ignore[import-untyped]

        clusters: list[str] = regex.findall(r"\X", text)
        return clusters
    except (ImportError, Exception):
        return _fallback_grapheme_clusters(text)


def count_grapheme_clusters(text: str) -> int:
    """Return the number of grapheme clusters in text."""
    return len(split_grapheme_clusters(text))


# Default cold-start speech rates in milliseconds per grapheme cluster.
DEFAULT_LANG_RATES: dict[str, float] = {
    "en": 75.0,
    "hi": 90.0,
    "ta": 85.0,
    "te": 85.0,
    "kn": 85.0,
    "mr": 90.0,
    "bn": 90.0,
    "gu": 90.0,
    "ml": 90.0,
    "pa": 90.0,
    "or": 90.0,
    "as": 90.0,
    "ur": 90.0,
}


class SpeakingRateEstimator:
    """Exponential Moving Average (EMA) speaking rate estimator per language.

    Tracks duration per grapheme cluster (\\X) across completed clauses.
    Provides safe cold-start defaults for English and Indic languages so
    estimates never produce 0 or absurd word counts.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        default_ms_per_unit: float = 80.0,
        lang_defaults: dict[str, float] | None = None,
    ) -> None:
        self._alpha = max(0.01, min(1.0, alpha))
        self._default_ms_per_unit = max(20.0, default_ms_per_unit)
        self._lang_defaults = (
            dict(lang_defaults) if lang_defaults is not None else dict(DEFAULT_LANG_RATES)
        )
        self._rates: dict[str, float] = {}

    def _normalize_lang(self, lang: str | None) -> str:
        if not lang:
            return ""
        return lang.strip().lower().split("-")[0].split("_")[0]

    def get_ms_per_unit(self, lang: str | None = None) -> float:
        """Return estimated ms per grapheme cluster for language."""
        norm = self._normalize_lang(lang)
        if norm in self._rates:
            return self._rates[norm]
        if norm in self._lang_defaults:
            return self._lang_defaults[norm]
        return self._default_ms_per_unit

    def update(self, lang: str | None, audio_ms: int, text: str) -> float:
        """Update EMA for language with duration and text from a completed clause."""
        norm = self._normalize_lang(lang)
        if audio_ms <= 0 or not text.strip():
            return self.get_ms_per_unit(norm)

        units = count_grapheme_clusters(text)
        if units <= 0:
            return self.get_ms_per_unit(norm)

        observed = audio_ms / units
        if norm not in self._rates:
            new_rate = observed
        else:
            current = self._rates[norm]
            new_rate = self._alpha * observed + (1.0 - self._alpha) * current

        # Clamp to physical speech range (20 ms to 300 ms per cluster)
        clamped = max(20.0, min(300.0, new_rate))
        self._rates[norm] = clamped
        return clamped

    def update_from_clause(self, clause: ClauseAudio) -> float | None:
        """Helper to update rate estimate from a completed synthesized clause."""
        if clause.synthesized and clause.duration_ms > 0 and clause.text.strip():
            return self.update(clause.lang, clause.duration_ms, clause.text)
        return None


@dataclass(frozen=True, slots=True)
class ClauseAudio:
    """TTS clause audio metadata and optional word timings."""

    text: str
    duration_ms: int
    words: tuple[WordTiming, ...] = ()
    lang: str | None = None
    synthesized: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.words, tuple):
            object.__setattr__(self, "words", tuple(self.words))


@dataclass(frozen=True, slots=True)
class HeardText:
    """Result of truncating assistant speech to what was heard by the user."""

    text: str
    words: tuple[str, ...]
    fully_played: bool
    played_ms: int
    marker: str = ""

    @property
    def heard_words(self) -> tuple[str, ...]:
        """Alias for words to support property tests."""
        return self.words


def truncate_to_heard(
    clauses: Sequence[ClauseAudio],
    played_ms: int,
    rate_est: SpeakingRateEstimator | None = None,
    marker: str | None = None,
    tuning: Tuning | None = None,
) -> HeardText:
    """Truncate assistant speech across clauses to what the user heard per ARCHITECTURE §8.3.

    1. Per sent clause, if word timings exist: midpoint rule.
    2. Else calibrated rate: grapheme cluster rate snapped down to a word boundary.
    3. Clauses never started and un-synthesized remainder dropped entirely.
    4. Suffix with truncation_marker if interrupted.
    """
    t = tuning or Tuning()
    marker_str = t.truncation_marker if marker is None else marker
    est = rate_est or SpeakingRateEstimator()

    eff_played_ms = max(0, played_ms)
    remaining_ms = eff_played_ms

    heard_words_list: list[str] = []
    all_words_list: list[str] = []
    all_clauses_fully_played = True

    for clause in clauses:
        if clause.words:
            clause_words = tuple(w.word for w in clause.words)
        else:
            clause_words = tuple(clause.text.split())

        all_words_list.extend(clause_words)

        if not clause.synthesized:
            all_clauses_fully_played = False
            continue

        if remaining_ms <= 0:
            if clause_words:
                all_clauses_fully_played = False
            continue

        clause_dur = max(0, clause.duration_ms)
        clause_played_ms = min(remaining_ms, clause_dur)
        remaining_ms -= clause_played_ms

        if clause_played_ms < clause_dur:
            all_clauses_fully_played = False

        if clause.words:
            # Word timings available: midpoint rule
            for wt in clause.words:
                midpoint = (wt.start_ms + wt.end_ms) / 2.0
                if midpoint <= clause_played_ms:
                    heard_words_list.append(wt.word)
                else:
                    all_clauses_fully_played = False
                    break
        else:
            # Calibrated rate fallback
            if clause_played_ms >= clause_dur:
                heard_words_list.extend(clause_words)
            elif clause_played_ms <= 0:
                if clause_words:
                    all_clauses_fully_played = False
            else:
                ms_per_unit = est.get_ms_per_unit(clause.lang)
                heard_units = int(clause_played_ms / ms_per_unit)
                clusters = split_grapheme_clusters(clause.text)

                # Locate word boundaries in cluster list
                word_spans: list[tuple[str, int, int]] = []
                curr_c = 0
                total_c = len(clusters)
                for token in clause_words:
                    token_len = count_grapheme_clusters(token)
                    while curr_c < total_c and clusters[curr_c].isspace():
                        curr_c += 1
                    w_start = curr_c
                    w_end = curr_c + token_len
                    word_spans.append((token, w_start, w_end))
                    curr_c = w_end

                for token_str, _w_start, w_end in word_spans:
                    if w_end <= heard_units:
                        heard_words_list.append(token_str)
                    else:
                        all_clauses_fully_played = False
                        break

    fully_played = all_clauses_fully_played and (len(heard_words_list) == len(all_words_list))

    if not fully_played:
        if heard_words_list:
            text = f"{' '.join(heard_words_list)}{marker_str}"
            effective_marker = marker_str
        else:
            text = ""
            effective_marker = ""
    else:
        text = " ".join(heard_words_list)
        effective_marker = ""

    return HeardText(
        text=text,
        words=tuple(heard_words_list),
        fully_played=fully_played,
        played_ms=eff_played_ms,
        marker=effective_marker,
    )
