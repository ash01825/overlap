"""Multi-script backchannel, stop, contrast, and question lexicons and matcher.

Pure Python core module (§7.2). Pre-loads YAML lexicons for 11 languages
(en + hi, ta, te, kn, ml, bn, mr, gu, pa, or) via importlib.resources at import
time into in-memory frozen data structures. Zero file I/O at call time.

Invariants:
- Union-of-languages matching with lang_hint as tiebreak.
- Normalization: NFC, casefold, punctuation/danda strip, elongation collapse.
- Fuzzy matching (Levenshtein distance <= 1, token len >= 4) ONLY for romanized
  backchannels, never for stop/contrast.
"""

from __future__ import annotations

import importlib.resources
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

import yaml  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class Analysis:
    """Lexical analysis of an utterance for full-duplex classification (§7)."""

    tokens: tuple[str, ...]
    bc_tokens: tuple[str, ...]
    stop_hit: bool
    contrast_hit: bool
    question: bool
    content_tokens: tuple[str, ...]

    @property
    def is_pure_backchannel(self) -> bool:
        """Return True if utterance contains only backchannels and no interruptions/content."""
        return (
            bool(self.tokens)
            and len(self.bc_tokens) == len(self.tokens)
            and not self.stop_hit
            and not self.contrast_hit
            and not self.question
            and len(self.content_tokens) == 0
        )


@dataclass(frozen=True, slots=True)
class LanguageLexicon:
    """Frozen lexical entries for a single language."""

    lang: str
    backchannel: tuple[str, ...]
    stop: tuple[str, ...]
    contrast: tuple[str, ...]
    question: tuple[str, ...]


# Punctuation to replace with spaces (including dandas, quotes, dashes)
_PUNCT_SPLIT_RE = re.compile(
    r"[\t\n\r\.,;:!?¿؟।॥\"“”‘’«»—–…\(\)\[\]\{\}\<\>\/\\\|\+\=\*\&\^\%\$\#\@\~-]+"
)
_DANDA_RE = re.compile(r"[।॥]")
_REPEAT_3_RE = re.compile(r"(.)\1{2,}")
_VOWELS = frozenset("aeiou")


def _levenshtein_le_1(s1: str, s2: str) -> bool:
    """Check if Levenshtein distance between s1 and s2 is <= 1 (fast exit)."""
    if s1 == s2:
        return True
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > 1:
        return False
    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1
    # len2 is either len1 or len1 + 1
    if len1 == len2:
        mismatches = 0
        for c1, c2 in zip(s1, s2, strict=True):
            if c1 != c2:
                mismatches += 1
                if mismatches > 1:
                    return False
        return True
    else:
        i = j = mismatches = 0
        while i < len1 and j < len2:
            if s1[i] == s2[j]:
                i += 1
                j += 1
            else:
                mismatches += 1
                if mismatches > 1:
                    return False
                j += 1
        return True


def _is_valid_romanized_fuzzy(token: str, cand: str) -> bool:
    """Check if edit distance <= 1 without turning content words into false positives."""
    if token == cand:
        return True
    len1, len2 = len(token), len(cand)
    if abs(len1 - len2) > 1:
        return False

    # For 4-letter words with 1 substitution:
    # Restrict to vowel variations (e.g. 'sari'/'seri') or 'h' variations.
    # Consonant substitutions (like 'l' vs 'n' in 'haal'/'haan', 'c' vs 's' in 'care'/'sare')
    # create false positives on content words.
    if len1 == 4 and len2 == 4:
        diff_t: str | None = None
        diff_c: str | None = None
        for c1, c2 in zip(token, cand, strict=True):
            if c1 != c2:
                if diff_t is not None:
                    return False
                diff_t = c1
                diff_c = c2
        if diff_t is None:
            return True
        if (diff_t not in _VOWELS and diff_t != "h") and (diff_c not in _VOWELS and diff_c != "h"):
            return False
        return True

    return _levenshtein_le_1(token, cand)


class Lexicon:
    """In-memory multi-script Indic and English lexicon matcher."""

    _instance: Lexicon | None = None

    def __init__(self, languages: Mapping[str, LanguageLexicon]) -> None:
        self._languages: dict[str, LanguageLexicon] = dict(languages)

        # Union sets of phrases (multi-word) and words (single-word)
        self._stop_phrases: set[tuple[str, ...]] = set()
        self._contrast_phrases: set[tuple[str, ...]] = set()
        self._question_phrases: set[tuple[str, ...]] = set()
        self._bc_phrases: set[tuple[str, ...]] = set()

        self._stop_words: set[str] = set()
        self._contrast_words: set[str] = set()
        self._question_words: set[str] = set()
        self._bc_words: set[str] = set()

        # Romanized backchannels with length >= 4 for fuzzy matching
        romanized_bc: set[str] = set()

        # Per-language lookups for tiebreak
        self._lang_stop: dict[str, set[str]] = {}
        self._lang_contrast: dict[str, set[str]] = {}
        self._lang_question: dict[str, set[str]] = {}
        self._lang_bc: dict[str, set[str]] = {}

        for lang, l_lex in self._languages.items():
            l_stop_w: set[str] = set()
            l_contrast_w: set[str] = set()
            l_question_w: set[str] = set()
            l_bc_w: set[str] = set()

            for item in l_lex.stop:
                norm = self._normalize_entry(item)
                parts = tuple(norm.split())
                if len(parts) > 1:
                    self._stop_phrases.add(parts)
                elif len(parts) == 1:
                    self._stop_words.add(parts[0])
                    l_stop_w.add(parts[0])

            for item in l_lex.contrast:
                norm = self._normalize_entry(item)
                parts = tuple(norm.split())
                if len(parts) > 1:
                    self._contrast_phrases.add(parts)
                elif len(parts) == 1:
                    self._contrast_words.add(parts[0])
                    l_contrast_w.add(parts[0])

            for item in l_lex.question:
                norm = self._normalize_entry(item)
                parts = tuple(norm.split())
                if len(parts) > 1:
                    self._question_phrases.add(parts)
                elif len(parts) == 1:
                    self._question_words.add(parts[0])
                    l_question_w.add(parts[0])

            for item in l_lex.backchannel:
                norm = self._normalize_entry(item)
                parts = tuple(norm.split())
                if len(parts) > 1:
                    self._bc_phrases.add(parts)
                elif len(parts) == 1:
                    w = parts[0]
                    self._bc_words.add(w)
                    l_bc_w.add(w)
                    if w.isascii() and len(w) >= 4:
                        romanized_bc.add(w)

            self._lang_stop[lang] = l_stop_w
            self._lang_contrast[lang] = l_contrast_w
            self._lang_question[lang] = l_question_w
            self._lang_bc[lang] = l_bc_w

        self._romanized_bc: tuple[str, ...] = tuple(romanized_bc)

    @classmethod
    def load(cls) -> Lexicon:
        """Load YAML lexicons from package resources."""
        if cls._instance is not None:
            return cls._instance

        lex_dir = importlib.resources.files("overlap.core").joinpath("lexicons")
        langs = ("en", "hi", "ta", "te", "kn", "ml", "bn", "mr", "gu", "pa", "or")
        lex_map: dict[str, LanguageLexicon] = {}

        for lang in langs:
            yaml_res = lex_dir.joinpath(f"{lang}.yaml")
            text = yaml_res.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
            lex_map[lang] = LanguageLexicon(
                lang=lang,
                backchannel=tuple(str(x) for x in data.get("backchannel", ())),
                stop=tuple(str(x) for x in data.get("stop", ())),
                contrast=tuple(str(x) for x in data.get("contrast", ())),
                question=tuple(str(x) for x in data.get("question", ())),
            )

        cls._instance = cls(lex_map)
        return cls._instance

    @property
    def languages(self) -> tuple[str, ...]:
        """Return tuple of available language codes."""
        return tuple(self._languages.keys())

    def get_language(self, lang: str) -> LanguageLexicon | None:
        """Get lexicon entries for specific language."""
        return self._languages.get(lang)

    @staticmethod
    def _normalize_entry(text: str) -> str:
        """NFC normalize, casefold, and strip extraneous symbols."""
        norm = unicodedata.normalize("NFC", text).casefold()
        norm = _PUNCT_SPLIT_RE.sub(" ", norm)
        return " ".join(norm.split())

    def _collapse_token(self, token: str) -> str:
        """Collapse elongations (e.g. 'haaaan' -> 'haan', 'rukoooo' -> 'ruko')."""
        if not _REPEAT_3_RE.search(token):
            return token

        # Collapse to 2 repeats (e.g. haaaan -> haan, hmmmm -> hmm)
        c2 = _REPEAT_3_RE.sub(r"\1\1", token)
        if (
            c2 in self._bc_words
            or c2 in self._stop_words
            or c2 in self._contrast_words
            or c2 in self._question_words
        ):
            return c2

        # Collapse to 1 repeat (e.g. rukoooo -> ruko, stopppp -> stop, waaaait -> wait)
        c1 = _REPEAT_3_RE.sub(r"\1", token)
        if (
            c1 in self._stop_words
            or c1 in self._contrast_words
            or c1 in self._question_words
            or c1 in self._bc_words
        ):
            return c1

        return c2

    def analyze(self, text: str, lang_hint: str | None = None) -> Analysis:
        """Analyze utterance text into lexical features (§7.1, §7.2)."""
        raw_text = unicodedata.normalize("NFC", text)
        has_qmark = ("?" in raw_text) or ("؟" in raw_text) or ("¿" in raw_text)

        # Normalize and extract tokens
        norm = raw_text.casefold()
        # Replace dandas and sentence punctuation with spaces
        norm = _PUNCT_SPLIT_RE.sub(" ", norm)
        raw_tokens = [t.strip("'\"") for t in norm.split() if t.strip("'\"")]

        # Collapse elongations per token
        tokens = tuple(self._collapse_token(t) for t in raw_tokens)
        n = len(tokens)
        if n == 0:
            return Analysis(
                tokens=(),
                bc_tokens=(),
                stop_hit=False,
                contrast_hit=False,
                question=has_qmark,
                content_tokens=(),
            )

        tag = [""] * n
        stop_hit = False
        contrast_hit = False
        question_hit = has_qmark

        # Multi-word phrase matching (longest phrase first: 3 down to 2)
        for span_len in (3, 2):
            for i in range(n - span_len + 1):
                if any(tag[i + k] != "" for k in range(span_len)):
                    continue
                sub = tokens[i : i + span_len]
                if sub in self._stop_phrases:
                    for k in range(span_len):
                        tag[i + k] = "stop"
                    stop_hit = True
                elif sub in self._contrast_phrases:
                    for k in range(span_len):
                        tag[i + k] = "contrast"
                    contrast_hit = True
                elif sub in self._question_phrases:
                    for k in range(span_len):
                        tag[i + k] = "question"
                    question_hit = True
                elif sub in self._bc_phrases:
                    for k in range(span_len):
                        tag[i + k] = "bc"

        # Single token classification
        for i in range(n):
            if tag[i] != "":
                continue
            t = tokens[i]

            # Lang-hint tiebreak check
            if lang_hint and lang_hint in self._languages:
                if t in self._lang_stop[lang_hint]:
                    tag[i] = "stop"
                    stop_hit = True
                    continue
                if t in self._lang_contrast[lang_hint]:
                    tag[i] = "contrast"
                    contrast_hit = True
                    continue
                if t in self._lang_question[lang_hint]:
                    tag[i] = "question"
                    question_hit = True
                    continue
                if t in self._lang_bc[lang_hint]:
                    tag[i] = "bc"
                    continue

            # Union exact matches
            if t in self._stop_words:
                tag[i] = "stop"
                stop_hit = True
            elif t in self._contrast_words:
                tag[i] = "contrast"
                contrast_hit = True
            elif t in self._question_words:
                tag[i] = "question"
                question_hit = True
            elif t in self._bc_words:
                tag[i] = "bc"
            else:
                # Fuzzy match: ONLY for romanized backchannel, len >= 4, edit dist <= 1
                matched_fuzzy = False
                if t.isascii() and len(t) >= 4:
                    for cand in self._romanized_bc:
                        if _is_valid_romanized_fuzzy(t, cand):
                            tag[i] = "bc"
                            matched_fuzzy = True
                            break
                if not matched_fuzzy:
                    tag[i] = "content"

        bc_tokens = tuple(tokens[i] for i in range(n) if tag[i] == "bc")
        content_tokens = tuple(tokens[i] for i in range(n) if tag[i] == "content")

        return Analysis(
            tokens=tokens,
            bc_tokens=bc_tokens,
            stop_hit=stop_hit,
            contrast_hit=contrast_hit,
            question=question_hit,
            content_tokens=content_tokens,
        )


# Global singleton instance loaded at import time
_DEFAULT_LEXICON = Lexicon.load()


def analyze(text: str, lang_hint: str | None = None) -> Analysis:
    """Analyze utterance text into lexical features using the global lexicon."""
    return _DEFAULT_LEXICON.analyze(text, lang_hint=lang_hint)
