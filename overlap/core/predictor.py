"""Agent backchannel predictor with pause, clause, energy triggers and pacing.

Core, pure implementation (§9, §10, INV-B1).
Pure Python stdlib only. No asyncio, no time, no socket, no unseeded random.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from overlap.core.types import BackchannelDue, FsmState, SttEvent, SttPartial, Tuning

# =============================================================================
# Language Constants: Clips, Conjunctions, and Terminal Particles
# =============================================================================

DEFAULT_CLIPS_BY_LANG: dict[str, tuple[str, ...]] = {
    "en": ("yeah", "okay", "right", "mhm", "got it", "sure"),
    "hi": ("haan", "theek hai", "ji", "achha", "sahi", "bilkul"),
    "ta": ("seri", "aama", "saringa", "aamanga"),
    "te": ("avunu", "sare", "alage", "avunandi"),
    "kn": ("howdu", "sari", "aytu", "sari sari"),
    "ml": ("athe", "sheri", "sari", "athe athe"),
    "bn": ("hyan", "thik", "accha", "thik ache"),
    "mr": ("ho", "barobar", "theek ahe", "achha"),
    "gu": ("ha", "barabar", "theek che", "saru"),
    "pa": ("haan", "haanji", "theek hai", "achha"),
    "or": ("haan", "thik achi", "aagyan", "achha"),
}

CONJUNCTIONS_BY_LANG: dict[str, frozenset[str]] = {
    "en": frozenset(
        {
            "and",
            "or",
            "but",
            "because",
            "so",
            "although",
            "though",
            "while",
            "when",
            "where",
            "if",
            "that",
            "since",
            "unless",
            "until",
            "yet",
            "whereas",
            "as",
            "after",
            "before",
        }
    ),
    "hi": frozenset(
        {
            "aur",
            "ya",
            "ki",
            "kyunki",
            "isliye",
            "lekin",
            "par",
            "magar",
            "kintu",
            "parantu",
            "taki",
            "yadi",
            "agar",
            "jab",
            "tab",
            "to",
            "और",
            "तथा",
            "या",
            "कि",
            "क्योंकि",
            "इसलिए",
            "लेकिन",
            "पर",
            "मगर",
            "किन्तु",
            "परन्तु",
            "ताकि",
            "यदि",
            "अगर",
            "जब",
            "तब",
            "तो",
        }
    ),
    "ta": frozenset(
        {
            "matrum",
            "aanaal",
            "aanal",
            "alladhu",
            "yenendral",
            "anal",
            "மற்றும்",
            "ஆனால்",
            "அல்லது",
            "ஏனென்றால்",
        }
    ),
    "te": frozenset(
        {
            "mariyu",
            "kaani",
            "kani",
            "leda",
            "endukante",
            "మరియు",
            "కానీ",
            "లేదా",
            "ఎందుకంటే",
        }
    ),
    "kn": frozenset(
        {
            "mattu",
            "aadare",
            "athava",
            "yaakendare",
            "ಮತ್ತು",
            "ಆದರೆ",
            "ಅಥವಾ",
            "ಯಾಕೆಂದರೆ",
        }
    ),
    "ml": frozenset(
        {
            "ennal",
            "athava",
            "karanam",
            "matrum",
            "എന്നാൽ",
            "അഥവാ",
            "കാരണം",
        }
    ),
    "bn": frozenset(
        {
            "ebong",
            "kintu",
            "ba",
            "karon",
            "aar",
            "এবং",
            "কিন্তু",
            "বা",
            "কারণ",
            "আর",
        }
    ),
    "mr": frozenset(
        {
            "ani",
            "pan",
            "kinva",
            "karan",
            "आणि",
            "पण",
            "किंवा",
            "कारण",
        }
    ),
    "gu": frozenset(
        {
            "ane",
            "pan",
            "athva",
            "karan ke",
            "અને",
            "પણ",
            "અથવા",
            "કારણ કે",
        }
    ),
    "pa": frozenset(
        {
            "te",
            "par",
            "ja",
            "kyonki",
            "ate",
            "ਅਤੇ",
            "ਪਰ",
            "ਜਾਂ",
            "ਕਿਉਂਕਿ",
            "ਤੇ",
        }
    ),
    "or": frozenset(
        {
            "o",
            "ebang",
            "kintu",
            "ba",
            "କିନ୍ତୁ",
            "ଏବଂ",
            "ଓ",
            "ବା",
        }
    ),
}

ALL_CONJUNCTIONS: frozenset[str] = frozenset.union(*CONJUNCTIONS_BY_LANG.values())

TERMINAL_PARTICLES_BY_LANG: dict[str, frozenset[str]] = {
    "en": frozenset({"right", "okay", "ok", "yeah", "see", "though"}),
    "hi": frozenset(
        {
            "na",
            "yaar",
            "bhai",
            "hai",
            "hain",
            "tha",
            "thi",
            "the",
            "hoga",
            "hogi",
            "honge",
            "ji",
            "न",
            "ना",
            "यार",
            "भाई",
            "है",
            "हैं",
            "था",
            "थी",
            "थे",
            "होगा",
            "होगी",
            "होंगे",
            "जी",
        }
    ),
    "ta": frozenset(
        {
            "dhaan",
            "la",
            "illa",
            "pa",
            "nga",
            "haan",
            "தான்",
            "ல",
            "இல்ல",
            "பா",
            "ங்க",
        }
    ),
    "te": frozenset(
        {
            "kada",
            "andi",
            "ga",
            "le",
            "kadha",
            "కదా",
            "అండి",
            "గా",
            "లే",
        }
    ),
    "kn": frozenset(
        {
            "alva",
            "ri",
            "kanri",
            "kane",
            "alla",
            "ಅಲ್ವಾ",
            "ರೀ",
            "ಕನ್ರೀ",
        }
    ),
    "ml": frozenset(
        {
            "allo",
            "ille",
            "aane",
            "alle",
            "അല്ലേ",
            "ആണ്",
            "അല്ലോ",
        }
    ),
    "bn": frozenset(
        {
            "na",
            "to",
            "tai",
            "tai na",
            "না",
            "তো",
            "তাই",
            "তাই না",
        }
    ),
    "mr": frozenset(
        {
            "na",
            "ka",
            "ahe",
            "aahe",
            "barobar",
            "ना",
            "का",
            "आहे",
        }
    ),
    "gu": frozenset(
        {
            "ne",
            "che",
            "saaru",
            "ને",
            "છે",
        }
    ),
    "pa": frozenset(
        {
            "na",
            "si",
            "hai",
            "hanji",
            "ਨਾ",
            "ਸੀ",
            "ਹੈ",
        }
    ),
    "or": frozenset(
        {
            "na",
            "achhi",
            "aagyan",
            "ନା",
            "ଅଛି",
        }
    ),
}

ALL_TERMINAL_PARTICLES: frozenset[str] = frozenset.union(*TERMINAL_PARTICLES_BY_LANG.values())

_TRAILING_MID_CLAUSE_PUNCT = re.compile(r"[,…]+$|\.{3,}$")
_CLAUSE_END_PUNCT = frozenset({".", "!", "?", "।", "॥", ";"})
_TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


def detect_language(text: str, default: str = "en") -> str:
    """Infer language code from script characters or fallback to default."""
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return "hi"
        if 0x0980 <= cp <= 0x09FF:
            return "bn"
        if 0x0A00 <= cp <= 0x0A7F:
            return "pa"
        if 0x0A80 <= cp <= 0x0AFF:
            return "gu"
        if 0x0B00 <= cp <= 0x0B7F:
            return "or"
        if 0x0B80 <= cp <= 0x0BFF:
            return "ta"
        if 0x0C00 <= cp <= 0x0C7F:
            return "te"
        if 0x0C80 <= cp <= 0x0CFF:
            return "kn"
        if 0x0D00 <= cp <= 0x0D7F:
            return "ml"
    return default


def is_mid_clause(text: str, lang: str | None = None) -> bool:
    """Return True if text ends with a comma, ellipsis, or trailing conjunction."""
    stripped = text.strip()
    if not stripped:
        return False

    if _TRAILING_MID_CLAUSE_PUNCT.search(stripped) is not None:
        return True

    tokens = _TOKEN_RE.findall(stripped.casefold())
    if not tokens:
        return False

    last_token = tokens[-1]
    if lang and lang in CONJUNCTIONS_BY_LANG:
        if last_token in CONJUNCTIONS_BY_LANG[lang]:
            return True
    return last_token in ALL_CONJUNCTIONS


def is_clause_complete(text: str, lang: str | None = None) -> bool:
    """Return True if text ends with clause-ending punctuation or a terminal particle."""
    stripped = text.strip()
    if not stripped:
        return False

    if is_mid_clause(stripped, lang=lang):
        return False

    if stripped[-1] in _CLAUSE_END_PUNCT:
        return True

    tokens = _TOKEN_RE.findall(stripped.casefold())
    if not tokens:
        return False

    if len(tokens) >= 2:
        last_two = f"{tokens[-2]} {tokens[-1]}"
        if lang and lang in TERMINAL_PARTICLES_BY_LANG:
            if last_two in TERMINAL_PARTICLES_BY_LANG[lang]:
                return True
        if last_two in ALL_TERMINAL_PARTICLES:
            return True

    last_token = tokens[-1]
    if lang and lang in TERMINAL_PARTICLES_BY_LANG:
        if last_token in TERMINAL_PARTICLES_BY_LANG[lang]:
            return True
    return last_token in ALL_TERMINAL_PARTICLES


# =============================================================================
# Pure Deterministic PRNG (INV AST Purity)
# =============================================================================


class _DeterministicRng:
    """Pure seeded 64-bit LCG random number generator avoiding forbidden random module."""

    def __init__(self, seed: int = 42) -> None:
        self._state: int = (seed & 0xFFFFFFFFFFFFFFFF) or 1

    def next_int(self, n: int) -> int:
        """Return integer in range [0, n)."""
        if n <= 1:
            return 0
        # 64-bit Linear Congruential Generator (Knuth MMIX)
        self._state = (self._state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self._state >> 33) % n

    def choice(self, seq: Sequence[str]) -> str:
        """Deterministically choose an element from a sequence."""
        if not seq:
            raise ValueError("Sequence cannot be empty")
        idx = self.next_int(len(seq))
        return seq[idx]


# =============================================================================
# BackchannelPredictor
# =============================================================================


class BackchannelPredictor:
    """Predicts agent backchannel timing and selects clips (§9, §10, INV-B1)."""

    def __init__(
        self,
        tuning: Tuning | None = None,
        seed: int = 42,
        default_lang: str = "en",
        clips_by_lang: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._tuning: Tuning = tuning or Tuning()
        self._seed: int = seed
        self._rng: _DeterministicRng = _DeterministicRng(seed)
        self._default_lang: str = default_lang

        if clips_by_lang is not None:
            self._clips_by_lang: dict[str, tuple[str, ...]] = {
                k: tuple(v) for k, v in clips_by_lang.items()
            }
        else:
            self._clips_by_lang = dict(DEFAULT_CLIPS_BY_LANG)

        # Turn & pause tracking
        self._in_turn: bool = False
        self._turn_start_ns: int | None = None
        self._vad_offset_ns: int | None = None
        self._pause_fired: bool = False

        # Prosodic energy tracking
        self._falling_energy_frames: int = 0
        self._prev_energy_ema: float | None = None

        # Lexical tracking
        self._latest_text: str = ""
        self._latest_lang: str | None = None

        # Playout & FSM tracking
        self._clip_in_flight: bool = False
        self._fsm_state: FsmState = FsmState.LISTENING

        # Pacing memory (INV-B1)
        self._last_fire_ns: int | None = None
        self._fire_history: deque[int] = deque()
        self._recent_clips: deque[str] = deque(maxlen=2)

    @property
    def is_clip_in_flight(self) -> bool:
        """Return True if a backchannel clip is currently playing."""
        return self._clip_in_flight

    @property
    def fsm_state(self) -> FsmState:
        """Return current FSM state."""
        return self._fsm_state

    @property
    def in_turn(self) -> bool:
        """Return True if user speech turn is currently active."""
        return self._in_turn

    def notify_clip_started(self) -> None:
        """Mark a backchannel clip as playing in flight."""
        self._clip_in_flight = True

    def notify_clip_done(self) -> None:
        """Mark clip playout completed."""
        self._clip_in_flight = False

    def set_fsm_state(self, state: FsmState) -> None:
        """Update tracked FSM state."""
        self._fsm_state = state

    def reset_turn(self) -> None:
        """Reset turn-level state when a user turn concludes or transitions."""
        self._in_turn = False
        self._turn_start_ns = None
        self._vad_offset_ns = None
        self._pause_fired = False
        self._falling_energy_frames = 0
        self._prev_energy_ema = None
        self._latest_text = ""
        self._latest_lang = None

    def reset(self) -> None:
        """Reset predictor completely including pacing and clip memory."""
        self.reset_turn()
        self._last_fire_ns = None
        self._fire_history.clear()
        self._recent_clips.clear()
        self._clip_in_flight = False
        self._fsm_state = FsmState.LISTENING
        self._rng = _DeterministicRng(self._seed)

    def _select_clip(self, lang: str) -> str:
        """Select a backchannel clip without repeating the last 2 triggers."""
        pool = (
            self._clips_by_lang.get(lang)
            or self._clips_by_lang.get(self._default_lang)
            or DEFAULT_CLIPS_BY_LANG.get("en", ("yeah", "okay", "right"))
        )

        candidates = [c for c in pool if c not in self._recent_clips]
        if not candidates:
            candidates = [c for c in pool if not self._recent_clips or c != self._recent_clips[-1]]
        if not candidates:
            candidates = list(pool)

        chosen = self._rng.choice(candidates)
        self._recent_clips.append(chosen)
        return chosen

    def update(
        self,
        now_ns: int,
        vad_state: bool | Any,
        partial: SttPartial | SttEvent | str | None = None,
        energy_ema: float | None = None,
        *,
        fsm_state: FsmState = FsmState.LISTENING,
        clip_in_flight: bool | None = None,
    ) -> BackchannelDue | None:
        """Evaluate acoustic, lexical, and timing features to decide backchannel trigger."""
        # 1. Extract vad boolean
        if isinstance(vad_state, bool):
            is_speech = vad_state
        elif hasattr(vad_state, "is_speech"):
            is_speech = bool(vad_state.is_speech)
        else:
            is_speech = bool(vad_state)

        # 2. Process STT partial if provided
        if partial is not None:
            if isinstance(partial, (SttPartial, SttEvent)):
                p_text = partial.text
                p_lang = partial.lang
            else:
                p_text = str(partial)
                p_lang = None

            if p_text.strip():
                self._latest_text = p_text.strip()
                self._latest_lang = p_lang or detect_language(p_text, default=self._default_lang)

        # 3. Handle speech onset vs pause transitions
        if is_speech:
            if not self._in_turn:
                self._in_turn = True
                self._turn_start_ns = now_ns
            self._vad_offset_ns = None
            self._pause_fired = False
            self._clip_in_flight = False
            self._falling_energy_frames = 0
            self._prev_energy_ema = None
            return None

        # User is not speaking (vad_state is False)
        if not self._in_turn or self._turn_start_ns is None:
            return None

        if self._vad_offset_ns is None:
            self._vad_offset_ns = now_ns
            self._pause_fired = False

        if self._pause_fired:
            return None

        # 4. Energy EMA slope tracking
        if energy_ema is not None:
            if self._prev_energy_ema is not None and energy_ema < self._prev_energy_ema:
                self._falling_energy_frames += 1
            elif self._prev_energy_ema is not None and energy_ema >= self._prev_energy_ema:
                self._falling_energy_frames = 0
            self._prev_energy_ema = energy_ema

        # 5. State gates: FSM must be LISTENING, no clip in flight
        if fsm_state != FsmState.LISTENING or self._fsm_state != FsmState.LISTENING:
            return None

        if clip_in_flight is True:
            return None
        if clip_in_flight is None and self._clip_in_flight:
            return None

        # 6. Check turn duration threshold (INV-B1, §9)
        turn_duration_ms = (now_ns - self._turn_start_ns) // 1_000_000
        if turn_duration_ms < self._tuning.bc_min_user_turn_ms:
            return None

        # 7. Check pause window (strictly measured from VAD offset)
        pause_ms = (now_ns - self._vad_offset_ns) // 1_000_000
        if not (self._tuning.bc_pause_min_ms <= pause_ms <= self._tuning.bc_pause_max_ms):
            return None

        # 8. Check clause completion / conjunction blacklist / energy fall
        text = self._latest_text
        lang = self._latest_lang or self._default_lang

        if text and is_mid_clause(text, lang=lang):
            return None

        clause_complete = bool(text and is_clause_complete(text, lang=lang))
        energy_fell = self._falling_energy_frames >= self._tuning.bc_energy_fall_frames

        if not (clause_complete or energy_fell):
            return None

        # 9. Cooldown gap check (INV-B1: >= 4 s)
        if self._last_fire_ns is not None:
            gap_ms = (now_ns - self._last_fire_ns) // 1_000_000
            if gap_ms < self._tuning.bc_min_gap_ms:
                return None

        # 10. Sliding window rate cap check (INV-B1: <= 6 / min)
        window_start_ns = now_ns - 60_000_000_000
        while self._fire_history and self._fire_history[0] <= window_start_ns:
            self._fire_history.popleft()

        if len(self._fire_history) >= self._tuning.bc_max_per_min:
            return None

        # 11. All conditions hold: select clip and emit trigger
        clip = self._select_clip(lang)
        self._last_fire_ns = now_ns
        self._fire_history.append(now_ns)
        self._pause_fired = True
        self._clip_in_flight = True

        return BackchannelDue(clip=clip, lang=lang)
