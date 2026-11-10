"""Unit tests for multi-script Indic and English lexicons and matcher.

Covers:
- All 11 YAML lexicons (en + 10 Indic: hi, ta, te, kn, ml, bn, mr, gu, pa, or)
- Table-driven tests (>= 120 cases) covering backchannels, stops, contrasts, questions
- Failure modes:
  * 'nahi'/'ruko'/'wait' treated as backchannel (must be stop/interruption)
  * 'haan lekin ...' classified as pure backchannel (must detect contrast)
  * elongations ('haaaan'), punctuation, dandas, casing
  * native-script vs romanized coverage
  * fuzzy match false positives on real content words
  * code-switched partial matching across languages
  * YAML load purity & performance (< 50 ms, no I/O at call time)
"""

from __future__ import annotations

import importlib.resources
import time
from dataclasses import FrozenInstanceError, dataclass
from unittest.mock import patch

import pytest
import yaml  # type: ignore[import-untyped]

from overlap.core.lexicon import Analysis, Lexicon, analyze

# =============================================================================
# Lexicon Files & Metadata Tests
# =============================================================================

REQUIRED_LANGUAGES = (
    "en",
    "hi",
    "ta",
    "te",
    "kn",
    "ml",
    "bn",
    "mr",
    "gu",
    "pa",
    "or",
)


def test_required_languages_present() -> None:
    """All 11 required language YAML files must exist in overlap/core/lexicons/."""
    lex_dir = importlib.resources.files("overlap.core").joinpath("lexicons")
    for lang in REQUIRED_LANGUAGES:
        yaml_path = lex_dir.joinpath(f"{lang}.yaml")
        assert yaml_path.is_file(), f"Missing lexicon file: {lang}.yaml"


def test_lexicon_header_and_fields() -> None:
    """Every lexicon YAML must have header comment and required fields."""
    lex_dir = importlib.resources.files("overlap.core").joinpath("lexicons")
    for lang in REQUIRED_LANGUAGES:
        yaml_resource = lex_dir.joinpath(f"{lang}.yaml")
        text = yaml_resource.read_text(encoding="utf-8")
        assert text.startswith("# needs native-speaker review"), (
            f"{lang}.yaml missing header: '# needs native-speaker review'"
        )

        data = yaml.safe_load(text)
        assert isinstance(data, dict), f"{lang}.yaml content must be a dict"
        for field in ("backchannel", "stop", "contrast", "question"):
            assert field in data, f"{lang}.yaml missing required field: {field}"
            assert isinstance(data[field], list), f"{lang}.yaml {field} must be a list"
            assert len(data[field]) > 0, f"{lang}.yaml {field} must not be empty"


def test_native_vs_romanized_coverage() -> None:
    """Indic language YAMLs must contain both native-script and romanized variants."""
    lex_dir = importlib.resources.files("overlap.core").joinpath("lexicons")
    indic_langs = [lng for lng in REQUIRED_LANGUAGES if lng != "en"]
    for lang in indic_langs:
        yaml_resource = lex_dir.joinpath(f"{lang}.yaml")
        data = yaml.safe_load(yaml_resource.read_text(encoding="utf-8"))
        for field in ("backchannel", "stop", "contrast", "question"):
            entries = data[field]
            has_native = any(not entry.isascii() for entry in entries)
            has_romanized = any(entry.isascii() for entry in entries)
            assert has_native, f"{lang}.yaml {field} lacks native script entries"
            assert has_romanized, f"{lang}.yaml {field} lacks romanized entries"


# =============================================================================
# Performance & Invariant Tests
# =============================================================================


def test_lexicon_load_time_under_budget() -> None:
    """Lexicon loading must complete in < 50 ms."""
    t0 = time.perf_counter()
    lexicon = Lexicon.load()
    dur_ms = (time.perf_counter() - t0) * 1000.0
    assert dur_ms < 50.0, f"Lexicon load took {dur_ms:.2f} ms (budget 50 ms)"
    assert len(lexicon.languages) >= 11


def test_zero_file_io_at_call_time() -> None:
    """Lexicon.analyze must perform zero file I/O at call time."""
    lexicon = Lexicon.load()

    with patch("builtins.open") as mock_open:
        res = lexicon.analyze("haan theek hai lekin ruko")
        assert res.stop_hit is True
        assert mock_open.call_count == 0


def test_analysis_dataclass_frozen_slots() -> None:
    """Analysis must be a frozen dataclass with slots."""
    res = analyze("haan")
    assert isinstance(res, Analysis)
    with pytest.raises(FrozenInstanceError):
        res.stop_hit = True  # type: ignore[misc]


# =============================================================================
# Table-Driven Test Suite (>= 120 cases)
# =============================================================================


@dataclass(frozen=True)
class Case:
    text: str
    tokens: tuple[str, ...]
    bc_tokens: tuple[str, ...]
    stop: bool
    contrast: bool
    question: bool
    content_tokens: tuple[str, ...]
    lang_hint: str | None = None
    desc: str = ""


TABLE_CASES: list[Case] = [
    # -------------------------------------------------------------------------
    # 10 cases explicitly mandated by done_when
    # -------------------------------------------------------------------------
    Case(
        "haan haan",
        ("haan", "haan"),
        ("haan", "haan"),
        False,
        False,
        False,
        (),
        "hi",
        "done_when #1",
    ),
    Case("ji", ("ji",), ("ji",), False, False, False, (), "hi", "done_when #2"),
    Case(
        "achha theek hai",
        ("achha", "theek", "hai"),
        ("achha", "theek", "hai"),
        False,
        False,
        False,
        (),
        "hi",
        "done_when #3",
    ),
    Case("seri", ("seri",), ("seri",), False, False, False, (), "ta", "done_when #4"),
    Case("howdu", ("howdu",), ("howdu",), False, False, False, (), "kn", "done_when #5"),
    Case(
        "haan lekin ruko",
        ("haan", "lekin", "ruko"),
        ("haan",),
        True,
        True,
        False,
        (),
        "hi",
        "done_when #6",
    ),
    Case("ek minute", ("ek", "minute"), (), True, False, False, (), "hi", "done_when #7"),
    Case("kya?", ("kya",), (), False, False, True, (), "hi", "done_when #8"),
    Case("nahi nahi", ("nahi", "nahi"), (), True, False, False, (), "hi", "done_when #9"),
    Case(
        "ok but wait", ("ok", "but", "wait"), ("ok",), True, True, False, (), "en", "done_when #10"
    ),
    # -------------------------------------------------------------------------
    # Hindi (hi) — Romanized & Devanagari
    # -------------------------------------------------------------------------
    Case("हाँ", ("हाँ",), ("हाँ",), False, False, False, (), "hi"),
    Case("जी", ("जी",), ("जी",), False, False, False, (), "hi"),
    Case("हाँ जी", ("हाँ", "जी"), ("हाँ", "जी"), False, False, False, (), "hi"),
    Case("हाँजी", ("हाँजी",), ("हाँजी",), False, False, False, (), "hi"),
    Case("अच्छा", ("अच्छा",), ("अच्छा",), False, False, False, (), "hi"),
    Case("ठीक है", ("ठीक", "है"), ("ठीक", "है"), False, False, False, (), "hi"),
    Case("हूँ", ("हूँ",), ("हूँ",), False, False, False, (), "hi"),
    Case("हम्म", ("हम्म",), ("हम्म",), False, False, False, (), "hi"),
    Case("रुको", ("रुको",), (), True, False, False, (), "hi"),
    Case("नहीं", ("नहीं",), (), True, False, False, (), "hi"),
    Case("बस", ("बस",), (), True, False, False, (), "hi"),
    Case("एक मिनट", ("एक", "मिनट"), (), True, False, False, (), "hi"),
    Case("लेकिन", ("लेकिन",), (), False, True, False, (), "hi"),
    Case("मगर", ("मगर",), (), False, True, False, (), "hi"),
    Case("परन्तु", ("परन्तु",), (), False, True, False, (), "hi"),
    Case("क्या", ("क्या",), (), False, False, True, (), "hi"),
    Case("क्यों", ("क्यों",), (), False, False, True, (), "hi"),
    Case("कैसे", ("कैसे",), (), False, False, True, (), "hi"),
    Case("कब", ("कब",), (), False, False, True, (), "hi"),
    Case("हाँजी बिल्कुल", ("हाँजी", "बिल्कुल"), ("हाँजी", "बिल्कुल"), False, False, False, (), "hi"),
    Case("haanji", ("haanji",), ("haanji",), False, False, False, (), "hi"),
    Case("theek hai", ("theek", "hai"), ("theek", "hai"), False, False, False, (), "hi"),
    Case("accha", ("accha",), ("accha",), False, False, False, (), "hi"),
    Case("thik hai", ("thik", "hai"), ("thik", "hai"), False, False, False, (), "hi"),
    Case("ruko ruko", ("ruko", "ruko"), (), True, False, False, (), "hi"),
    Case("ek minute ruko", ("ek", "minute", "ruko"), (), True, False, False, (), "hi"),
    Case("lekin ruko", ("lekin", "ruko"), (), True, True, False, (), "hi"),
    Case("kyun?", ("kyun",), (), False, False, True, (), "hi"),
    Case("kahan", ("kahan",), (), False, False, True, (), "hi"),
    Case("kaise", ("kaise",), (), False, False, True, (), "hi"),
    # -------------------------------------------------------------------------
    # Tamil (ta) — Romanized & Tamil script
    # -------------------------------------------------------------------------
    Case("சரி", ("சரி",), ("சரி",), False, False, False, (), "ta"),
    Case("ஆமா", ("ஆமா",), ("ஆமா",), False, False, False, (), "ta"),
    Case("ஆமாம்", ("ஆமாம்",), ("ஆமாம்",), False, False, False, (), "ta"),
    Case("ஆமாங்க", ("ஆமாங்க",), ("ஆமாங்க",), False, False, False, (), "ta"),
    Case("நில்லு", ("நில்லு",), (), True, False, False, (), "ta"),
    Case("நில்லுங்க", ("நில்லுங்க",), (), True, False, False, (), "ta"),
    Case("வேண்டாம்", ("வேண்டாம்",), (), True, False, False, (), "ta"),
    Case("ஒரு நிமிஷம்", ("ஒரு", "நிமிஷம்"), (), True, False, False, (), "ta"),
    Case("ஆனால்", ("ஆனால்",), (), False, True, False, (), "ta"),
    Case("ஆனா", ("ஆனா",), (), False, True, False, (), "ta"),
    Case("என்ன", ("என்ன",), (), False, False, True, (), "ta"),
    Case("எப்படி", ("எப்படி",), (), False, False, True, (), "ta"),
    Case("எங்க", ("எங்க",), (), False, False, True, (), "ta"),
    Case("aama", ("aama",), ("aama",), False, False, False, (), "ta"),
    Case("aamam", ("aamam",), ("aamam",), False, False, False, (), "ta"),
    Case("nillu", ("nillu",), (), True, False, False, (), "ta"),
    Case("oru nimisham", ("oru", "nimisham"), (), True, False, False, (), "ta"),
    Case("aanaal", ("aanaal",), (), False, True, False, (), "ta"),
    Case("enna?", ("enna",), (), False, False, True, (), "ta"),
    Case("edhuku", ("edhuku",), (), False, False, True, (), "ta"),
    # -------------------------------------------------------------------------
    # Kannada (kn) — Romanized & Kannada script
    # -------------------------------------------------------------------------
    Case("ಹೌದು", ("ಹೌದು",), ("ಹೌದು",), False, False, False, (), "kn"),
    Case("ಸರಿ", ("ಸರಿ",), ("ಸರಿ",), False, False, False, (), "kn"),
    Case("ಆಯ್ತು", ("ಆಯ್ತು",), ("ಆಯ್ತು",), False, False, False, (), "kn"),
    Case("ನಿಲ್ಲಿಸಿ", ("ನಿಲ್ಲಿಸಿ",), (), True, False, False, (), "kn"),
    Case("ಬೇಡ", ("ಬೇಡ",), (), True, False, False, (), "kn"),
    Case("ಒಂದು ನಿಮಿಷ", ("ಒಂದು", "ನಿಮಿಷ"), (), True, False, False, (), "kn"),
    Case("ಆದರೆ", ("ಆದರೆ",), (), False, True, False, (), "kn"),
    Case("ಆದ್ರೆ", ("ಆದ್ರೆ",), (), False, True, False, (), "kn"),
    Case("ಏನು", ("ಏನು",), (), False, False, True, (), "kn"),
    Case("ಯಾಕೆ", ("ಯಾಕೆ",), (), False, False, True, (), "kn"),
    Case("haudu", ("haudu",), ("haudu",), False, False, False, (), "kn"),
    Case("aithu", ("aithu",), ("aithu",), False, False, False, (), "kn"),
    Case("nillisi", ("nillisi",), (), True, False, False, (), "kn"),
    Case("ondu nimisha", ("ondu", "nimisha"), (), True, False, False, (), "kn"),
    Case("aadre", ("aadre",), (), False, True, False, (), "kn"),
    Case("yenu?", ("yenu",), (), False, False, True, (), "kn"),
    Case("yaake", ("yaake",), (), False, False, True, (), "kn"),
    # -------------------------------------------------------------------------
    # Telugu (te) — Romanized & Telugu script
    # -------------------------------------------------------------------------
    Case("అవును", ("అవును",), ("అవును",), False, False, False, (), "te"),
    Case("సరే", ("సరే",), ("సరే",), False, False, False, (), "te"),
    Case("అలాగే", ("అలాగే",), ("అలాగే",), False, False, False, (), "te"),
    Case("ఆగండి", ("ఆగండి",), (), True, False, False, (), "te"),
    Case("వద్దు", ("వద్దు",), (), True, False, False, (), "te"),
    Case("ఒక నిమిషం", ("ఒక", "నిమిషం"), (), True, False, False, (), "te"),
    Case("కానీ", ("కానీ",), (), False, True, False, (), "te"),
    Case("ఏంటి", ("ఏంటి",), (), False, False, True, (), "te"),
    Case("ఎందుకు", ("ఎందుకు",), (), False, False, True, (), "te"),
    Case("avunu", ("avunu",), ("avunu",), False, False, False, (), "te"),
    Case("sare", ("sare",), ("sare",), False, False, False, (), "te"),
    Case("alage", ("alage",), ("alage",), False, False, False, (), "te"),
    Case("aagandi", ("aagandi",), (), True, False, False, (), "te"),
    Case("oka nimisham", ("oka", "nimisham"), (), True, False, False, (), "te"),
    Case("kaani", ("kaani",), (), False, True, False, (), "te"),
    Case("enti?", ("enti",), (), False, False, True, (), "te"),
    Case("enduku", ("enduku",), (), False, False, True, (), "te"),
    # -------------------------------------------------------------------------
    # Malayalam (ml) — Romanized & Malayalam script
    # -------------------------------------------------------------------------
    Case("അതെ", ("അതെ",), ("അതെ",), False, False, False, (), "ml"),
    Case("ശരി", ("ശരി",), ("ശരി",), False, False, False, (), "ml"),
    Case("നില്‍ക്കൂ", ("നില്‍ക്കൂ",), (), True, False, False, (), "ml"),
    Case("വേണ്ട", ("വേണ്ട",), (), True, False, False, (), "ml"),
    Case("ഒരു നിമിഷം", ("ഒരു", "നിമിഷം"), (), True, False, False, (), "ml"),
    Case("പക്ഷേ", ("പക്ഷേ",), (), False, True, False, (), "ml"),
    Case("എന്താ", ("എന്താ",), (), False, False, True, (), "ml"),
    Case("എങ്ങനെ", ("എങ്ങനെ",), (), False, False, True, (), "ml"),
    Case("athe", ("athe",), ("athe",), False, False, False, (), "ml"),
    Case("sheri", ("sheri",), ("sheri",), False, False, False, (), "ml"),
    Case("nilkku", ("nilkku",), (), True, False, False, (), "ml"),
    Case("venda", ("venda",), (), True, False, False, (), "ml"),
    Case("pakshe", ("pakshe",), (), False, True, False, (), "ml"),
    Case("entha?", ("entha",), (), False, False, True, (), "ml"),
    # -------------------------------------------------------------------------
    # Bengali (bn) — Romanized & Bengali script
    # -------------------------------------------------------------------------
    Case("হ্যাঁ", ("হ্যাঁ",), ("হ্যাঁ",), False, False, False, (), "bn"),
    Case("আচ্ছা", ("আচ্ছা",), ("আচ্ছা",), False, False, False, (), "bn"),
    Case("ঠিক আছে", ("ঠিক", "আছে"), ("ঠিক", "আছে"), False, False, False, (), "bn"),
    Case("থামো", ("থামো",), (), True, False, False, (), "bn"),
    Case("এক মিনিট", ("এক", "মিনিট"), (), True, False, False, (), "bn"),
    Case("কিন্তু", ("কিন্তু",), (), False, True, False, (), "bn"),
    Case("কী", ("কী",), (), False, False, True, (), "bn"),
    Case("কেন", ("কেন",), (), False, False, True, (), "bn"),
    Case("hyan", ("hyan",), ("hyan",), False, False, False, (), "bn"),
    Case("thik ache", ("thik", "ache"), ("thik", "ache"), False, False, False, (), "bn"),
    Case("thamo", ("thamo",), (), True, False, False, (), "bn"),
    Case("kintu", ("kintu",), (), False, True, False, (), "bn"),
    Case("ki?", ("ki",), (), False, False, True, (), "bn"),
    # -------------------------------------------------------------------------
    # Marathi (mr) — Romanized & Marathi Devanagari
    # -------------------------------------------------------------------------
    Case("हो", ("हो",), ("हो",), False, False, False, (), "mr"),
    Case("होय", ("होय",), ("होय",), False, False, False, (), "mr"),
    Case("बरोबर", ("बरोबर",), ("बरोबर",), False, False, False, (), "mr"),
    Case("थांबा", ("थांबा",), (), True, False, False, (), "mr"),
    Case("नको", ("नको",), (), True, False, False, (), "mr"),
    Case("पण", ("पण",), (), False, True, False, (), "mr"),
    Case("काय", ("काय",), (), False, False, True, (), "mr"),
    Case("कुठे", ("कुठे",), (), False, False, True, (), "mr"),
    Case("barobar", ("barobar",), ("barobar",), False, False, False, (), "mr"),
    Case("theek ahe", ("theek", "ahe"), ("theek", "ahe"), False, False, False, (), "mr"),
    Case("thamba", ("thamba",), (), True, False, False, (), "mr"),
    Case("pan", ("pan",), (), False, True, False, (), "mr"),
    Case("kay?", ("kay",), (), False, False, True, (), "mr"),
    # -------------------------------------------------------------------------
    # Gujarati (gu) — Romanized & Gujarati script
    # -------------------------------------------------------------------------
    Case("હા", ("હા",), ("હા",), False, False, False, (), "gu"),
    Case("સારું", ("સારું",), ("સારું",), False, False, False, (), "gu"),
    Case("બરાબર", ("બરાબર",), ("બરાબર",), False, False, False, (), "gu"),
    Case("ઉભા રહો", ("ઉભા", "રહો"), (), True, False, False, (), "gu"),
    Case("નથી", ("નથી",), (), True, False, False, (), "gu"),
    Case("પણ", ("પણ",), (), False, True, False, (), "gu"),
    Case("શું", ("શું",), (), False, False, True, (), "gu"),
    Case("કેમ", ("કેમ",), (), False, False, True, (), "gu"),
    Case("saru", ("saru",), ("saru",), False, False, False, (), "gu"),
    Case("theek che", ("theek", "che"), ("theek", "che"), False, False, False, (), "gu"),
    Case("ubha raho", ("ubha", "raho"), (), True, False, False, (), "gu"),
    Case("shu?", ("shu",), (), False, False, True, (), "gu"),
    # -------------------------------------------------------------------------
    # Punjabi (pa) — Romanized & Gurmukhi script
    # -------------------------------------------------------------------------
    Case("ਹਾਂ", ("ਹਾਂ",), ("ਹਾਂ",), False, False, False, (), "pa"),
    Case("ਹਾਂਜੀ", ("ਹਾਂਜੀ",), ("ਹਾਂਜੀ",), False, False, False, (), "pa"),
    Case("ਠੀਕ ਆ", ("ਠੀਕ", "ਆ"), ("ਠੀਕ", "ਆ"), False, False, False, (), "pa"),
    Case("ਅੱਛਾ", ("ਅੱਛਾ",), ("ਅੱਛਾ",), False, False, False, (), "pa"),
    Case("ਖਲੋ", ("ਖਲੋ",), (), True, False, False, (), "pa"),
    Case("ਪਰ", ("ਪਰ",), (), False, True, False, (), "pa"),
    Case("ਕੀ", ("ਕੀ",), (), False, False, True, (), "pa"),
    Case("ਕਿਵੇਂ", ("ਕਿਵੇਂ",), (), False, False, True, (), "pa"),
    Case("theek aa", ("theek", "aa"), ("theek", "aa"), False, False, False, (), "pa"),
    Case("khalo", ("khalo",), (), True, False, False, (), "pa"),
    Case("kiven?", ("kiven",), (), False, False, True, (), "pa"),
    # -------------------------------------------------------------------------
    # Odia (or) — Romanized & Odia script
    # -------------------------------------------------------------------------
    Case("ହଁ", ("ହଁ",), ("ହଁ",), False, False, False, (), "or"),
    Case("ଆଜ୍ଞା", ("ଆଜ୍ଞା",), ("ଆଜ୍ଞା",), False, False, False, (), "or"),
    Case("ଠିକ ଅଛି", ("ଠିକ", "ଅଛି"), ("ଠିକ", "ଅଛି"), False, False, False, (), "or"),
    Case("ରୁହ", ("ରୁହ",), (), True, False, False, (), "or"),
    Case("ଗୋଟେ ମିନିଟ", ("ଗୋଟେ", "ମିନିଟ"), (), True, False, False, (), "or"),
    Case("କିନ୍ତୁ", ("କିନ୍ତୁ",), (), False, True, False, (), "or"),
    Case("କଣ", ("କଣ",), (), False, False, True, (), "or"),
    Case("କାହିଁକି", ("କାହିଁକି",), (), False, False, True, (), "or"),
    Case("aagyan", ("aagyan",), ("aagyan",), False, False, False, (), "or"),
    Case("thik achi", ("thik", "achi"), ("thik", "achi"), False, False, False, (), "or"),
    Case("ruha", ("ruha",), (), True, False, False, (), "or"),
    Case("kana?", ("kana",), (), False, False, True, (), "or"),
    # -------------------------------------------------------------------------
    # English (en)
    # -------------------------------------------------------------------------
    Case("yeah", ("yeah",), ("yeah",), False, False, False, (), "en"),
    Case("okay", ("okay",), ("okay",), False, False, False, (), "en"),
    Case("right", ("right",), ("right",), False, False, False, (), "en"),
    Case("mhm", ("mhm",), ("mhm",), False, False, False, (), "en"),
    Case("uh-huh", ("uh", "huh"), ("uh", "huh"), False, False, False, (), "en"),
    Case("got it", ("got", "it"), ("got", "it"), False, False, False, (), "en"),
    Case("sure", ("sure",), ("sure",), False, False, False, (), "en"),
    Case("cool", ("cool",), ("cool",), False, False, False, (), "en"),
    Case("wait", ("wait",), (), True, False, False, (), "en"),
    Case("stop", ("stop",), (), True, False, False, (), "en"),
    Case("hold on", ("hold", "on"), (), True, False, False, (), "en"),
    Case("hang on", ("hang", "on"), (), True, False, False, (), "en"),
    Case("one minute", ("one", "minute"), (), True, False, False, (), "en"),
    Case("one second", ("one", "second"), (), True, False, False, (), "en"),
    Case("but", ("but",), (), False, True, False, (), "en"),
    Case("however", ("however",), (), False, True, False, (), "en"),
    Case("what?", ("what",), (), False, False, True, (), "en"),
    Case("why", ("why",), (), False, False, True, (), "en"),
    Case("how", ("how",), (), False, False, True, (), "en"),
    # -------------------------------------------------------------------------
    # Code-switched utterances across languages
    # -------------------------------------------------------------------------
    Case("haan but wait", ("haan", "but", "wait"), ("haan",), True, True, False, ()),
    Case("seri but wait", ("seri", "but", "wait"), ("seri",), True, True, False, ()),
    Case("howdu but wait", ("howdu", "but", "wait"), ("howdu",), True, True, False, ()),
    Case("avunu but wait", ("avunu", "but", "wait"), ("avunu",), True, True, False, ()),
    Case("sheri but wait", ("sheri", "but", "wait"), ("sheri",), True, True, False, ()),
    Case(
        "accha let's go", ("accha", "let's", "go"), ("accha",), False, False, False, ("let's", "go")
    ),
    Case(
        "haan ji definitely",
        ("haan", "ji", "definitely"),
        ("haan", "ji"),
        False,
        False,
        False,
        ("definitely",),
    ),
    Case(
        "theek hai one second",
        ("theek", "hai", "one", "second"),
        ("theek", "hai"),
        True,
        False,
        False,
        (),
    ),
    Case("ok ruko ek minute", ("ok", "ruko", "ek", "minute"), ("ok",), True, False, False, ()),
    Case("seri one minute", ("seri", "one", "minute"), ("seri",), True, False, False, ()),
    # -------------------------------------------------------------------------
    # Elongation collapse & casing & punctuation/danda normalization
    # -------------------------------------------------------------------------
    Case(
        "haaaan", ("haan",), ("haan",), False, False, False, (), None, "elongation collapse to haan"
    ),
    Case("haaaaaan", ("haan",), ("haan",), False, False, False, (), None, "extreme elongation"),
    Case("hmmmm", ("hmm",), ("hmm",), False, False, False, (), None, "hmm elongation"),
    Case("hmmmmm", ("hmm",), ("hmm",), False, False, False, (), None, "extreme hmm"),
    Case("rukoooo", ("ruko",), (), True, False, False, (), None, "stop elongation"),
    Case("waaaait", ("wait",), (), True, False, False, (), None, "wait elongation"),
    Case("stopppp", ("stop",), (), True, False, False, (), None, "stop elongation"),
    Case("naaaahi", ("nahi",), (), True, False, False, (), None, "nahi elongation"),
    Case("achhaaaa", ("achha",), ("achha",), False, False, False, (), None, "achha elongation"),
    Case("HAAN", ("haan",), ("haan",), False, False, False, (), None, "uppercase"),
    Case("SeRi", ("seri",), ("seri",), False, False, False, (), None, "mixed case"),
    Case("HoWdU", ("howdu",), ("howdu",), False, False, False, (), None, "mixed case"),
    Case("Ek Minute", ("ek", "minute"), (), True, False, False, (), None, "title case"),
    Case(
        "haan, ji", ("haan", "ji"), ("haan", "ji"), False, False, False, (), None, "comma separated"
    ),
    Case(
        "haan... ji",
        ("haan", "ji"),
        ("haan", "ji"),
        False,
        False,
        False,
        (),
        None,
        "ellipsis separated",
    ),
    Case(
        "haan । ruko", ("haan", "ruko"), ("haan",), True, False, False, (), None, "danda separated"
    ),
    Case(
        "haan ॥ ruko",
        ("haan", "ruko"),
        ("haan",),
        True,
        False,
        False,
        (),
        None,
        "double danda separated",
    ),
    Case(
        "achha! theek hai.",
        ("achha", "theek", "hai"),
        ("achha", "theek", "hai"),
        False,
        False,
        False,
        (),
        None,
        "exclamation and period",
    ),
    Case(
        "weather?",
        ("weather",),
        (),
        False,
        False,
        True,
        ("weather",),
        None,
        "question mark on content word",
    ),
    Case(
        "kya haal hai",
        ("kya", "haal", "hai"),
        (),
        False,
        False,
        True,
        ("haal", "hai"),
        None,
        "question with content",
    ),
    # -------------------------------------------------------------------------
    # Content words & False Positive Protection
    # -------------------------------------------------------------------------
    Case(
        "weather is nice",
        ("weather", "is", "nice"),
        (),
        False,
        False,
        False,
        ("weather", "is", "nice"),
        None,
        "content utterance",
    ),
    Case(
        "cancel my flight",
        ("cancel", "my", "flight"),
        (),
        False,
        False,
        False,
        ("cancel", "my", "flight"),
        None,
        "content utterance",
    ),
    Case(
        "book a table tomorrow",
        ("book", "a", "table", "tomorrow"),
        (),
        False,
        False,
        False,
        ("book", "a", "table", "tomorrow"),
        None,
        "content utterance",
    ),
    Case(
        "delhi temperature",
        ("delhi", "temperature"),
        (),
        False,
        False,
        False,
        ("delhi", "temperature"),
        None,
        "content utterance",
    ),
    Case(
        "play music",
        ("play", "music"),
        (),
        False,
        False,
        False,
        ("play", "music"),
        None,
        "content utterance",
    ),
    Case(
        "tell me a joke",
        ("tell", "me", "a", "joke"),
        (),
        False,
        False,
        False,
        ("tell", "me", "a", "joke"),
        None,
        "content utterance",
    ),
    # -------------------------------------------------------------------------
    # Fuzzy Matching (romanized backchannel, len >= 4, edit dist <= 1)
    # -------------------------------------------------------------------------
    Case(
        "howdhu",
        ("howdhu",),
        ("howdhu",),
        False,
        False,
        False,
        (),
        None,
        "fuzzy howdu (len 6, dist 1)",
    ),
    Case("acchaa", ("acchaa",), ("acchaa",), False, False, False, (), None, "fuzzy accha (dist 1)"),
    Case(
        "barabar",
        ("barabar",),
        ("barabar",),
        False,
        False,
        False,
        (),
        None,
        "barabar/barobar match",
    ),
]


def test_table_cases_count_at_least_120() -> None:
    """Table-driven test suite must have at least 120 cases per done_when."""
    assert len(TABLE_CASES) >= 120, f"Got {len(TABLE_CASES)} cases, expected >= 120"


@pytest.mark.parametrize("case", TABLE_CASES, ids=lambda c: f"{c.text}-{c.lang_hint or 'any'}")
def test_lexicon_analyze_table_cases(case: Case) -> None:
    """Run table-driven cases against Lexicon.analyze."""
    res = analyze(case.text, lang_hint=case.lang_hint)
    assert res.tokens == case.tokens, (
        f"Tokens mismatch for '{case.text}': {res.tokens} != {case.tokens}"
    )
    assert res.bc_tokens == case.bc_tokens, (
        f"bc_tokens mismatch for '{case.text}': {res.bc_tokens} != {case.bc_tokens}"
    )
    assert res.stop_hit is case.stop, (
        f"stop_hit mismatch for '{case.text}': {res.stop_hit} != {case.stop}"
    )
    assert res.contrast_hit is case.contrast, (
        f"contrast_hit mismatch for '{case.text}': {res.contrast_hit} != {case.contrast}"
    )
    assert res.question is case.question, (
        f"question mismatch for '{case.text}': {res.question} != {case.question}"
    )
    assert res.content_tokens == case.content_tokens, (
        f"content_tokens mismatch for '{case.text}': {res.content_tokens} != {case.content_tokens}"
    )


# =============================================================================
# Specific Failure Mode Invariant Tests
# =============================================================================


def test_failure_mode_nahi_ruko_wait_not_backchannel() -> None:
    """'nahi', 'ruko', 'wait' must NEVER be treated as backchannels."""
    for word in ("nahi", "ruko", "wait", "नहीं", "रुको", "stop", "bas"):
        res = analyze(word)
        assert res.stop_hit is True, f"'{word}' must trigger stop_hit"
        assert res.bc_tokens == (), f"'{word}' must NOT be in bc_tokens"
        assert res.is_pure_backchannel is False


def test_failure_mode_contrast_token_not_missed() -> None:
    """'haan lekin ...' must not be classified as pure backchannel."""
    res = analyze("haan lekin ruko")
    assert res.contrast_hit is True
    assert res.stop_hit is True
    assert res.is_pure_backchannel is False


def test_fuzzy_matching_never_on_stop_or_contrast() -> None:
    """Fuzzy matching must NEVER apply to stop or contrast words."""
    # A near-miss like 'wiat' (for wait) or 'sotp' (for stop) must NOT trigger
    # stop_hit via fuzzy match!
    res = analyze("wiat")
    assert res.stop_hit is False
    assert res.bc_tokens == ()
    assert res.content_tokens == ("wiat",)


def test_fuzzy_matching_never_on_short_tokens() -> None:
    """Fuzzy matching must NOT apply to tokens with len < 4."""
    # 'ji' (len 2) - 'pi' should NOT match 'ji'
    res = analyze("pi")
    assert res.bc_tokens == ()
    assert res.content_tokens == ("pi",)

    # 'ok' (len 2) - 'on' should NOT match 'ok'
    res = analyze("on")
    assert res.bc_tokens == ()
