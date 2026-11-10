"""Labelled overlap corpus schema, loader, hash-based split, and validators (§7, §12).

Corpus format (JSONL):
    {
      "id": str,
      "lang": str,
      "timeline": [
        {"t_ms": int, "vad": bool, "loudness": float, "text": str | None, "conf": float | None},
        ...
      ],
      "label": "BACKCHANNEL" | "INTERRUPT" | "SIDE_TALK"
    }

Invariants & safeguards:
- >= 200 cases across >= 8 languages, balanced across BACKCHANNEL / INTERRUPT / SIDE_TALK.
- Fixed deterministic train / held-out split by SHA-256 hash of `case.id`.
- PII validator prevents accidental leakage of emails, phone numbers, Aadhaar, or PANs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from overlap.core.types import Label

SplitName = Literal["train", "held_out"]

DEFAULT_CORPUS_DIR: Path = Path(__file__).resolve().parents[2] / "bench" / "corpus"

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")
_AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[\s-]\d{4}[\s-]\d{4}(?!\d)")
_LONG_DIGITS_RE = re.compile(r"(?<!\d)\d{10,16}(?!\d)")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")


def assign_split(case_id: str, train_ratio: float = 0.70) -> SplitName:
    """Deterministically assign a corpus case ID to 'train' or 'held_out' via SHA-256 hash."""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0.0, 1.0), got {train_ratio}")
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") / float(0xFFFFFFFF)
    return "train" if bucket < train_ratio else "held_out"


@dataclass(frozen=True, slots=True)
class TimelineStep:
    """A single point in an overlap event timeline."""

    t_ms: int
    vad: bool
    loudness: float
    text: str | None = None
    conf: float | None = None
    is_final: bool = False

    @property
    def user_speaking(self) -> bool:
        """Alias matching OverlapFeatures.user_speaking."""
        return self.vad

    @property
    def rel_loudness_db(self) -> float:
        """Alias matching OverlapFeatures.rel_loudness_db."""
        return self.loudness

    @property
    def partial_text(self) -> str | None:
        """Alias matching OverlapFeatures.partial_text."""
        return self.text

    @property
    def stt_conf(self) -> float | None:
        """Alias matching OverlapFeatures.stt_conf."""
        return self.conf

    def to_dict(self) -> dict[str, Any]:
        """Serialize timeline step to a JSON-compatible dictionary."""
        out: dict[str, Any] = {
            "t_ms": self.t_ms,
            "vad": self.vad,
            "loudness": self.loudness,
            "text": self.text,
            "conf": self.conf,
        }
        if self.is_final:
            out["is_final"] = True
        return out


@dataclass(frozen=True, slots=True)
class CorpusCase:
    """A single labelled overlap evaluation case."""

    id: str
    lang: str
    timeline: tuple[TimelineStep, ...]
    label: Label
    category: str = ""

    @property
    def split(self) -> SplitName:
        """Deterministic hash-based split ('train' or 'held_out')."""
        return assign_split(self.id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize corpus case to a JSON-compatible dictionary."""
        out: dict[str, Any] = {
            "id": self.id,
            "lang": self.lang,
            "timeline": [s.to_dict() for s in self.timeline],
            "label": self.label.value,
        }
        if self.category:
            out["category"] = self.category
        return out


def _parse_vad(raw: Any) -> bool:
    """Parse boolean or 'on'/'off' string VAD flag."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        norm = raw.strip().lower()
        if norm in ("on", "true", "1", "active", "yes"):
            return True
        if norm in ("off", "false", "0", "inactive", "no"):
            return False
        raise ValueError(f"Unrecognized vad string value: {raw!r}")
    if isinstance(raw, (int, float)):
        return bool(raw)
    raise ValueError(f"Invalid vad field type: {type(raw).__name__}")


def _parse_step(raw: Mapping[str, Any]) -> TimelineStep:
    """Parse a dictionary into a TimelineStep."""
    t_ms = int(raw["t_ms"])
    if t_ms < 0:
        raise ValueError(f"Timeline t_ms must be >= 0, got {t_ms}")

    vad_val = raw.get("vad", raw.get("user_speaking", True))
    vad = _parse_vad(vad_val)

    loud_raw = raw.get("loudness", raw.get("rel_loudness_db", 0.0))
    loudness = float(loud_raw)

    text_raw = raw.get("text", raw.get("partial_text"))
    text = str(text_raw) if text_raw is not None else None

    conf_raw = raw.get("conf", raw.get("stt_conf"))
    conf = float(conf_raw) if conf_raw is not None else None

    is_final = bool(raw.get("is_final", False))
    return TimelineStep(
        t_ms=t_ms,
        vad=vad,
        loudness=loudness,
        text=text,
        conf=conf,
        is_final=is_final,
    )


def parse_case(raw: Mapping[str, Any]) -> CorpusCase:
    """Parse a raw JSON object into a validated CorpusCase."""
    case_id = str(raw.get("id", "")).strip()
    if not case_id:
        raise ValueError("Corpus case missing non-empty 'id'")

    lang = str(raw.get("lang", "")).strip()
    if not lang:
        raise ValueError(f"Corpus case {case_id!r} missing non-empty 'lang'")

    raw_timeline = raw.get("timeline")
    if not isinstance(raw_timeline, Sequence) or len(raw_timeline) == 0:
        raise ValueError(f"Corpus case {case_id!r} must have a non-empty 'timeline' list")

    steps = tuple(_parse_step(item) for item in raw_timeline)
    # Ensure timeline is sorted by t_ms
    for i in range(1, len(steps)):
        if steps[i].t_ms < steps[i - 1].t_ms:
            raise ValueError(f"Corpus case {case_id!r} timeline out of order at {steps[i].t_ms} ms")

    label_str = str(raw.get("label", "")).strip().upper()
    try:
        label = Label(label_str)
    except ValueError as exc:
        raise ValueError(f"Corpus case {case_id!r} has invalid label {label_str!r}") from exc

    if label == Label.PENDING:
        raise ValueError(f"Corpus case {case_id!r} ground-truth label cannot be PENDING")

    category = str(raw.get("category", "")).strip()
    return CorpusCase(
        id=case_id,
        lang=lang,
        timeline=steps,
        label=label,
        category=category,
    )


def _check_string_for_pii(value: str, context: str) -> None:
    """Raise ValueError if string contains email, phone, Aadhaar, or PAN PII patterns."""
    if _EMAIL_RE.search(value):
        raise ValueError(f"PII leak detected (email pattern) in {context}: {value!r}")
    if _PHONE_RE.search(value) or _LONG_DIGITS_RE.search(value):
        raise ValueError(f"PII leak detected (phone/number pattern) in {context}: {value!r}")
    if _AADHAAR_RE.search(value):
        raise ValueError(f"PII leak detected (Aadhaar pattern) in {context}: {value!r}")
    if _PAN_RE.search(value):
        raise ValueError(f"PII leak detected (PAN pattern) in {context}: {value!r}")


def validate_no_pii(cases: Sequence[CorpusCase]) -> None:
    """Verify that no corpus case ID, category, or transcript text leaks real PII."""
    for case in cases:
        _check_string_for_pii(case.id, f"case.id ({case.id})")
        _check_string_for_pii(case.category, f"case.category ({case.id})")
        for idx, step in enumerate(case.timeline):
            if step.text is not None:
                _check_string_for_pii(step.text, f"case {case.id} step[{idx}].text")


def validate_corpus_balance(
    cases: Sequence[CorpusCase],
    *,
    min_cases: int = 200,
    min_languages: int = 8,
    min_class_ratio: float = 0.20,
) -> None:
    """Validate corpus size, language diversity, duplicate IDs, and label balance."""
    total = len(cases)
    if total < min_cases:
        raise ValueError(f"Corpus size {total} is below minimum required {min_cases}")

    seen_ids: set[str] = set()
    languages: set[str] = set()
    counts: dict[Label, int] = {
        Label.BACKCHANNEL: 0,
        Label.INTERRUPT: 0,
        Label.SIDE_TALK: 0,
    }

    for case in cases:
        if case.id in seen_ids:
            raise ValueError(f"Duplicate corpus case id: {case.id!r}")
        seen_ids.add(case.id)
        languages.add(case.lang)
        if case.label not in counts:
            raise ValueError(f"Unexpected ground-truth label {case.label} in {case.id!r}")
        counts[case.label] += 1

    if len(languages) < min_languages:
        raise ValueError(
            f"Corpus has {len(languages)} languages ({sorted(languages)}), "
            f"requires >= {min_languages}"
        )

    for label, count in counts.items():
        ratio = count / float(total)
        if ratio < min_class_ratio:
            raise ValueError(
                f"Corpus imbalance detected: label {label.value} has {count}/{total} "
                f"({ratio * 100.0:.1f}% < required {min_class_ratio * 100.0:.1f}%)"
            )


def split_corpus(
    cases: Sequence[CorpusCase],
    train_ratio: float = 0.70,
) -> tuple[tuple[CorpusCase, ...], tuple[CorpusCase, ...]]:
    """Split corpus deterministically into (train_cases, held_out_cases) by hash of case.id."""
    train: list[CorpusCase] = []
    held_out: list[CorpusCase] = []
    for case in cases:
        if assign_split(case.id, train_ratio=train_ratio) == "train":
            train.append(case)
        else:
            held_out.append(case)
    return (tuple(train), tuple(held_out))


def load_corpus(
    path: str | Path | None = None,
    *,
    validate: bool = True,
) -> tuple[CorpusCase, ...]:
    """Load labelled overlap corpus from a JSONL file or directory of JSONL files."""
    target = Path(path) if path is not None else DEFAULT_CORPUS_DIR
    jsonl_files: list[Path] = []
    if target.is_dir():
        jsonl_files = sorted(target.glob("*.jsonl"))
    elif target.is_file():
        jsonl_files = [target]
    else:
        raise FileNotFoundError(f"Corpus path not found: {target}")

    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl corpus files found in {target}")

    cases: list[CorpusCase] = []
    for file_path in jsonl_files:
        with file_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    raw = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {file_path}:{line_no}: {exc}") from exc
                if not isinstance(raw, Mapping):
                    raise ValueError(f"Expected JSON object at {file_path}:{line_no}")
                cases.append(parse_case(raw))

    result = tuple(cases)
    if validate:
        validate_no_pii(result)
        validate_corpus_balance(result)
    return result
