"""Blind A/B human rating CLI and session manager (`overlap.bench.rating_cli`).

Per ARCHITECTURE §12:
- Collects for each anonymized pair (`pair_01`..`pair_N`):
  1. Naturalness 1–5 for Arm A and Arm B
  2. 'felt interrupted' (yes/no) for Arm A and Arm B
  3. 'felt ignored' (yes/no) for Arm A and Arm B
  4. 'thought call dropped' (yes/no) for Arm A and Arm B
  5. Forced preference ('A' or 'B'; ties are not permitted)
- Strictly preserves blinding during rating: reads only `blind_manifest.json` and
  refuses to reveal or access `_private/assignment_key.json` until every pair in
  the manifest has a submitted rating.
- Once all ratings are submitted, optionally unblinds and writes `reports/ab/summary.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from overlap.bench.ab import (
    AbAnalysisSummary,
    RaterSubmission,
    analyze_ab_results,
    load_assignment_key,
    write_ab_summary_report,
)


class UnblindPrematureError(RuntimeError):
    """Raised when attempting to unblind or reveal A/B identities before all pairs are rated."""


@dataclass(frozen=True, slots=True)
class BlindPairPrompt:
    """Rater-facing metadata for a single anonymized pair (contains no system names)."""

    pair_id: str
    title: str
    lang: str
    clip_a: str
    clip_b: str


def load_blind_manifest(manifest_path: str | Path) -> tuple[BlindPairPrompt, ...]:
    """Load the rater-facing `blind_manifest.json` without reading the assignment key."""
    p = Path(manifest_path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    pairs = raw.get("pairs", [])
    prompts: list[BlindPairPrompt] = []
    for item in pairs:
        # Guard against accidental leakage of arm identities in manifest
        if "arm_a_system" in item or "arm_b_system" in item:
            raise ValueError(f"Manifest {p} is not blind: contains system identity fields")
        prompts.append(
            BlindPairPrompt(
                pair_id=str(item["pair_id"]),
                title=str(item.get("title", item["pair_id"])),
                lang=str(item.get("lang", "en-IN")),
                clip_a=str(item["clip_A"]),
                clip_b=str(item["clip_B"]),
            )
        )
    return tuple(prompts)


def save_ratings_jsonl(ratings: Sequence[RaterSubmission], path: str | Path) -> None:
    """Persist collected `RaterSubmission` entries to a JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(r), sort_keys=True) for r in ratings]
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_ratings_jsonl(path: str | Path) -> tuple[RaterSubmission, ...]:
    """Load `RaterSubmission` records from a JSONL file."""
    p = Path(path)
    if not p.exists():
        return ()
    out: list[RaterSubmission] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line_s = line.strip()
        if not line_s:
            continue
        item = json.loads(line_s)
        out.append(
            RaterSubmission(
                pair_id=str(item["pair_id"]),
                rater_id=str(item["rater_id"]),
                naturalness_a=int(item["naturalness_a"]),
                naturalness_b=int(item["naturalness_b"]),
                felt_interrupted_a=bool(item["felt_interrupted_a"]),
                felt_interrupted_b=bool(item["felt_interrupted_b"]),
                felt_ignored_a=bool(item["felt_ignored_a"]),
                felt_ignored_b=bool(item["felt_ignored_b"]),
                thought_call_dropped_a=bool(item["thought_call_dropped_a"]),
                thought_call_dropped_b=bool(item["thought_call_dropped_b"]),
                preferred_arm=str(item["preferred_arm"]),
                notes=str(item.get("notes", "")),
            )
        )
    return tuple(out)


def _parse_bool_answer(raw: str) -> bool:
    val = raw.strip().lower()
    if val in ("y", "yes", "true", "1"):
        return True
    if val in ("n", "no", "false", "0"):
        return False
    raise ValueError(f"Expected yes/no (y/n), got {raw!r}")


def _parse_rating_1_to_5(raw: str) -> int:
    val = int(raw.strip())
    if not (1 <= val <= 5):
        raise ValueError(f"Rating must be an integer from 1 to 5, got {val}")
    return val


def _parse_forced_preference(raw: str) -> str:
    val = raw.strip().upper()
    if val not in ("A", "B"):
        raise ValueError(f"Forced preference must be 'A' or 'B' (no ties allowed), got {raw!r}")
    return val


class BlindRatingSession:
    """Stateful blind rating session enforcing that unblinding is blocked until completion."""

    def __init__(
        self,
        manifest: Sequence[BlindPairPrompt],
        *,
        rater_id: str = "rater_01",
    ) -> None:
        self._manifest = tuple(manifest)
        self._rater_id = rater_id
        self._submissions: dict[str, RaterSubmission] = {}

    @property
    def manifest(self) -> tuple[BlindPairPrompt, ...]:
        """Rater-facing blind pair prompts."""
        return self._manifest

    @property
    def rater_id(self) -> str:
        """Identifier for the current human rater."""
        return self._rater_id

    @property
    def is_complete(self) -> bool:
        """Return True iff every pair in the manifest has been rated."""
        required = {p.pair_id for p in self._manifest}
        return required.issubset(self._submissions.keys()) and len(required) > 0

    @property
    def submissions(self) -> tuple[RaterSubmission, ...]:
        """Submitted ratings in manifest order."""
        return tuple(
            self._submissions[p.pair_id] for p in self._manifest if p.pair_id in self._submissions
        )

    def submit_rating(self, submission: RaterSubmission) -> None:
        """Record a rating for a single pair after validating pair_id membership."""
        valid_ids = {p.pair_id for p in self._manifest}
        if submission.pair_id not in valid_ids:
            raise KeyError(f"Unknown pair_id {submission.pair_id!r} not in manifest")
        self._submissions[submission.pair_id] = submission

    def unblind_and_summarize(
        self,
        key_path: str | Path,
        *,
        report_path: str | Path | None = None,
    ) -> AbAnalysisSummary:
        """Unblind ratings using `key_path` ONLY after all pairs in the manifest are rated."""
        if not self.is_complete:
            missing = [p.pair_id for p in self._manifest if p.pair_id not in self._submissions]
            raise UnblindPrematureError(
                f"Cannot unblind before all pairs are rated; missing pairs: {missing}"
            )
        assignments = load_assignment_key(key_path)
        summary = analyze_ab_results(assignments, self.submissions)
        if report_path is not None:
            write_ab_summary_report(summary, assignments=assignments, report_path=report_path)
        return summary


def collect_pair_rating_interactive(
    prompt: BlindPairPrompt,
    *,
    rater_id: str,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], Any] = print,
) -> RaterSubmission:
    """Interactively prompt a human rater for one blind A/B pair."""
    output_fn(f"\n=== Blind Evaluation: {prompt.pair_id} ({prompt.title}) [lang={prompt.lang}] ===")
    output_fn(f"  Listen to Arm A: {prompt.clip_a}")
    output_fn(f"  Listen to Arm B: {prompt.clip_b}")

    def _ask(question: str, parser: Callable[[str], Any]) -> Any:
        while True:
            raw = input_fn(question)
            try:
                return parser(raw)
            except ValueError as exc:
                output_fn(f"  Invalid input: {exc}. Please try again.")

    nat_a: int = _ask("  Arm A — Naturalness (1-5): ", _parse_rating_1_to_5)
    int_a: bool = _ask("  Arm A — Did you feel interrupted? (y/n): ", _parse_bool_answer)
    ign_a: bool = _ask("  Arm A — Did you feel ignored? (y/n): ", _parse_bool_answer)
    drp_a: bool = _ask("  Arm A — Did you think the call dropped? (y/n): ", _parse_bool_answer)

    nat_b: int = _ask("  Arm B — Naturalness (1-5): ", _parse_rating_1_to_5)
    int_b: bool = _ask("  Arm B — Did you feel interrupted? (y/n): ", _parse_bool_answer)
    ign_b: bool = _ask("  Arm B — Did you feel ignored? (y/n): ", _parse_bool_answer)
    drp_b: bool = _ask("  Arm B — Did you think the call dropped? (y/n): ", _parse_bool_answer)

    pref: str = _ask(
        "  Forced Preference — Which arm felt more natural? (A/B): ",
        _parse_forced_preference,
    )

    return RaterSubmission(
        pair_id=prompt.pair_id,
        rater_id=rater_id,
        naturalness_a=nat_a,
        naturalness_b=nat_b,
        felt_interrupted_a=int_a,
        felt_interrupted_b=int_b,
        felt_ignored_a=ign_a,
        felt_ignored_b=ign_b,
        thought_call_dropped_a=drp_a,
        thought_call_dropped_b=drp_b,
        preferred_arm=pref,
    )


def run_interactive_rating_cli(
    manifest_path: str | Path,
    ratings_out_path: str | Path,
    *,
    rater_id: str = "rater_01",
    key_path: str | Path | None = None,
    report_path: str | Path | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], Any] = print,
) -> tuple[RaterSubmission, ...]:
    """Run a full blind rating session over `manifest_path` and save ratings."""
    manifest = load_blind_manifest(manifest_path)
    session = BlindRatingSession(manifest, rater_id=rater_id)

    for pair_prompt in manifest:
        sub = collect_pair_rating_interactive(
            pair_prompt,
            rater_id=rater_id,
            input_fn=input_fn,
            output_fn=output_fn,
        )
        session.submit_rating(sub)

    save_ratings_jsonl(session.submissions, ratings_out_path)
    output_fn(f"\nAll {len(session.submissions)} ratings recorded to {ratings_out_path}.")

    if key_path is not None and Path(key_path).exists():
        summary = session.unblind_and_summarize(key_path, report_path=report_path)
        ci_lo, ci_hi = summary.preference_wilson_ci
        output_fn(
            f"Unblinded Summary: OVERLAP preferred {summary.overlap_wins}/{summary.total_ratings} "
            f"({summary.overlap_preference_rate * 100.0:.1f}%, "
            f"95% Wilson CI [{ci_lo * 100.0:.1f}%, {ci_hi * 100.0:.1f}%], "
            f"sign test p={summary.sign_test.p_value:.4f})"
        )

    return session.submissions


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m overlap.bench.rating_cli`."""
    parser = argparse.ArgumentParser(
        prog="overlap.bench.rating_cli",
        description=(
            "Collect blind human A/B naturalness ratings and unblind only after completion."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/ab/blind_manifest.json"),
        help="Path to blind_manifest.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/ab/ratings.jsonl"),
        help="Output JSONL file for submitted ratings.",
    )
    parser.add_argument(
        "--rater-id",
        type=str,
        default="rater_01",
        help="Rater identifier string.",
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=Path("reports/ab/_private/assignment_key.json"),
        help="Assignment key JSON (accessed only after all ratings are submitted).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/ab/summary.md"),
        help="Output path for unblinded Markdown summary report.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.manifest.exists():
        print(f"ERROR: blind manifest not found at {args.manifest}", file=sys.stderr)
        return 1

    run_interactive_rating_cli(
        args.manifest,
        args.output,
        rater_id=args.rater_id,
        key_path=args.key,
        report_path=args.report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
