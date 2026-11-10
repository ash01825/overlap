"""Threshold grid search over Tuning parameters on train split with held-out evaluation (§7, §12).

Safeguards:
- Strictly separates deterministic train and held-out splits (raises ValueError if empty).
- Selects optimal Tuning parameters on the train split; evaluates generalization on held-out.
- Never mutates `Tuning` defaults in-place; explicitly reports any recommended parameter deltas.
- Completely deterministic: identical corpus and grid produce byte-identical Markdown reports.

CLI usage:
    python -m overlap.bench.sweep [--output reports/classifier_sweep.md]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from overlap.bench.corpus import CorpusCase, load_corpus, split_corpus
from overlap.bench.replay import ReplayReport, replay_corpus
from overlap.core.types import Tuning

DEFAULT_SWEEP_REPORT_PATH: Path = (
    Path(__file__).resolve().parents[2] / "reports" / "classifier_sweep.md"
)

SWEEP_FIELDS: tuple[str, ...] = (
    "classify_min_ms",
    "content_min_overlap_ms",
    "bc_min_ms",
    "bc_max_overlap_ms",
    "assertive_db",
    "sidetalk_db",
)


@dataclass(frozen=True, slots=True)
class ParameterGrid:
    """Grid of Tuning threshold values to evaluate on the train split."""

    classify_min_ms: tuple[int, ...] = (220, 250)
    content_min_overlap_ms: tuple[int, ...] = (220, 250)
    bc_min_ms: tuple[int, ...] = (180, 200)
    bc_max_overlap_ms: tuple[int, ...] = (600,)
    assertive_db: tuple[float, ...] = (2.0, 3.0, 4.0)
    sidetalk_db: tuple[float, ...] = (-10.5, -9.0, -7.5)

    def iter_tunings(self, base: Tuning | None = None) -> tuple[Tuning, ...]:
        """Generate all candidate Tuning configurations in deterministic order."""
        cfg = base if base is not None else Tuning()
        tunings: list[Tuning] = []
        for c_min, cont_min, b_min, b_max, a_db, s_db in itertools.product(
            self.classify_min_ms,
            self.content_min_overlap_ms,
            self.bc_min_ms,
            self.bc_max_overlap_ms,
            self.assertive_db,
            self.sidetalk_db,
        ):
            tunings.append(
                replace(
                    cfg,
                    classify_min_ms=int(c_min),
                    content_min_overlap_ms=int(cont_min),
                    bc_min_ms=int(b_min),
                    bc_max_overlap_ms=int(b_max),
                    assertive_db=float(a_db),
                    sidetalk_db=float(s_db),
                )
            )
        return tuple(tunings)


@dataclass(frozen=True, slots=True)
class SweepCandidate:
    """Train-split evaluation summary for a single candidate Tuning."""

    tuning: Tuning
    train_accuracy: float
    train_fir: float
    train_missed_rate: float
    train_verdict_p95_ms: float
    train_targets_met: bool
    deltas_from_baseline: int


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Complete outcome of a threshold sweep across train and held-out splits."""

    total_cases: int
    train_size: int
    held_out_size: int
    languages: tuple[str, ...]
    baseline_tuning: Tuning
    best_tuning: Tuning
    recommended_changes: dict[str, tuple[Any, Any]]
    candidates: tuple[SweepCandidate, ...]
    baseline_train_report: ReplayReport
    baseline_held_out_report: ReplayReport
    best_train_report: ReplayReport
    best_held_out_report: ReplayReport
    generalization_gap: float
    held_out_passed: bool
    findings: tuple[str, ...]


def _count_deltas(candidate: Tuning, baseline: Tuning) -> int:
    """Count how many swept parameters differ from baseline Tuning."""
    return sum(
        1
        for field_name in SWEEP_FIELDS
        if getattr(candidate, field_name) != getattr(baseline, field_name)
    )


def diff_tuning(baseline: Tuning, candidate: Tuning) -> dict[str, tuple[Any, Any]]:
    """Return explicit map of {field: (baseline_val, candidate_val)} for changed fields."""
    changes: dict[str, tuple[Any, Any]] = {}
    for field_name in SWEEP_FIELDS:
        b_val = getattr(baseline, field_name)
        c_val = getattr(candidate, field_name)
        if b_val != c_val:
            changes[field_name] = (b_val, c_val)
    return changes


def _candidate_sort_key(cand: SweepCandidate) -> tuple[Any, ...]:
    """Deterministic ranking key for train-split candidates (smaller tuple is better).

    Priority:
    1. Meets train targets (FIR <= 5% and missed <= 5%)
    2. Lowest combined error (train_fir + train_missed_rate)
    3. Highest train accuracy
    4. Fewest parameter deltas from baseline Tuning() (regularization against overfitting)
    5. Lowest p95 verdict latency
    6. Deterministic parameter values
    """
    t = cand.tuning
    return (
        0 if cand.train_targets_met else 1,
        round(cand.train_fir + cand.train_missed_rate, 6),
        round(1.0 - cand.train_accuracy, 6),
        cand.deltas_from_baseline,
        round(cand.train_verdict_p95_ms, 3),
        t.classify_min_ms,
        t.content_min_overlap_ms,
        t.bc_min_ms,
        t.bc_max_overlap_ms,
        t.assertive_db,
        t.sidetalk_db,
    )


def run_sweep(
    cases: Sequence[CorpusCase],
    *,
    grid: ParameterGrid | None = None,
    baseline: Tuning | None = None,
    train_ratio: float = 0.70,
) -> SweepResult:
    """Run grid search over Tuning thresholds on the train split and evaluate on held-out."""
    if len(cases) == 0:
        raise ValueError("Cannot run sweep on an empty corpus")

    train_cases, held_out_cases = split_corpus(cases, train_ratio=train_ratio)
    if len(train_cases) == 0:
        raise ValueError("Sweep requires a non-empty train split to fit thresholds")
    if len(held_out_cases) == 0:
        raise ValueError("Sweep requires a non-empty held-out split to prevent overfitting")

    base_tuning = baseline if baseline is not None else Tuning()
    param_grid = grid if grid is not None else ParameterGrid()
    candidate_tunings = param_grid.iter_tunings(base_tuning)
    if not candidate_tunings:
        raise ValueError("ParameterGrid produced 0 candidate Tuning configurations")

    evaluated: list[SweepCandidate] = []
    train_reports_by_idx: dict[int, ReplayReport] = {}

    for idx, cand_tuning in enumerate(candidate_tunings):
        t_rep = replay_corpus(train_cases, cand_tuning)
        train_reports_by_idx[idx] = t_rep
        p95_ms = (
            float(t_rep.derived_metrics.t_verdict_p95_ms)
            if t_rep.derived_metrics.t_verdict_p95_ms is not None
            else 0.0
        )
        targets_met = (
            t_rep.false_interruption_rate <= 0.05 and t_rep.missed_interruption_rate <= 0.05
        )
        evaluated.append(
            SweepCandidate(
                tuning=cand_tuning,
                train_accuracy=t_rep.accuracy,
                train_fir=t_rep.false_interruption_rate,
                train_missed_rate=t_rep.missed_interruption_rate,
                train_verdict_p95_ms=p95_ms,
                train_targets_met=targets_met,
                deltas_from_baseline=_count_deltas(cand_tuning, base_tuning),
            )
        )

    ranked = tuple(sorted(evaluated, key=_candidate_sort_key))
    best_cand = ranked[0]
    best_tuning = best_cand.tuning

    baseline_train_report = replay_corpus(train_cases, base_tuning)
    baseline_held_out_report = replay_corpus(held_out_cases, base_tuning)
    best_train_report = replay_corpus(train_cases, best_tuning)
    best_held_out_report = replay_corpus(held_out_cases, best_tuning)

    changes = diff_tuning(base_tuning, best_tuning)
    gen_gap = best_train_report.accuracy - best_held_out_report.accuracy

    held_out_ok = (
        best_held_out_report.false_interruption_rate <= 0.05
        and best_held_out_report.missed_interruption_rate <= 0.05
    )

    findings: list[str] = []
    if best_held_out_report.false_interruption_rate > 0.05:
        findings.append(
            f"FINDING: Held-out FIR {best_held_out_report.false_interruption_rate * 100.0:.2f}% "
            "exceeds 5.0% target."
        )
    if best_held_out_report.missed_interruption_rate > 0.05:
        findings.append(
            f"FINDING: Held-out missed-interruption rate "
            f"{best_held_out_report.missed_interruption_rate * 100.0:.2f}% exceeds 5.0% target."
        )
    if not findings:
        findings.append(
            "Held-out FIR and missed-interruption rate both meet the <= 5.0% "
            "ARCHITECTURE §12 targets."
        )

    languages = tuple(sorted({c.lang for c in cases}))

    return SweepResult(
        total_cases=len(cases),
        train_size=len(train_cases),
        held_out_size=len(held_out_cases),
        languages=languages,
        baseline_tuning=base_tuning,
        best_tuning=best_tuning,
        recommended_changes=changes,
        candidates=ranked,
        baseline_train_report=baseline_train_report,
        baseline_held_out_report=baseline_held_out_report,
        best_train_report=best_train_report,
        best_held_out_report=best_held_out_report,
        generalization_gap=gen_gap,
        held_out_passed=held_out_ok,
        findings=tuple(findings),
    )


def format_sweep_report(result: SweepResult) -> str:
    """Format a deterministic Markdown report for `reports/classifier_sweep.md`."""
    b_ho = result.baseline_held_out_report
    s_ho = result.best_held_out_report
    s_tr = result.best_train_report

    b_p95 = (
        f"{b_ho.derived_metrics.t_verdict_p95_ms:.1f} ms"
        if b_ho.derived_metrics.t_verdict_p95_ms is not None
        else "N/A"
    )
    s_p95 = (
        f"{s_ho.derived_metrics.t_verdict_p95_ms:.1f} ms"
        if s_ho.derived_metrics.t_verdict_p95_ms is not None
        else "N/A"
    )

    status_str = "PASS" if result.held_out_passed else "FAIL"

    lines: list[str] = [
        "# Overlap Classifier Threshold Sweep Report",
        "",
        "## 1. Corpus & Split Summary",
        "",
        f"- **Total Corpus Cases:** {result.total_cases}",
        f"- **Train Split (SHA-256 < 0.70):** {result.train_size} cases",
        f"- **Held-Out Split (SHA-256 >= 0.70):** {result.held_out_size} cases",
        f"- **Languages ({len(result.languages)}):** `{', '.join(result.languages)}`",
        f"- **Grid Candidates Evaluated:** {len(result.candidates)}",
        f"- **Held-Out Target Evaluation:** **{status_str}**",
        "",
        "## 2. Train vs Held-Out Performance",
        "",
        (
            "| Split / Configuration | Cases | Accuracy | FIR (Target <= 5.0%) | "
            "Missed Interrupt (Target <= 5.0%) | `t_verdict` p95 | Status |"
        ),
        "|---|---:|---:|---:|---:|---:|---|",
        (
            f"| **Train (Selected)** | {s_tr.total} | {s_tr.accuracy * 100.0:.2f}% | "
            f"{s_tr.false_interruption_rate * 100.0:.2f}% "
            f"({s_tr.false_interruption_count}/{s_tr.ground_truth_backchannels}) | "
            f"{s_tr.missed_interruption_rate * 100.0:.2f}% "
            f"({s_tr.missed_interruption_count}/{s_tr.ground_truth_interruptions}) | "
            f"{s_tr.derived_metrics.t_verdict_p95_ms or 0.0:.1f} ms | "
            f"{'PASS' if s_tr.passed else 'FAIL'} |"
        ),
        (
            f"| **Held-Out (Baseline `Tuning()`)** | {b_ho.total} | "
            f"{b_ho.accuracy * 100.0:.2f}% | "
            f"{b_ho.false_interruption_rate * 100.0:.2f}% "
            f"({b_ho.false_interruption_count}/{b_ho.ground_truth_backchannels}) | "
            f"{b_ho.missed_interruption_rate * 100.0:.2f}% "
            f"({b_ho.missed_interruption_count}/{b_ho.ground_truth_interruptions}) | "
            f"{b_p95} | {'PASS' if b_ho.passed else 'FAIL'} |"
        ),
        (
            f"| **Held-Out (Selected `Tuning`)** | {s_ho.total} | "
            f"{s_ho.accuracy * 100.0:.2f}% | "
            f"{s_ho.false_interruption_rate * 100.0:.2f}% "
            f"({s_ho.false_interruption_count}/{s_ho.ground_truth_backchannels}) | "
            f"{s_ho.missed_interruption_rate * 100.0:.2f}% "
            f"({s_ho.missed_interruption_count}/{s_ho.ground_truth_interruptions}) | "
            f"{s_p95} | **{status_str}** |"
        ),
        "",
        (
            f"- **Generalization Gap (Train − Held-Out Accuracy):** "
            f"{result.generalization_gap * 100.0:+.2f}%"
        ),
        "",
        "### Findings",
        "",
    ]
    for finding in result.findings:
        lines.append(f"- {finding}")

    lines.extend(
        [
            "",
            "## 3. Recommended `Tuning` Default Changes",
            "",
        ]
    )
    if not result.recommended_changes:
        lines.append(
            "No changes to `Tuning` defaults recommended (`0` parameter deltas; "
            "baseline `Tuning()` achieves optimal train score and meets all held-out targets)."
        )
    else:
        lines.extend(
            [
                "| Parameter | Baseline Default | Recommended Value |",
                "|---|---:|---:|",
            ]
        )
        for param_name, (old_val, new_val) in sorted(result.recommended_changes.items()):
            lines.append(f"| `{param_name}` | `{old_val}` | `{new_val}` |")

    lines.extend(
        [
            "",
            "## 4. Held-Out Confusion Matrix",
            "",
            "| Ground Truth \\ Predicted | `BACKCHANNEL` | `INTERRUPT` | `SIDE_TALK` |",
            "|---|---:|---:|---:|",
        ]
    )
    for gt_label in ("BACKCHANNEL", "INTERRUPT", "SIDE_TALK"):
        row = s_ho.confusion_matrix[gt_label]
        lines.append(
            f"| **`{gt_label}`** | {row['BACKCHANNEL']} | {row['INTERRUPT']} | {row['SIDE_TALK']} |"
        )

    lines.extend(
        [
            "",
            "## 5. Held-Out Per-Language Breakdown",
            "",
            "| Language | Cases | Accuracy | FIR | Missed Interrupt | BC / INT / ST |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for lm in s_ho.per_language:
        counts = (
            f"{lm.ground_truth_backchannels}/"
            f"{lm.ground_truth_interruptions}/"
            f"{lm.ground_truth_sidetalk}"
        )
        lines.append(
            f"| `{lm.lang}` | {lm.total} | {lm.accuracy * 100.0:.1f}% | "
            f"{lm.fir * 100.0:.1f}% | {lm.missed_interruption_rate * 100.0:.1f}% | {counts} |"
        )

    lines.extend(
        [
            "",
            "## 6. Grid-Search ROC / Threshold Table (Train Split — Top 15 & Bottom 5)",
            "",
            (
                "| Rank | `classify_min_ms` | `content_min_overlap_ms` | `bc_min_ms` | "
                "`assertive_db` | `sidetalk_db` | Train Acc | Train FIR | "
                "Train Missed | Deltas |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    top_n = result.candidates[:15]
    bottom_n = result.candidates[-5:] if len(result.candidates) > 15 else ()
    for rank_idx, cand in enumerate(top_n, start=1):
        t = cand.tuning
        lines.append(
            f"| {rank_idx} | {t.classify_min_ms} | {t.content_min_overlap_ms} | {t.bc_min_ms} | "
            f"{t.assertive_db:.1f} | {t.sidetalk_db:.1f} | {cand.train_accuracy * 100.0:.2f}% | "
            f"{cand.train_fir * 100.0:.2f}% | {cand.train_missed_rate * 100.0:.2f}% | "
            f"{cand.deltas_from_baseline} |"
        )
    if bottom_n:
        lines.append("| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |")
        start_rank = len(result.candidates) - len(bottom_n) + 1
        for offset, cand in enumerate(bottom_n):
            t = cand.tuning
            lines.append(
                f"| {start_rank + offset} | {t.classify_min_ms} | {t.content_min_overlap_ms} | "
                f"{t.bc_min_ms} | {t.assertive_db:.1f} | {t.sidetalk_db:.1f} | "
                f"{cand.train_accuracy * 100.0:.2f}% | {cand.train_fir * 100.0:.2f}% | "
                f"{cand.train_missed_rate * 100.0:.2f}% | {cand.deltas_from_baseline} |"
            )

    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m overlap.bench.sweep`."""
    parser = argparse.ArgumentParser(
        description="Grid-search Tuning thresholds on train split and write held-out report."
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Optional path to corpus .jsonl file or directory.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SWEEP_REPORT_PATH),
        help="Path to write Markdown sweep report (default: reports/classifier_sweep.md).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    corpus_path = Path(args.corpus) if args.corpus is not None else None
    cases = load_corpus(corpus_path)
    result = run_sweep(cases)
    report_md = format_sweep_report(result)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")
    print(report_md)
    return 0 if result.held_out_passed else 1


if __name__ == "__main__":
    sys.exit(main())
