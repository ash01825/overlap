"""Offline corpus replay evaluator running the real OverlapClassifier + FSM pipeline (§7, §12).

Runs each CorpusCase through `overlap.core.classifier.classify` and `overlap.core.fsm.FSM`
(with zero duplicated classification or transition logic) while recording telemetry into
`TelemetryCollector` and deriving ARCHITECTURE §12 metrics.

CLI usage:
    python -m overlap.bench.replay [corpus_path] [--split all|train|held_out]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import overlap.core.classifier as classifier_mod
from overlap.bench.corpus import CorpusCase, SplitName, assign_split, load_corpus
from overlap.core.fsm import FSM
from overlap.core.lexicon import analyze
from overlap.core.metrics import (
    DerivedMetrics,
    MetricCheck,
    derive_metrics,
    evaluate_targets,
    rate,
)
from overlap.core.types import (
    FlushPlayout,
    FsmState,
    Label,
    OverlapFeatures,
    SetGain,
    SttPartial,
    TimerFired,
    Tuning,
    VadOffset,
    VadOnset,
    Verdict,
)
from overlap.telemetry.collector import TelemetryCollector

VALID_LABELS: tuple[Label, ...] = (
    Label.BACKCHANNEL,
    Label.INTERRUPT,
    Label.SIDE_TALK,
)


@dataclass(frozen=True, slots=True)
class CaseReplayResult:
    """Replay outcome for a single CorpusCase."""

    case_id: str
    lang: str
    split: SplitName
    expected: Label
    predicted: Label
    final_fsm_state: FsmState
    verdict_ms: int
    duck_ms: int
    flushed: bool
    reason: str
    correct: bool


@dataclass(frozen=True, slots=True)
class LanguageMetrics:
    """Per-language evaluation metrics row."""

    lang: str
    total: int
    correct: int
    accuracy: float
    fir: float
    false_interruptions: int
    ground_truth_backchannels: int
    missed_interruption_rate: float
    missed_interruptions: int
    ground_truth_interruptions: int
    ground_truth_sidetalk: int


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """Aggregated evaluation report across a replayed corpus."""

    results: tuple[CaseReplayResult, ...]
    total: int
    correct: int
    accuracy: float
    false_interruption_rate: float
    false_interruption_count: int
    ground_truth_backchannels: int
    missed_interruption_rate: float
    missed_interruption_count: int
    ground_truth_interruptions: int
    ground_truth_sidetalk: int
    sidetalk_false_interruptions: int
    confusion_matrix: dict[str, dict[str, int]]
    per_language: tuple[LanguageMetrics, ...]
    derived_metrics: DerivedMetrics
    target_checks: tuple[MetricCheck, ...]
    passed: bool


def replay_case(
    case: CorpusCase,
    tuning: Tuning | None = None,
    *,
    collector: TelemetryCollector | None = None,
    base_ts_ns: int = 0,
) -> CaseReplayResult:
    """Run a single CorpusCase through the real FSM and OverlapClassifier pipeline."""
    cfg = tuning if tuning is not None else Tuning()
    fsm = FSM(tuning=cfg, initial_state=FsmState.SPEAKING)

    # Identify initial VAD onset timestamp in ms
    onset_ms = 0
    for step in case.timeline:
        if step.vad:
            onset_ms = step.t_ms
            break

    user_speaking = False
    rel_loudness_db = 0.0
    partial_text: str | None = None
    last_partial_ms: int | None = None
    stt_conf: float | None = None
    is_final = False

    duck_ms = 0
    flushed = False
    deciding_verdict: Verdict | None = None
    verdict_ms = cfg.classify_cap_ms

    min_tick_ms = onset_ms + cfg.classify_min_ms
    cap_tick_ms = onset_ms + cfg.classify_cap_ms
    min_tick_fired = False
    cap_tick_fired = False

    def _evaluate_in_ducked(eval_t_ms: int) -> bool:
        """Invoke real classify() and transition FSM while in DUCKED. Returns True if yielded."""
        nonlocal deciding_verdict, verdict_ms, flushed
        if fsm.state != FsmState.DUCKED:
            return False

        overlap_ms = max(0, eval_t_ms - onset_ms)
        partial_age = (
            max(0, eval_t_ms - last_partial_ms)
            if (partial_text is not None and last_partial_ms is not None)
            else None
        )
        features = OverlapFeatures(
            overlap_ms=overlap_ms,
            user_speaking=user_speaking,
            rel_loudness_db=rel_loudness_db,
            partial_text=partial_text,
            partial_age_ms=partial_age,
            stt_conf=stt_conf,
            lang=case.lang,
            is_final=is_final,
        )
        now_ns = base_ts_ns + eval_t_ms * 1_000_000
        verdict = classifier_mod.classify(features, cfg, decided_at_ns=now_ns)
        effects = fsm.transition(verdict, now_ns=now_ns)

        if verdict.label != Label.PENDING:
            deciding_verdict = verdict
            verdict_ms = overlap_ms
            if any(isinstance(eff, FlushPlayout) for eff in effects):
                flushed = True
                return True
        return False

    for step in case.timeline:
        t_ms = step.t_ms

        # Fire intermediate soft tick (classify_min_ms) if it falls before this step
        if fsm.state == FsmState.DUCKED and not min_tick_fired and min_tick_ms < t_ms:
            min_tick_fired = True
            if _evaluate_in_ducked(min_tick_ms):
                break

        # Fire intermediate hard cap (classify_cap_ms) if it falls before this step
        if fsm.state == FsmState.DUCKED and not cap_tick_fired and cap_tick_ms < t_ms:
            cap_tick_fired = True
            if _evaluate_in_ducked(cap_tick_ms):
                break

        now_ns = base_ts_ns + t_ms * 1_000_000
        prev_vad = user_speaking
        user_speaking = step.vad
        rel_loudness_db = step.loudness
        if step.text is not None:
            partial_text = step.text
            last_partial_ms = t_ms
        if step.conf is not None:
            stt_conf = step.conf
        is_final = step.is_final

        # Handle VAD transitions on the FSM
        if not prev_vad and user_speaking:
            effects = fsm.transition(VadOnset(ts_ns=now_ns), now_ns=now_ns)
            if any(isinstance(eff, SetGain) for eff in effects):
                duck_ms = max(0, t_ms - onset_ms)
        elif prev_vad and not user_speaking:
            fsm.transition(VadOffset(ts_ns=now_ns), now_ns=now_ns)

        # Handle late escalation in SPEAKING state (ARCHITECTURE §6 row 9)
        if (
            fsm.state == FsmState.SPEAKING
            and fsm.context.last_backchannel_ns is not None
            and user_speaking
            and step.text is not None
            and step.text.strip()
        ):
            analysis = analyze(step.text, lang_hint=case.lang)
            has_escalation_signal = (
                analysis.stop_hit
                or analysis.contrast_hit
                or analysis.question
                or len(analysis.content_tokens) > 0
                or len(analysis.tokens) > cfg.bc_max_tokens
            )
            if has_escalation_signal:
                fsm.transition(
                    SttPartial(
                        text=step.text,
                        conf=stt_conf,
                        lang=case.lang,
                        ts_ns=now_ns,
                    ),
                    now_ns=now_ns,
                )

        # Evaluate classifier if in DUCKED state
        if fsm.state == FsmState.DUCKED:
            if t_ms >= min_tick_ms:
                min_tick_fired = True
            if t_ms >= cap_tick_ms:
                cap_tick_fired = True
            if _evaluate_in_ducked(t_ms):
                break

    # Post-timeline timer ticks if still in DUCKED
    if fsm.state == FsmState.DUCKED and not min_tick_fired:
        min_tick_fired = True
        _evaluate_in_ducked(min_tick_ms)

    if fsm.state == FsmState.DUCKED and not cap_tick_fired:
        cap_tick_fired = True
        _evaluate_in_ducked(cap_tick_ms)

    if fsm.state == FsmState.DUCKED:
        cap_ns = base_ts_ns + cap_tick_ms * 1_000_000
        effects = fsm.transition(TimerFired(name="classify_cap"), now_ns=cap_ns)
        if any(isinstance(eff, FlushPlayout) for eff in effects):
            flushed = True

    if fsm.state == FsmState.YIELDED or flushed:
        predicted = Label.INTERRUPT
        reason = deciding_verdict.reason if deciding_verdict is not None else "fsm_yielded"
    elif deciding_verdict is not None and deciding_verdict.label != Label.PENDING:
        predicted = deciding_verdict.label
        reason = deciding_verdict.reason
    else:
        predicted = Label.BACKCHANNEL
        reason = "fsm_speaking_restored"

    # Emit structured telemetry into TelemetryCollector if provided
    if collector is not None:
        onset_ns = base_ts_ns + onset_ms * 1_000_000
        duck_ns = onset_ns + duck_ms * 1_000_000
        verdict_ns = onset_ns + verdict_ms * 1_000_000
        last_step_ms = max(s.t_ms for s in case.timeline)
        offset_ns = base_ts_ns + max(last_step_ms, onset_ms + verdict_ms) * 1_000_000

        collector.emit("vad.onset", onset_ns, case_id=case.id, lang=case.lang)
        collector.emit(
            "duck.applied",
            duck_ns,
            onset_ts_ns=onset_ns,
            case_id=case.id,
        )
        collector.emit(
            "verdict",
            verdict_ns,
            onset_ts_ns=onset_ns,
            label=predicted.value,
            ground_truth=case.label.value,
            reason=reason,
            lang=case.lang,
            case_id=case.id,
        )
        if flushed:
            collector.emit(
                "yield.flush",
                verdict_ns,
                onset_ts_ns=onset_ns,
                last_sample_ts_ns=verdict_ns,
                interruption=True,
                case_id=case.id,
            )
        collector.emit("vad.offset", offset_ns, case_id=case.id)

    return CaseReplayResult(
        case_id=case.id,
        lang=case.lang,
        split=assign_split(case.id),
        expected=case.label,
        predicted=predicted,
        final_fsm_state=fsm.state,
        verdict_ms=verdict_ms,
        duck_ms=duck_ms,
        flushed=flushed,
        reason=reason,
        correct=(predicted == case.label),
    )


def replay_corpus(
    cases: Sequence[CorpusCase],
    tuning: Tuning | None = None,
    *,
    validate_balance: bool = False,
) -> ReplayReport:
    """Replay a sequence of CorpusCase items through the classifier+FSM pipeline."""
    if len(cases) == 0:
        raise ValueError("Cannot replay an empty corpus")

    cfg = tuning if tuning is not None else Tuning()
    collector = TelemetryCollector(max_memory_events=max(10_000, len(cases) * 10))

    results: list[CaseReplayResult] = []
    confusion: dict[str, dict[str, int]] = {
        lbl.value: {pred.value: 0 for pred in VALID_LABELS} for lbl in VALID_LABELS
    }

    lang_buckets: dict[str, list[CaseReplayResult]] = {}

    for idx, case in enumerate(cases):
        base_ts_ns = (idx + 1) * 10_000_000_000  # 10s virtual spacing per case
        res = replay_case(case, cfg, collector=collector, base_ts_ns=base_ts_ns)
        results.append(res)
        confusion[res.expected.value][res.predicted.value] += 1
        lang_buckets.setdefault(res.lang, []).append(res)

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = rate(correct, total)

    gt_bc = sum(1 for r in results if r.expected == Label.BACKCHANNEL)
    false_int = sum(
        1 for r in results if r.expected == Label.BACKCHANNEL and r.predicted == Label.INTERRUPT
    )
    fir = rate(false_int, gt_bc)

    gt_int = sum(1 for r in results if r.expected == Label.INTERRUPT)
    missed_int = sum(
        1 for r in results if r.expected == Label.INTERRUPT and r.predicted != Label.INTERRUPT
    )
    missed_rate = rate(missed_int, gt_int)

    gt_st = sum(1 for r in results if r.expected == Label.SIDE_TALK)
    st_false_int = sum(
        1 for r in results if r.expected == Label.SIDE_TALK and r.predicted == Label.INTERRUPT
    )

    per_lang: list[LanguageMetrics] = []
    for lang in sorted(lang_buckets.keys()):
        bucket = lang_buckets[lang]
        l_total = len(bucket)
        l_correct = sum(1 for r in bucket if r.correct)
        l_gt_bc = sum(1 for r in bucket if r.expected == Label.BACKCHANNEL)
        l_false_int = sum(
            1 for r in bucket if r.expected == Label.BACKCHANNEL and r.predicted == Label.INTERRUPT
        )
        l_gt_int = sum(1 for r in bucket if r.expected == Label.INTERRUPT)
        l_missed_int = sum(
            1 for r in bucket if r.expected == Label.INTERRUPT and r.predicted != Label.INTERRUPT
        )
        l_gt_st = sum(1 for r in bucket if r.expected == Label.SIDE_TALK)
        per_lang.append(
            LanguageMetrics(
                lang=lang,
                total=l_total,
                correct=l_correct,
                accuracy=rate(l_correct, l_total),
                fir=rate(l_false_int, l_gt_bc),
                false_interruptions=l_false_int,
                ground_truth_backchannels=l_gt_bc,
                missed_interruption_rate=rate(l_missed_int, l_gt_int),
                missed_interruptions=l_missed_int,
                ground_truth_interruptions=l_gt_int,
                ground_truth_sidetalk=l_gt_st,
            )
        )

    derived = derive_metrics(collector.events)
    checks = evaluate_targets(derived, cfg)
    all_passed = fir <= 0.05 and missed_rate <= 0.05 and all(check.passed for check in checks)
    if validate_balance and (gt_bc == 0 or gt_int == 0 or gt_st == 0):
        all_passed = False

    return ReplayReport(
        results=tuple(results),
        total=total,
        correct=correct,
        accuracy=accuracy,
        false_interruption_rate=fir,
        false_interruption_count=false_int,
        ground_truth_backchannels=gt_bc,
        missed_interruption_rate=missed_rate,
        missed_interruption_count=missed_int,
        ground_truth_interruptions=gt_int,
        ground_truth_sidetalk=gt_st,
        sidetalk_false_interruptions=st_false_int,
        confusion_matrix=confusion,
        per_language=tuple(per_lang),
        derived_metrics=derived,
        target_checks=checks,
        passed=all_passed,
    )


def format_replay_report(report: ReplayReport, title: str = "OVERLAP CORPUS REPLAY REPORT") -> str:
    """Format ReplayReport into a human- and machine-readable table with PASS/FAIL."""
    lines: list[str] = [
        "=" * 78,
        title,
        "=" * 78,
        f"Total Cases:               {report.total}",
        f"Correct Predictions:       {report.correct} ({report.accuracy * 100.0:.2f}%)",
        (
            f"False Interruption Rate:   {report.false_interruption_rate * 100.0:.2f}% "
            f"({report.false_interruption_count}/{report.ground_truth_backchannels}) "
            "[Target <= 5.0%]"
        ),
        (
            f"Missed Interruption Rate:  {report.missed_interruption_rate * 100.0:.2f}% "
            f"({report.missed_interruption_count}/{report.ground_truth_interruptions}) "
            "[Target <= 5.0%]"
        ),
        (
            f"Side-Talk False Interrupt: "
            f"{report.sidetalk_false_interruptions}/{report.ground_truth_sidetalk}"
        ),
        "",
        "Confusion Matrix (rows = ground truth, cols = predicted):",
        f"  {'Ground Truth':<15} | {'BACKCHANNEL':>12} | {'INTERRUPT':>10} | {'SIDE_TALK':>10}",
        "  " + "-" * 56,
    ]
    for gt_label in ("BACKCHANNEL", "INTERRUPT", "SIDE_TALK"):
        row = report.confusion_matrix[gt_label]
        lines.append(
            f"  {gt_label:<15} | {row['BACKCHANNEL']:>12} | "
            f"{row['INTERRUPT']:>10} | {row['SIDE_TALK']:>10}"
        )

    lines.extend(
        [
            "",
            "Per-Language Breakdown:",
            (
                f"  {'Lang':<6} | {'Total':>5} | {'Acc %':>7} | "
                f"{'FIR %':>7} | {'Missed %':>8} | {'BC/INT/ST':>11}"
            ),
            "  " + "-" * 58,
        ]
    )
    for lm in report.per_language:
        counts_str = (
            f"{lm.ground_truth_backchannels}/"
            f"{lm.ground_truth_interruptions}/"
            f"{lm.ground_truth_sidetalk}"
        )
        lines.append(
            f"  {lm.lang:<6} | {lm.total:>5} | {lm.accuracy * 100.0:>6.1f}% | "
            f"{lm.fir * 100.0:>6.1f}% | {lm.missed_interruption_rate * 100.0:>7.1f}% | "
            f"{counts_str:>11}"
        )

    lines.extend(
        [
            "",
            "ARCHITECTURE §12 Target Checks:",
            f"  {'Metric':<26} | {'Observed':<26} | {'Target':<22} | {'Status':<6}",
            "  " + "-" * 86,
        ]
    )
    for chk in report.target_checks:
        status_str = "PASS" if chk.passed else "FAIL"
        lines.append(
            f"  {chk.name:<26} | {chk.observed_str:<26} | {chk.target_str:<22} | {status_str:<6}"
        )

    overall_str = "PASS" if report.passed else "FAIL"
    lines.extend(
        [
            "=" * 78,
            f"OVERALL: {overall_str}",
            "=" * 78,
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m overlap.bench.replay`."""
    parser = argparse.ArgumentParser(
        description="Run offline overlap corpus replay through real Classifier + FSM."
    )
    parser.add_argument(
        "corpus",
        nargs="?",
        default=None,
        help="Optional path to corpus .jsonl file or directory (defaults to bench/corpus/).",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "held_out"),
        default="all",
        help="Which deterministic split to evaluate (default: all).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    corpus_path = Path(args.corpus) if args.corpus is not None else None
    cases = load_corpus(corpus_path, validate=(args.split == "all"))
    if args.split != "all":
        cases = tuple(c for c in cases if c.split == args.split)

    report = replay_corpus(cases)
    title = f"OVERLAP CORPUS REPLAY REPORT (split={args.split}, n={len(cases)})"
    print(format_replay_report(report, title=title))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
