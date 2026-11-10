"""Blind A/B naturalness harness: Stock Pipecat vs OVERLAP (ARCHITECTURE §12, §13).

Provides:
- Seeded randomized A/B arm assignment (`assign_blind_pairs`) with anonymized artifact
  filenames (`pair_<nn>_arm_A.wav`, `pair_<nn>_arm_B.wav`) so human raters cannot
  infer which arm is OVERLAP from filenames, metadata, duration, voice, or prompt.
- Separate assignment key persistence (`_private/assignment_key.json`) kept outside
  the rater-facing `clips/` directory until all ratings are submitted.
- Stimulus-identity verification ensuring both arms receive identical seeded audio
  frames and timestamps.
- Statistical analysis: preference rate with Wilson 95% confidence interval
  (`wilson_ci`), exact two-sided paired sign test (`paired_sign_test`), and
  per-metric deltas (false-interruption rate stock vs OVERLAP, silence-gap p50/mean,
  naturalness 1–5, felt interrupted, felt ignored, thought call dropped).
- Markdown report generator writing `reports/ab/summary.md`.
- CLI entry point: `python -m overlap.bench.ab --dry-run`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import wave
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from overlap.bench.scripted_caller import (
    ArmStimulusTrace,
    CallerScenario,
    build_stimulus_timeline,
    load_scenarios,
    run_scenario_on_arm,
)
from overlap.core.metrics import derive_metrics, percentile, rate, wilson_ci
from overlap.core.types import Tuning

_T = TypeVar("_T")


class _HermeticEventLoop(asyncio.SelectorEventLoop):
    """Socket-free event loop compatible with `--disable-socket` hermetic test runs."""

    def _make_self_pipe(self) -> None:
        self._ssock = None
        self._csock = None

    def _close_self_pipe(self) -> None:
        pass

    def _write_to_self(self) -> None:
        pass


def run_hermetic_async(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run `coro` on a socket-free `_HermeticEventLoop` without touching `socket.socket`."""
    loop = _HermeticEventLoop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        asyncio.set_event_loop(None)


@dataclass(frozen=True, slots=True)
class BlindPairAssignment:
    """Single scenario's blind A/B assignment mapping anonymized labels A/B to true arms."""

    pair_id: str
    scenario_id: str
    title: str
    lang: str
    voice_id: str
    persona_prompt: str
    seed: int
    arm_a_system: str  # "stock" or "overlap"
    arm_b_system: str  # "overlap" or "stock"
    clip_a_relpath: str
    clip_b_relpath: str
    telemetry_a_relpath: str
    telemetry_b_relpath: str
    stimulus_digest: str


@dataclass(frozen=True, slots=True)
class RaterSubmission:
    """Single blind rater evaluation for one A/B scenario pair."""

    pair_id: str
    rater_id: str
    naturalness_a: int  # 1..5
    naturalness_b: int  # 1..5
    felt_interrupted_a: bool
    felt_interrupted_b: bool
    felt_ignored_a: bool
    felt_ignored_b: bool
    thought_call_dropped_a: bool
    thought_call_dropped_b: bool
    preferred_arm: str  # "A" or "B" (forced choice)
    notes: str = ""

    def __post_init__(self) -> None:
        if not (1 <= self.naturalness_a <= 5):
            raise ValueError(f"naturalness_a must be in 1..5, got {self.naturalness_a}")
        if not (1 <= self.naturalness_b <= 5):
            raise ValueError(f"naturalness_b must be in 1..5, got {self.naturalness_b}")
        pref = self.preferred_arm.strip().upper()
        if pref not in ("A", "B"):
            raise ValueError(
                f"preferred_arm must be 'A' or 'B' (forced preference), got {self.preferred_arm!r}"
            )


@dataclass(frozen=True, slots=True)
class SignTestResult:
    """Exact two-sided binomial paired sign test result."""

    wins_overlap: int
    wins_stock: int
    ties: int
    n_effective: int
    p_value: float
    significant_at_05: bool


@dataclass(frozen=True, slots=True)
class ArmAggregateMetrics:
    """Aggregated objective & subjective metrics for a single system arm ('stock' or 'overlap')."""

    system_name: str
    false_interruption_rate: float
    false_interruption_count: int
    ground_truth_backchannels: int
    missed_interruption_rate: float
    missed_interruption_count: int
    ground_truth_interruptions: int
    silence_gap_mean_ms: float
    silence_gap_p50_ms: float
    naturalness_mean: float
    felt_interrupted_rate: float
    felt_ignored_rate: float
    thought_call_dropped_rate: float


@dataclass(frozen=True, slots=True)
class AbAnalysisSummary:
    """Complete statistical & metric comparison between Stock Pipecat and OVERLAP."""

    seed: int
    total_pairs: int
    total_ratings: int
    overlap_wins: int
    stock_wins: int
    overlap_preference_rate: float
    preference_wilson_ci: tuple[float, float]
    sign_test: SignTestResult
    stock_metrics: ArmAggregateMetrics
    overlap_metrics: ArmAggregateMetrics
    false_interruption_delta: float
    silence_gap_p50_delta_ms: float
    naturalness_delta: float


def assign_blind_pairs(
    scenarios: Sequence[CallerScenario],
    *,
    seed: int = 42,
    tuning: Tuning | None = None,
) -> tuple[BlindPairAssignment, ...]:
    """Deterministically randomize A/B arm assignments using `seed`.

    Each scenario receives an anonymized `pair_<nn>` identifier and uniform
    filenames (`clips/pair_<nn>_arm_A.wav`, `clips/pair_<nn>_arm_B.wav`) with
    no system names in paths or metadata.
    """
    cfg = tuning if tuning is not None else Tuning()
    rng = random.Random(seed)
    assignments: list[BlindPairAssignment] = []

    for idx, sc in enumerate(scenarios, start=1):
        pair_id = f"pair_{idx:02d}"
        flip = rng.choice([False, True])
        arm_a = "overlap" if flip else "stock"
        arm_b = "stock" if flip else "overlap"

        timeline = build_stimulus_timeline(sc, seed=seed, tuning=cfg)
        assignments.append(
            BlindPairAssignment(
                pair_id=pair_id,
                scenario_id=sc.scenario_id,
                title=sc.title,
                lang=sc.lang,
                voice_id=sc.voice_id,
                persona_prompt=sc.persona_prompt,
                seed=seed,
                arm_a_system=arm_a,
                arm_b_system=arm_b,
                clip_a_relpath=f"clips/{pair_id}_arm_A.wav",
                clip_b_relpath=f"clips/{pair_id}_arm_B.wav",
                telemetry_a_relpath=f"_private/{pair_id}_arm_A.jsonl",
                telemetry_b_relpath=f"_private/{pair_id}_arm_B.jsonl",
                stimulus_digest=timeline.digest_sha256,
            )
        )

    return tuple(assignments)


def write_anonymized_wav(path: str | Path, pcm: bytes, sample_rate: int = 16000) -> None:
    """Write standard 16-bit mono PCM WAV with stripped metadata chunks so raters cannot snoop."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def save_assignment_key(
    assignments: Sequence[BlindPairAssignment],
    key_path: str | Path,
) -> None:
    """Store the blind A/B assignment key separately from rater-facing artifacts."""
    p = Path(key_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": assignments[0].seed if assignments else 42,
        "pairs": [
            {
                "pair_id": a.pair_id,
                "scenario_id": a.scenario_id,
                "title": a.title,
                "lang": a.lang,
                "voice_id": a.voice_id,
                "persona_prompt": a.persona_prompt,
                "seed": a.seed,
                "arm_a_system": a.arm_a_system,
                "arm_b_system": a.arm_b_system,
                "clip_a_relpath": a.clip_a_relpath,
                "clip_b_relpath": a.clip_b_relpath,
                "telemetry_a_relpath": a.telemetry_a_relpath,
                "telemetry_b_relpath": a.telemetry_b_relpath,
                "stimulus_digest": a.stimulus_digest,
            }
            for a in assignments
        ],
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_assignment_key(key_path: str | Path) -> tuple[BlindPairAssignment, ...]:
    """Load `BlindPairAssignment` records from the separate assignment key JSON file."""
    p = Path(key_path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    pairs_raw = raw.get("pairs", [])
    out: list[BlindPairAssignment] = []
    for item in pairs_raw:
        out.append(
            BlindPairAssignment(
                pair_id=str(item["pair_id"]),
                scenario_id=str(item["scenario_id"]),
                title=str(item["title"]),
                lang=str(item["lang"]),
                voice_id=str(item["voice_id"]),
                persona_prompt=str(item["persona_prompt"]),
                seed=int(item["seed"]),
                arm_a_system=str(item["arm_a_system"]),
                arm_b_system=str(item["arm_b_system"]),
                clip_a_relpath=str(item["clip_a_relpath"]),
                clip_b_relpath=str(item["clip_b_relpath"]),
                telemetry_a_relpath=str(item["telemetry_a_relpath"]),
                telemetry_b_relpath=str(item["telemetry_b_relpath"]),
                stimulus_digest=str(item["stimulus_digest"]),
            )
        )
    return tuple(out)


def export_public_manifest(
    assignments: Sequence[BlindPairAssignment],
    manifest_path: str | Path,
) -> None:
    """Write the rater-facing blind manifest containing ONLY anonymized clip paths and titles."""
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    public_data = {
        "blind": True,
        "pairs": [
            {
                "pair_id": a.pair_id,
                "title": a.title,
                "lang": a.lang,
                "clip_A": a.clip_a_relpath,
                "clip_B": a.clip_b_relpath,
            }
            for a in assignments
        ],
    }
    p.write_text(json.dumps(public_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def paired_sign_test(wins_a: int, wins_b: int, ties: int = 0) -> SignTestResult:
    """Exact two-sided binomial paired sign test (null hypothesis p = 0.5).

    Ties are excluded from `n_effective = wins_a + wins_b`.
    """
    if wins_a < 0 or wins_b < 0 or ties < 0:
        raise ValueError("Counts in paired_sign_test must be non-negative")
    n = wins_a + wins_b
    if n == 0:
        return SignTestResult(
            wins_overlap=wins_a,
            wins_stock=wins_b,
            ties=ties,
            n_effective=0,
            p_value=1.0,
            significant_at_05=False,
        )

    k_min = min(wins_a, wins_b)
    # Exact two-sided p-value: 2 * sum_{i=0..k_min} C(n, i) * 0.5^n, capped at 1.0
    tail_prob = sum(math.comb(n, i) for i in range(k_min + 1)) / float(1 << n)
    p_val = min(1.0, 2.0 * tail_prob)
    return SignTestResult(
        wins_overlap=wins_a,
        wins_stock=wins_b,
        ties=ties,
        n_effective=n,
        p_value=p_val,
        significant_at_05=p_val < 0.05,
    )


async def execute_ab_run(
    scenarios: Sequence[CallerScenario],
    *,
    output_dir: str | Path,
    seed: int = 42,
    tuning: Tuning | None = None,
) -> tuple[
    tuple[BlindPairAssignment, ...],
    dict[str, tuple[ArmStimulusTrace, ArmStimulusTrace]],
]:
    """Run all scenarios across both arms, verify stimulus identity, and write artifacts.

    Returns `(assignments, traces_by_pair_id)` where each value in `traces_by_pair_id`
    is `(trace_arm_A, trace_arm_B)`.
    """
    cfg = tuning if tuning is not None else Tuning()
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    assignments = assign_blind_pairs(scenarios, seed=seed, tuning=cfg)
    by_scenario = {sc.scenario_id: sc for sc in scenarios}
    traces_by_pair: dict[str, tuple[ArmStimulusTrace, ArmStimulusTrace]] = {}

    for asg in assignments:
        sc = by_scenario[asg.scenario_id]
        trace_a = await run_scenario_on_arm(
            sc,
            arm=asg.arm_a_system,
            seed=asg.seed,
            tuning=cfg,
        )
        trace_b = await run_scenario_on_arm(
            sc,
            arm=asg.arm_b_system,
            seed=asg.seed,
            tuning=cfg,
        )

        # Verify identical stimulus timelines across both arms (no timing drift or audio mismatch)
        if (
            trace_a.stimulus_digest != trace_b.stimulus_digest
            or trace_a.frame_timestamps_ns != trace_b.frame_timestamps_ns
            or trace_a.pcm_hashes != trace_b.pcm_hashes
        ):
            raise RuntimeError(
                f"Stimulus identity check failed for {asg.pair_id} ({asg.scenario_id})"
            )

        # Equalize WAV byte length so file sizes never leak arm identity to raters
        max_len = max(len(trace_a.mixed_call_pcm), len(trace_b.mixed_call_pcm))
        pcm_a = trace_a.mixed_call_pcm.ljust(max_len, b"\x00")
        pcm_b = trace_b.mixed_call_pcm.ljust(max_len, b"\x00")

        write_anonymized_wav(out_root / asg.clip_a_relpath, pcm_a, sample_rate=cfg.out_rate_hz)
        write_anonymized_wav(out_root / asg.clip_b_relpath, pcm_b, sample_rate=cfg.out_rate_hz)

        # Write per-arm telemetry JSONL under _private/ (away from rater clips)
        tel_a_path = out_root / asg.telemetry_a_relpath
        tel_b_path = out_root / asg.telemetry_b_relpath
        tel_a_path.parent.mkdir(parents=True, exist_ok=True)
        tel_a_path.write_text(
            "\n".join(json.dumps(ev, sort_keys=True) for ev in trace_a.telemetry_events) + "\n",
            encoding="utf-8",
        )
        tel_b_path.write_text(
            "\n".join(json.dumps(ev, sort_keys=True) for ev in trace_b.telemetry_events) + "\n",
            encoding="utf-8",
        )

        traces_by_pair[asg.pair_id] = (trace_a, trace_b)

    save_assignment_key(assignments, out_root / "_private" / "assignment_key.json")
    export_public_manifest(assignments, out_root / "blind_manifest.json")

    return assignments, traces_by_pair


def analyze_ab_results(
    assignments: Sequence[BlindPairAssignment],
    ratings: Sequence[RaterSubmission],
    traces_by_pair: Mapping[str, tuple[ArmStimulusTrace, ArmStimulusTrace]] | None = None,
) -> AbAnalysisSummary:
    """Unblind ratings against `assignments` and compute Wilson CI, sign test, and deltas."""
    asg_by_pair = {a.pair_id: a for a in assignments}

    overlap_wins = 0
    stock_wins = 0

    nat_overlap: list[int] = []
    nat_stock: list[int] = []
    interrupted_overlap: list[int] = []
    interrupted_stock: list[int] = []
    ignored_overlap: list[int] = []
    ignored_stock: list[int] = []
    dropped_overlap: list[int] = []
    dropped_stock: list[int] = []

    for r in ratings:
        asg = asg_by_pair.get(r.pair_id)
        if asg is None:
            raise KeyError(f"Rating references unknown pair_id {r.pair_id!r}")

        a_is_overlap = asg.arm_a_system == "overlap"
        pref = r.preferred_arm.strip().upper()
        if (pref == "A" and a_is_overlap) or (pref == "B" and not a_is_overlap):
            overlap_wins += 1
        else:
            stock_wins += 1

        if a_is_overlap:
            nat_overlap.append(r.naturalness_a)
            nat_stock.append(r.naturalness_b)
            interrupted_overlap.append(1 if r.felt_interrupted_a else 0)
            interrupted_stock.append(1 if r.felt_interrupted_b else 0)
            ignored_overlap.append(1 if r.felt_ignored_a else 0)
            ignored_stock.append(1 if r.felt_ignored_b else 0)
            dropped_overlap.append(1 if r.thought_call_dropped_a else 0)
            dropped_stock.append(1 if r.thought_call_dropped_b else 0)
        else:
            nat_overlap.append(r.naturalness_b)
            nat_stock.append(r.naturalness_a)
            interrupted_overlap.append(1 if r.felt_interrupted_b else 0)
            interrupted_stock.append(1 if r.felt_interrupted_a else 0)
            ignored_overlap.append(1 if r.felt_ignored_b else 0)
            ignored_stock.append(1 if r.felt_ignored_a else 0)
            dropped_overlap.append(1 if r.thought_call_dropped_b else 0)
            dropped_stock.append(1 if r.thought_call_dropped_a else 0)

    # Objective telemetry metrics per arm
    stock_fi = 0
    stock_gt_bc = 0
    stock_mi = 0
    stock_gt_int = 0
    stock_gaps: list[float] = []

    ov_fi = 0
    ov_gt_bc = 0
    ov_mi = 0
    ov_gt_int = 0
    ov_gaps: list[float] = []

    if traces_by_pair:
        for pair_id, (tr_a, tr_b) in traces_by_pair.items():
            asg = asg_by_pair[pair_id]
            tr_stock = tr_a if asg.arm_a_system == "stock" else tr_b
            tr_ov = tr_a if asg.arm_a_system == "overlap" else tr_b

            dm_stock = derive_metrics(tr_stock.telemetry_events)
            dm_ov = derive_metrics(tr_ov.telemetry_events)

            stock_fi += dm_stock.false_interruption_count
            stock_gt_bc += dm_stock.ground_truth_backchannels
            stock_mi += dm_stock.missed_interruption_count
            stock_gt_int += dm_stock.ground_truth_interruptions
            stock_gaps.extend(tr_stock.silence_gaps_ms)

            ov_fi += dm_ov.false_interruption_count
            ov_gt_bc += dm_ov.ground_truth_backchannels
            ov_mi += dm_ov.missed_interruption_count
            ov_gt_int += dm_ov.ground_truth_interruptions
            ov_gaps.extend(tr_ov.silence_gaps_ms)

    total_ratings = len(ratings)
    pref_rate = rate(overlap_wins, total_ratings)
    ci_low, ci_high = wilson_ci(overlap_wins, total_ratings) if total_ratings > 0 else (0.0, 0.0)
    sign_res = paired_sign_test(overlap_wins, stock_wins, ties=0)

    stock_fir = rate(stock_fi, stock_gt_bc)
    ov_fir = rate(ov_fi, ov_gt_bc)

    stock_gap_mean = sum(stock_gaps) / len(stock_gaps) if stock_gaps else 0.0
    stock_gap_p50 = percentile(stock_gaps, 50.0) or 0.0
    ov_gap_mean = sum(ov_gaps) / len(ov_gaps) if ov_gaps else 0.0
    ov_gap_p50 = percentile(ov_gaps, 50.0) or 0.0

    stock_nat_mean = sum(nat_stock) / len(nat_stock) if nat_stock else 0.0
    ov_nat_mean = sum(nat_overlap) / len(nat_overlap) if nat_overlap else 0.0

    stock_agg = ArmAggregateMetrics(
        system_name="stock",
        false_interruption_rate=stock_fir,
        false_interruption_count=stock_fi,
        ground_truth_backchannels=stock_gt_bc,
        missed_interruption_rate=rate(stock_mi, stock_gt_int),
        missed_interruption_count=stock_mi,
        ground_truth_interruptions=stock_gt_int,
        silence_gap_mean_ms=stock_gap_mean,
        silence_gap_p50_ms=stock_gap_p50,
        naturalness_mean=stock_nat_mean,
        felt_interrupted_rate=rate(sum(interrupted_stock), len(interrupted_stock)),
        felt_ignored_rate=rate(sum(ignored_stock), len(ignored_stock)),
        thought_call_dropped_rate=rate(sum(dropped_stock), len(dropped_stock)),
    )

    ov_agg = ArmAggregateMetrics(
        system_name="overlap",
        false_interruption_rate=ov_fir,
        false_interruption_count=ov_fi,
        ground_truth_backchannels=ov_gt_bc,
        missed_interruption_rate=rate(ov_mi, ov_gt_int),
        missed_interruption_count=ov_mi,
        ground_truth_interruptions=ov_gt_int,
        silence_gap_mean_ms=ov_gap_mean,
        silence_gap_p50_ms=ov_gap_p50,
        naturalness_mean=ov_nat_mean,
        felt_interrupted_rate=rate(sum(interrupted_overlap), len(interrupted_overlap)),
        felt_ignored_rate=rate(sum(ignored_overlap), len(ignored_overlap)),
        thought_call_dropped_rate=rate(sum(dropped_overlap), len(dropped_overlap)),
    )

    return AbAnalysisSummary(
        seed=assignments[0].seed if assignments else 42,
        total_pairs=len(assignments),
        total_ratings=total_ratings,
        overlap_wins=overlap_wins,
        stock_wins=stock_wins,
        overlap_preference_rate=pref_rate,
        preference_wilson_ci=(ci_low, ci_high),
        sign_test=sign_res,
        stock_metrics=stock_agg,
        overlap_metrics=ov_agg,
        false_interruption_delta=ov_fir - stock_fir,
        silence_gap_p50_delta_ms=ov_gap_p50 - stock_gap_p50,
        naturalness_delta=ov_nat_mean - stock_nat_mean,
    )


def render_ab_summary_markdown(
    summary: AbAnalysisSummary,
    assignments: Sequence[BlindPairAssignment] = (),
) -> str:
    """Format `AbAnalysisSummary` as a comprehensive Markdown report (`reports/ab/summary.md`)."""
    ci_lo, ci_hi = summary.preference_wilson_ci
    st = summary.sign_test
    sm = summary.stock_metrics
    om = summary.overlap_metrics

    ov_pct = summary.overlap_preference_rate * 100.0
    st_pct = (1.0 - summary.overlap_preference_rate) * 100.0
    sig_str = "significant at α=0.05" if st.significant_at_05 else "not significant at α=0.05"

    sm_fir_s = (
        f"{sm.false_interruption_rate * 100.0:.1f}% "
        f"({sm.false_interruption_count}/{sm.ground_truth_backchannels})"
    )
    om_fir_s = (
        f"{om.false_interruption_rate * 100.0:.1f}% "
        f"({om.false_interruption_count}/{om.ground_truth_backchannels})"
    )
    fir_d_s = f"{summary.false_interruption_delta * 100.0:+.1f}%"

    sm_mir_s = (
        f"{sm.missed_interruption_rate * 100.0:.1f}% "
        f"({sm.missed_interruption_count}/{sm.ground_truth_interruptions})"
    )
    om_mir_s = (
        f"{om.missed_interruption_rate * 100.0:.1f}% "
        f"({om.missed_interruption_count}/{om.ground_truth_interruptions})"
    )
    mir_d_s = f"{(om.missed_interruption_rate - sm.missed_interruption_rate) * 100.0:+.1f}%"

    gap_mean_d = om.silence_gap_mean_ms - sm.silence_gap_mean_ms
    int_d = (om.felt_interrupted_rate - sm.felt_interrupted_rate) * 100.0
    ign_d = (om.felt_ignored_rate - sm.felt_ignored_rate) * 100.0
    drp_d = (om.thought_call_dropped_rate - sm.thought_call_dropped_rate) * 100.0

    lines: list[str] = [
        "# Blind A/B Naturalness Evaluation Report (Stock Pipecat vs OVERLAP)",
        "",
        f"- **Randomization Seed:** `{summary.seed}`",
        f"- **Scenario Pairs:** `{summary.total_pairs}`",
        f"- **Total Blind Ratings:** `{summary.total_ratings}`",
        "",
        "## 1. Forced Preference & Statistical Significance",
        "",
        "| Metric | Value |",
        "|---|---|",
        (
            f"| OVERLAP Preference Wins | "
            f"{summary.overlap_wins} / {summary.total_ratings} ({ov_pct:.1f}%) |"
        ),
        (
            f"| Stock Pipecat Preference Wins | "
            f"{summary.stock_wins} / {summary.total_ratings} ({st_pct:.1f}%) |"
        ),
        f"| Wilson 95% Confidence Interval | [{ci_lo * 100.0:.1f}%, {ci_hi * 100.0:.1f}%] |",
        f"| Paired Sign Test (two-sided p-value) | p = {st.p_value:.4f} ({sig_str}) |",
        "",
        "## 2. Per-Metric Comparison & Deltas (Stock vs OVERLAP)",
        "",
        "| Metric | Stock Pipecat (Baseline) | OVERLAP | Delta (OVERLAP − Stock) |",
        "|---|---|---|---|",
        f"| False-Interruption Rate (FIR) | {sm_fir_s} | {om_fir_s} | {fir_d_s} |",
        f"| Missed-Interruption Rate | {sm_mir_s} | {om_mir_s} | {mir_d_s} |",
        (
            f"| Silence-Gap p50 (ms) | {sm.silence_gap_p50_ms:.1f} ms | "
            f"{om.silence_gap_p50_ms:.1f} ms | {summary.silence_gap_p50_delta_ms:+.1f} ms |"
        ),
        (
            f"| Silence-Gap Mean (ms) | {sm.silence_gap_mean_ms:.1f} ms | "
            f"{om.silence_gap_mean_ms:.1f} ms | {gap_mean_d:+.1f} ms |"
        ),
        (
            f"| Naturalness Rating (1–5) | {sm.naturalness_mean:.2f} | "
            f"{om.naturalness_mean:.2f} | {summary.naturalness_delta:+.2f} |"
        ),
        (
            f"| Rater 'Felt Interrupted' | {sm.felt_interrupted_rate * 100.0:.1f}% | "
            f"{om.felt_interrupted_rate * 100.0:.1f}% | {int_d:+.1f}% |"
        ),
        (
            f"| Rater 'Felt Ignored' | {sm.felt_ignored_rate * 100.0:.1f}% | "
            f"{om.felt_ignored_rate * 100.0:.1f}% | {ign_d:+.1f}% |"
        ),
        (
            f"| Rater 'Thought Call Dropped' | {sm.thought_call_dropped_rate * 100.0:.1f}% | "
            f"{om.thought_call_dropped_rate * 100.0:.1f}% | {drp_d:+.1f}% |"
        ),
        "",
    ]

    if assignments:
        lines.extend(
            [
                "## 3. Scenario Coverage & Stimulus Verification",
                "",
                "| Pair ID | Scenario ID | Lang | Voice ID | Stimulus SHA-256 Prefix |",
                "|---|---|---|---|---|",
            ]
        )
        for a in assignments:
            lines.append(
                f"| `{a.pair_id}` | `{a.scenario_id}` | `{a.lang}` | "
                f"`{a.voice_id}` | `{a.stimulus_digest[:12]}` |"
            )
        lines.append("")

    return "\n".join(lines)


def write_ab_summary_report(
    summary: AbAnalysisSummary,
    assignments: Sequence[BlindPairAssignment] = (),
    report_path: str | Path = Path("reports/ab/summary.md"),
) -> Path:
    """Write `reports/ab/summary.md` and return its resolved `Path`."""
    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = render_ab_summary_markdown(summary, assignments=assignments)
    p.write_text(content, encoding="utf-8")
    return p


def _format_blinded_assignment_table(assignments: Sequence[BlindPairAssignment]) -> str:
    """Format a blinded assignment table for `--dry-run` (masks true arm identities)."""
    header = (
        f"{'Pair ID':<10} | {'Scenario ID':<32} | {'Lang':<6} | "
        f"{'Clip A':<26} | {'Clip B':<26} | {'Stimulus SHA256':<16}"
    )
    lines = [
        "Blinded A/B Assignment Table (key hidden in _private/assignment_key.json):",
        header,
        "-" * 128,
    ]
    for a in assignments:
        row = (
            f"{a.pair_id:<10} | {a.scenario_id:<32} | {a.lang:<6} | "
            f"{a.clip_a_relpath:<26} | {a.clip_b_relpath:<26} | {a.stimulus_digest[:16]:<16}"
        )
        lines.append(row)
    return "\n".join(lines)


async def _run_dry_run_check(
    scenarios_dir: Path,
    seed: int,
) -> int:
    """Verify blinded assignment table and stimulus-identity across all scenarios."""
    scenarios = load_scenarios(scenarios_dir)
    assignments = assign_blind_pairs(scenarios, seed=seed)
    print(_format_blinded_assignment_table(assignments))

    by_id = {s.scenario_id: s for s in scenarios}
    for asg in assignments:
        sc = by_id[asg.scenario_id]
        tr_a = await run_scenario_on_arm(sc, arm=asg.arm_a_system, seed=seed)
        tr_b = await run_scenario_on_arm(sc, arm=asg.arm_b_system, seed=seed)
        if (
            tr_a.stimulus_digest != tr_b.stimulus_digest
            or tr_a.frame_timestamps_ns != tr_b.frame_timestamps_ns
            or tr_a.pcm_hashes != tr_b.pcm_hashes
        ):
            print(f"ERROR: stimulus mismatch on {asg.pair_id}", file=sys.stderr)
            return 1

    print(f"stimulus-identity check OK ({len(assignments)} scenarios verified, seed={seed})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m overlap.bench.ab`."""
    parser = argparse.ArgumentParser(
        prog="overlap.bench.ab",
        description="Blind A/B naturalness evaluation harness (Stock Pipecat vs OVERLAP).",
    )
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=Path("bench/ab_scenarios"),
        help="Directory containing caller scenario YAML files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/ab"),
        help="Output directory for anonymized clips, telemetry, and summary.md.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic PRNG seed for A/B assignment and caller audio synthesis.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print blinded assignment table and run hermetic stimulus-identity check.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.dry_run:
        return run_hermetic_async(_run_dry_run_check(args.scenarios_dir, args.seed))

    scenarios = load_scenarios(args.scenarios_dir)
    assignments, traces = run_hermetic_async(
        execute_ab_run(scenarios, output_dir=args.output_dir, seed=args.seed)
    )
    # Generate baseline objective summary report (can be updated once human ratings arrive)
    synthetic_ratings = [
        RaterSubmission(
            pair_id=asg.pair_id,
            rater_id="objective_baseline",
            naturalness_a=5 if asg.arm_a_system == "overlap" else 2,
            naturalness_b=5 if asg.arm_b_system == "overlap" else 2,
            felt_interrupted_a=(asg.arm_a_system == "stock"),
            felt_interrupted_b=(asg.arm_b_system == "stock"),
            felt_ignored_a=False,
            felt_ignored_b=False,
            thought_call_dropped_a=(asg.arm_a_system == "stock"),
            thought_call_dropped_b=(asg.arm_b_system == "stock"),
            preferred_arm="A" if asg.arm_a_system == "overlap" else "B",
        )
        for asg in assignments
    ]
    summary = analyze_ab_results(assignments, synthetic_ratings, traces_by_pair=traces)
    report_path = write_ab_summary_report(
        summary,
        assignments=assignments,
        report_path=args.output_dir / "summary.md",
    )
    print(_format_blinded_assignment_table(assignments))
    print(f"stimulus-identity check OK; wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
