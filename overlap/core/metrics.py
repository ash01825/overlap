"""Pure metric math and derivation from telemetry events per ARCHITECTURE §10 & §12.

Pure Python stdlib + numpy only. No asyncio, no I/O, no time.time(), no network.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from overlap.core.types import Label, Tuning


@dataclass(frozen=True, slots=True)
class DerivedMetrics:
    """All derived evaluation metrics per ARCHITECTURE §12."""

    t_duck_ms: tuple[float, ...] = ()
    t_duck_p50_ms: float | None = None
    t_duck_p95_ms: float | None = None

    t_verdict_ms: tuple[float, ...] = ()
    t_verdict_p50_ms: float | None = None
    t_verdict_p95_ms: float | None = None

    t_stop_ms: tuple[float, ...] = ()
    t_stop_p50_ms: float | None = None
    t_stop_p95_ms: float | None = None

    false_interruption_rate: float = 0.0
    false_interruption_count: int = 0
    ground_truth_backchannels: int = 0

    missed_interruption_rate: float = 0.0
    missed_interruption_count: int = 0
    ground_truth_interruptions: int = 0

    bc_timing_ms: tuple[float, ...] = ()
    bc_timing_p50_ms: float | None = None
    bc_timing_min_ms: float | None = None
    bc_timing_max_ms: float | None = None

    bc_rate_per_min: float = 0.0
    bc_count: int = 0
    user_speech_ms: float = 0.0

    truncation_error_words: tuple[int, ...] = ()
    truncation_error_max_words: int = 0
    truncation_error_p95_words: float | None = None

    echo_false_duck: int = 0
    stale_events_dropped: int = 0


@dataclass(frozen=True, slots=True)
class MetricCheck:
    """Single target comparison row for evaluation reporting."""

    name: str
    observed_str: str
    target_str: str
    passed: bool


def percentile(values: Sequence[float | int], q: float) -> float | None:
    """Compute linear-interpolated percentile q in [0, 100] on a sequence of values.

    Returns None on empty sequence, exact value on single-element sequence.
    Raises ValueError if q is outside [0, 100].
    """
    if q < 0.0 or q > 100.0:
        raise ValueError(f"Percentile q must be in [0, 100], got {q}")
    if not values:
        return None
    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]

    pos = (q / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def rate(numerator: int | float, denominator: int | float) -> float:
    """Safe rate computation returning 0.0 when denominator <= 0."""
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Compute Wilson score confidence interval (low, high) in [0.0, 1.0].

    Returns (0.0, 0.0) when total <= 0.
    Raises ValueError if successes < 0 or successes > total.
    """
    if total <= 0:
        return (0.0, 0.0)
    if successes < 0 or successes > total:
        raise ValueError(f"Invalid successes={successes} for total={total}")

    n = float(total)
    p = float(successes) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half_width = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom

    low = max(0.0, min(1.0, center - half_width))
    high = max(0.0, min(1.0, center + half_width))
    return (low, high)


def _extract_event(ev: Any) -> tuple[str, int, dict[str, Any]]:
    """Normalize an event dict or object into (name, ts_ns, fields_dict)."""
    if isinstance(ev, Mapping):
        name = str(ev.get("name") or ev.get("event") or "")
        ts_ns = int(ev.get("ts_ns", 0))
        if "fields" in ev and isinstance(ev["fields"], Mapping):
            fields = dict(ev["fields"])
            for k, v in ev.items():
                if k not in ("name", "event", "ts_ns", "fields"):
                    fields.setdefault(k, v)
        else:
            fields = {k: v for k, v in ev.items() if k not in ("name", "event", "ts_ns")}
        return (name, ts_ns, fields)

    name = str(getattr(ev, "name", None) or getattr(ev, "event", "") or "")
    ts_ns = int(getattr(ev, "ts_ns", 0))
    raw_fields = getattr(ev, "fields", None)
    if isinstance(raw_fields, Mapping):
        fields = dict(raw_fields)
    else:
        fields = {}
    return (name, ts_ns, fields)


def _normalize_label(val: Any) -> str | None:
    """Normalize Label enum or string to uppercase label string."""
    if val is None:
        return None
    if isinstance(val, Label):
        return val.value
    s = str(val).strip().upper()
    if s.startswith("LABEL."):
        s = s.split(".", 1)[1]
    return s if s else None


def derive_metrics(events: Sequence[Any]) -> DerivedMetrics:
    """Derive all ARCHITECTURE §12 evaluation metrics from a sequence of telemetry events.

    Stale-epoch events (`stale=True` or `epoch < max_seen_epoch`) are filtered out
    and counted in `stale_events_dropped`.
    """
    t_duck_list: list[float] = []
    t_verdict_list: list[float] = []
    t_stop_list: list[float] = []
    bc_timing_list: list[float] = []
    truncation_errors: list[int] = []

    false_interruption_count = 0
    ground_truth_bc = 0
    missed_interruption_count = 0
    ground_truth_int = 0
    echo_false_duck = 0
    bc_count = 0
    stale_dropped = 0

    max_epoch = 0
    last_onset_ns: int | None = None
    last_onset_is_echo: bool = False
    duck_measured_for_onset: bool = False
    verdict_measured_for_onset: bool = False
    stop_measured_for_onset: bool = False
    last_verdict_is_interrupt: bool = False

    last_pause_start_ns: int | None = None
    speech_start_ns: int | None = None
    accumulated_user_speech_ms: float = 0.0
    explicit_user_speech_ms: float = 0.0

    for raw_ev in events:
        name, ts_ns, fields = _extract_event(raw_ev)
        if not name:
            continue

        # Check stale epoch
        if bool(fields.get("stale", False)):
            stale_dropped += 1
            continue

        ev_epoch = fields.get("epoch")
        cur_epoch = fields.get("current_epoch")
        if isinstance(cur_epoch, int) and cur_epoch > max_epoch:
            max_epoch = cur_epoch

        if isinstance(ev_epoch, int):
            if ev_epoch < max_epoch:
                stale_dropped += 1
                continue
            if ev_epoch > max_epoch:
                max_epoch = ev_epoch

        if name in ("fsm.ignored", "supervisor.stale_epoch_dropped"):
            if fields.get("reason") == "stale_epoch" or name == "supervisor.stale_epoch_dropped":
                stale_dropped += 1
            continue

        if "user_speech_ms" in fields and isinstance(fields["user_speech_ms"], (int, float)):
            explicit_user_speech_ms = max(explicit_user_speech_ms, float(fields["user_speech_ms"]))

        if name == "vad.onset":
            is_echo = bool(fields.get("is_echo", False)) or bool(fields.get("self_audio", False))
            tag_str = str(fields.get("tag", ""))
            if "SELF_BACKCHANNEL" in tag_str or "SELF_TTS" in tag_str:
                is_echo = True

            last_onset_ns = ts_ns
            last_onset_is_echo = is_echo
            duck_measured_for_onset = False
            verdict_measured_for_onset = False
            stop_measured_for_onset = False
            last_verdict_is_interrupt = False

            if not is_echo and speech_start_ns is None:
                speech_start_ns = ts_ns

        elif name == "vad.offset":
            last_pause_start_ns = ts_ns
            if speech_start_ns is not None and ts_ns >= speech_start_ns:
                accumulated_user_speech_ms += (ts_ns - speech_start_ns) / 1_000_000.0
                speech_start_ns = None

        elif name == "duck.applied":
            is_echo_duck = bool(fields.get("is_echo", False)) or bool(
                fields.get("self_audio", False)
            )
            tag_str = str(fields.get("tag", ""))
            if "SELF_BACKCHANNEL" in tag_str or "SELF_TTS" in tag_str:
                is_echo_duck = True
            if last_onset_is_echo:
                is_echo_duck = True

            if is_echo_duck:
                echo_false_duck += 1
            else:
                onset_ref = fields.get("onset_ts_ns", last_onset_ns)
                if (
                    isinstance(onset_ref, int)
                    and ts_ns >= onset_ref
                    and not duck_measured_for_onset
                ):
                    t_duck_list.append((ts_ns - onset_ref) / 1_000_000.0)
                    duck_measured_for_onset = True

        elif name == "verdict":
            label = _normalize_label(fields.get("label"))
            if label == "PENDING":
                continue

            onset_ref = fields.get("onset_ts_ns", last_onset_ns)
            if isinstance(onset_ref, int) and ts_ns >= onset_ref and not verdict_measured_for_onset:
                t_verdict_list.append((ts_ns - onset_ref) / 1_000_000.0)
                verdict_measured_for_onset = True

            last_verdict_is_interrupt = label == "INTERRUPT"

            gt = _normalize_label(
                fields.get("ground_truth") or fields.get("expected_label") or fields.get("truth")
            )
            if gt == "BACKCHANNEL":
                ground_truth_bc += 1
                if label == "INTERRUPT":
                    false_interruption_count += 1
            elif gt == "INTERRUPT":
                ground_truth_int += 1
                if label != "INTERRUPT":
                    missed_interruption_count += 1

        elif name == "yield.flush":
            # ARCHITECTURE §12: t_stop is onset -> last agent sample, on real interruption
            # Must NEVER be measured from verdict timestamp.
            onset_ref = fields.get("onset_ts_ns", last_onset_ns)
            is_interruption = bool(fields.get("interruption", True)) and (
                last_verdict_is_interrupt or last_onset_ns is not None
            )
            if (
                is_interruption
                and isinstance(onset_ref, int)
                and ts_ns >= onset_ref
                and not stop_measured_for_onset
            ):
                last_sample_ns = int(fields.get("last_sample_ts_ns", ts_ns))
                if last_sample_ns >= onset_ref:
                    t_stop_list.append((last_sample_ns - onset_ref) / 1_000_000.0)
                    stop_measured_for_onset = True

        elif name == "yield.truncate":
            if "error_words" in fields and isinstance(fields["error_words"], (int, float)):
                truncation_errors.append(abs(int(fields["error_words"])))
            elif "heard_words" in fields and "truth_words" in fields:
                hw = fields["heard_words"]
                tw = fields["truth_words"]
                hw_count = len(hw.split()) if isinstance(hw, str) else int(hw)
                tw_count = len(tw.split()) if isinstance(tw, str) else int(tw)
                truncation_errors.append(abs(hw_count - tw_count))

        elif name == "bc.agent.fire":
            bc_count += 1
            if "pause_start_ns" in fields and isinstance(fields["pause_start_ns"], int):
                last_pause_start_ns = fields["pause_start_ns"]
            if "turn_duration_ms" in fields and isinstance(
                fields["turn_duration_ms"], (int, float)
            ):
                explicit_user_speech_ms = max(
                    explicit_user_speech_ms, float(fields["turn_duration_ms"])
                )

        elif name == "bc.agent.clip_start":
            pause_ref = fields.get("pause_start_ns", last_pause_start_ns)
            if isinstance(pause_ref, int) and ts_ns >= pause_ref:
                bc_timing_list.append((ts_ns - pause_ref) / 1_000_000.0)
            elif "pause_ms" in fields and isinstance(fields["pause_ms"], (int, float)):
                bc_timing_list.append(float(fields["pause_ms"]))

    total_speech_ms = max(accumulated_user_speech_ms, explicit_user_speech_ms)
    speech_minutes = max(total_speech_ms / 60_000.0, 1.0)
    bc_rate_val = rate(bc_count, speech_minutes) if total_speech_ms > 0 else 0.0

    fir = rate(false_interruption_count, ground_truth_bc)
    missed = rate(missed_interruption_count, ground_truth_int)

    trunc_p95 = percentile(truncation_errors, 95.0)
    trunc_max = max(truncation_errors) if truncation_errors else 0

    return DerivedMetrics(
        t_duck_ms=tuple(t_duck_list),
        t_duck_p50_ms=percentile(t_duck_list, 50.0),
        t_duck_p95_ms=percentile(t_duck_list, 95.0),
        t_verdict_ms=tuple(t_verdict_list),
        t_verdict_p50_ms=percentile(t_verdict_list, 50.0),
        t_verdict_p95_ms=percentile(t_verdict_list, 95.0),
        t_stop_ms=tuple(t_stop_list),
        t_stop_p50_ms=percentile(t_stop_list, 50.0),
        t_stop_p95_ms=percentile(t_stop_list, 95.0),
        false_interruption_rate=fir,
        false_interruption_count=false_interruption_count,
        ground_truth_backchannels=ground_truth_bc,
        missed_interruption_rate=missed,
        missed_interruption_count=missed_interruption_count,
        ground_truth_interruptions=ground_truth_int,
        bc_timing_ms=tuple(bc_timing_list),
        bc_timing_p50_ms=percentile(bc_timing_list, 50.0),
        bc_timing_min_ms=min(bc_timing_list) if bc_timing_list else None,
        bc_timing_max_ms=max(bc_timing_list) if bc_timing_list else None,
        bc_rate_per_min=bc_rate_val,
        bc_count=bc_count,
        user_speech_ms=total_speech_ms,
        truncation_error_words=tuple(truncation_errors),
        truncation_error_max_words=trunc_max,
        truncation_error_p95_words=trunc_p95,
        echo_false_duck=echo_false_duck,
        stale_events_dropped=stale_dropped,
    )


def evaluate_targets(
    metrics: DerivedMetrics,
    tuning: Tuning | None = None,
) -> tuple[MetricCheck, ...]:
    """Evaluate DerivedMetrics against ARCHITECTURE §12 targets using Tuning constants."""
    cfg = tuning if tuning is not None else Tuning()

    # 1. t_duck: p95 <= duck_latency_cap_ms (40 ms)
    duck_ok = metrics.t_duck_p95_ms is None or metrics.t_duck_p95_ms <= float(
        cfg.duck_latency_cap_ms
    )
    duck_obs = f"p95={metrics.t_duck_p95_ms:.1f} ms" if metrics.t_duck_p95_ms is not None else "N/A"

    # 2. t_verdict: p95 <= classify_cap_ms (400 ms)
    verdict_ok = metrics.t_verdict_p95_ms is None or metrics.t_verdict_p95_ms <= float(
        cfg.classify_cap_ms
    )
    verdict_obs = (
        f"p95={metrics.t_verdict_p95_ms:.1f} ms" if metrics.t_verdict_p95_ms is not None else "N/A"
    )

    # 3. t_stop: p50 <= 350 ms, p95 <= 450 ms
    stop_p50_limit = 350.0
    stop_p95_limit = float(cfg.classify_cap_ms + cfg.interrupt_stop_cap_ms + 10)
    stop_ok = (metrics.t_stop_p50_ms is None or metrics.t_stop_p50_ms <= stop_p50_limit) and (
        metrics.t_stop_p95_ms is None or metrics.t_stop_p95_ms <= stop_p95_limit
    )
    if metrics.t_stop_p50_ms is not None and metrics.t_stop_p95_ms is not None:
        stop_obs = f"p50={metrics.t_stop_p50_ms:.1f} ms, p95={metrics.t_stop_p95_ms:.1f} ms"
    else:
        stop_obs = "N/A"

    # 4. false_interruption_rate <= 5%
    fir_ok = metrics.false_interruption_rate <= 0.05
    fir_obs = f"{metrics.false_interruption_rate * 100.0:.1f}%"

    # 5. missed_interruption_rate <= 5%
    missed_ok = metrics.missed_interruption_rate <= 0.05
    missed_obs = f"{metrics.missed_interruption_rate * 100.0:.1f}%"

    # 6. bc_timing: 200-450 ms (bc_pause_min_ms .. bc_pause_max_ms + bc_clip_start_ms)
    bc_min_target = float(cfg.bc_pause_min_ms)
    bc_max_target = float(cfg.bc_pause_max_ms + cfg.bc_clip_start_ms)
    if metrics.bc_timing_ms:
        bc_timing_ok = all(bc_min_target <= v <= bc_max_target for v in metrics.bc_timing_ms)
        bc_timing_obs = (
            f"{metrics.bc_timing_min_ms:.1f}–{metrics.bc_timing_max_ms:.1f} ms "
            f"(p50={metrics.bc_timing_p50_ms:.1f} ms)"
        )
    else:
        bc_timing_ok = True
        bc_timing_obs = "N/A"

    # 7. bc_rate <= bc_max_per_min (6 / min)
    bc_rate_ok = metrics.bc_rate_per_min <= float(cfg.bc_max_per_min)
    bc_rate_obs = f"{metrics.bc_rate_per_min:.2f}/min"

    # 8. truncation_error_words <= 1
    trunc_ok = metrics.truncation_error_max_words <= 1
    trunc_obs = (
        f"max={metrics.truncation_error_max_words} words"
        if metrics.truncation_error_words
        else "0 words"
    )

    # 9. echo_false_duck == 0
    echo_ok = metrics.echo_false_duck == 0
    echo_obs = str(metrics.echo_false_duck)

    return (
        MetricCheck(
            name="t_duck",
            observed_str=duck_obs,
            target_str=f"p95 <= {cfg.duck_latency_cap_ms} ms",
            passed=duck_ok,
        ),
        MetricCheck(
            name="t_verdict",
            observed_str=verdict_obs,
            target_str=f"p95 <= {cfg.classify_cap_ms} ms",
            passed=verdict_ok,
        ),
        MetricCheck(
            name="t_stop",
            observed_str=stop_obs,
            target_str=f"p50 <= {int(stop_p50_limit)} ms, p95 <= {int(stop_p95_limit)} ms",
            passed=stop_ok,
        ),
        MetricCheck(
            name="false_interruption_rate",
            observed_str=fir_obs,
            target_str="<= 5.0%",
            passed=fir_ok,
        ),
        MetricCheck(
            name="missed_interruption_rate",
            observed_str=missed_obs,
            target_str="<= 5.0%",
            passed=missed_ok,
        ),
        MetricCheck(
            name="bc_timing",
            observed_str=bc_timing_obs,
            target_str=f"{int(bc_min_target)}–{int(bc_max_target)} ms",
            passed=bc_timing_ok,
        ),
        MetricCheck(
            name="bc_rate",
            observed_str=bc_rate_obs,
            target_str=f"<= {cfg.bc_max_per_min}/min",
            passed=bc_rate_ok,
        ),
        MetricCheck(
            name="truncation_error_words",
            observed_str=trunc_obs,
            target_str="<= 1 word",
            passed=trunc_ok,
        ),
        MetricCheck(
            name="echo_false_duck",
            observed_str=echo_obs,
            target_str="== 0",
            passed=echo_ok,
        ),
    )
