"""Pure overlap classifier for full-duplex conversational turn-taking (§7, §10).

Evaluates acoustic, lexical, and temporal features against ordered rules 1-6
(first match wins). Provides early-exit decisions for interruptions, acoustic
priors during STT latency, backchannel recognition, side-talk discrimination,
and a deterministic hard-cap fail-safe (INV-L4).

Architecture invariants:
- Pure stdlib only: zero async, time.time, random, network, or mutable state.
- Invariant INV-L4: Never returns PENDING at or after classify_cap_ms.
- Invariant INV-L11: Call latency <= 0.2 ms p99 (O(tokens), allocation-light).
- Fully deterministic: identical features produce identical verdicts.
"""

from __future__ import annotations

from overlap.core.lexicon import analyze
from overlap.core.types import Label, OverlapFeatures, Tuning, Verdict


def classify(
    features: OverlapFeatures,
    tuning: Tuning | None = None,
    *,
    decided_at_ns: int = 0,
) -> Verdict:
    """Classify overlapping user speech using ordered rules 1-6 (§7.1).

    First match wins:
    1. Stop/contrast lexicon -> INTERRUPT immediately.
    2. Question or >= 2 content tokens, or 1 content token with
       overlap >= content_min_overlap_ms -> INTERRUPT (early exit).
    3. All tokens in backchannel lexicon (<= bc_max_tokens,
       overlap_ms <= bc_max_overlap_ms) -> BACKCHANNEL (confident once
       overlap_ms >= bc_min_ms or speech ended).
    4. No text yet (STT latency): acoustic prior only after classify_min_ms:
       short overlap + speech ended -> BACKCHANNEL; rel_loudness_db >= assertive_db
       and user_speaking -> INTERRUPT; else PENDING.
    5. SIDE_TALK: rel_loudness_db <= sidetalk_db and no lexicon hit and low
       stt_conf -> SIDE_TALK (ignore).
    6. Hard cap (classify_cap_ms): never returns PENDING. Fail-safe policy:
       still speaking with no backchannel-only evidence -> INTERRUPT;
       otherwise BACKCHANNEL. Tuning.failsafe_bias can flip.

    Args:
        features: Acoustic, lexical, and timing features of overlapping speech.
        tuning: Tuning configuration parameters (defaults to Tuning()).
        decided_at_ns: Virtual or system timestamp of decision (nanoseconds).

    Returns:
        Verdict with decision label, confidence, telemetry reason, and timestamp.
    """
    if tuning is None:
        tuning = Tuning()

    # Step 0: Lexical analysis if partial transcript is present
    analysis = None
    if features.partial_text is not None and features.partial_text.strip():
        analysis = analyze(features.partial_text, lang_hint=features.lang)

    has_lexicon_hit = analysis is not None and (
        analysis.stop_hit
        or analysis.contrast_hit
        or analysis.question
        or len(analysis.bc_tokens) > 0
    )

    # Conservative candidate check for Rule 5 (quiet + low STT confidence + no lexicon hit)
    is_sidetalk_candidate = (
        features.rel_loudness_db <= tuning.sidetalk_db
        and not has_lexicon_hit
        and features.stt_conf is not None
        and features.stt_conf < 0.5
    )

    # =========================================================================
    # Rule 1: Stop / contrast lexicon -> INTERRUPT immediately (§7.1.1)
    # =========================================================================
    if analysis is not None:
        if analysis.stop_hit:
            return Verdict(
                label=Label.INTERRUPT,
                confidence=1.0 if features.stt_conf is None else float(features.stt_conf),
                reason="stop_lexicon",
                overlap_ms=features.overlap_ms,
                decided_at_ns=decided_at_ns,
            )
        if analysis.contrast_hit:
            return Verdict(
                label=Label.INTERRUPT,
                confidence=1.0 if features.stt_conf is None else float(features.stt_conf),
                reason="contrast_lexicon",
                overlap_ms=features.overlap_ms,
                decided_at_ns=decided_at_ns,
            )

    # =========================================================================
    # Rule 2: Question or content tokens -> INTERRUPT (§7.1.2)
    # =========================================================================
    if analysis is not None:
        if analysis.question:
            return Verdict(
                label=Label.INTERRUPT,
                confidence=1.0 if features.stt_conf is None else float(features.stt_conf),
                reason="question",
                overlap_ms=features.overlap_ms,
                decided_at_ns=decided_at_ns,
            )
        if not is_sidetalk_candidate:
            if len(analysis.content_tokens) >= 2:
                return Verdict(
                    label=Label.INTERRUPT,
                    confidence=1.0 if features.stt_conf is None else float(features.stt_conf),
                    reason="content_tokens",
                    overlap_ms=features.overlap_ms,
                    decided_at_ns=decided_at_ns,
                )
            if (
                len(analysis.content_tokens) == 1
                and features.overlap_ms >= tuning.content_min_overlap_ms
            ):
                return Verdict(
                    label=Label.INTERRUPT,
                    confidence=0.9 if features.stt_conf is None else float(features.stt_conf),
                    reason="content_min_overlap",
                    overlap_ms=features.overlap_ms,
                    decided_at_ns=decided_at_ns,
                )

    # =========================================================================
    # Rule 3: Pure backchannel lexicon (§7.1.3)
    # =========================================================================
    if analysis is not None and len(analysis.tokens) > 0 and analysis.is_pure_backchannel:
        # Escalation: pure backchannel with long overlap (> bc_max_overlap_ms)
        if features.overlap_ms > tuning.bc_max_overlap_ms:
            return Verdict(
                label=Label.INTERRUPT,
                confidence=0.9,
                reason="bc_max_overlap_exceeded",
                overlap_ms=features.overlap_ms,
                decided_at_ns=decided_at_ns,
            )
        # Escalation: too many backchannel repetitions (> bc_max_tokens)
        if len(analysis.tokens) > tuning.bc_max_tokens:
            return Verdict(
                label=Label.INTERRUPT,
                confidence=0.9,
                reason="bc_max_tokens_exceeded",
                overlap_ms=features.overlap_ms,
                decided_at_ns=decided_at_ns,
            )
        # Confident once overlap_ms >= bc_min_ms or speech ended
        if features.overlap_ms >= tuning.bc_min_ms or not features.user_speaking:
            return Verdict(
                label=Label.BACKCHANNEL,
                confidence=0.95 if features.stt_conf is None else float(features.stt_conf),
                reason="backchannel_lexicon",
                overlap_ms=features.overlap_ms,
                decided_at_ns=decided_at_ns,
            )
        # Speech still accumulating before bc_min_ms
        if features.overlap_ms < tuning.classify_cap_ms:
            return Verdict(
                label=Label.PENDING,
                confidence=0.5,
                reason="bc_accumulating",
                overlap_ms=features.overlap_ms,
                decided_at_ns=decided_at_ns,
            )

    # Single content token accumulating before content_min_overlap_ms
    if (
        analysis is not None
        and len(analysis.content_tokens) == 1
        and not is_sidetalk_candidate
        and features.overlap_ms < tuning.content_min_overlap_ms
        and features.overlap_ms < tuning.classify_cap_ms
    ):
        return Verdict(
            label=Label.PENDING,
            confidence=0.5,
            reason="content_accumulating",
            overlap_ms=features.overlap_ms,
            decided_at_ns=decided_at_ns,
        )

    # =========================================================================
    # Rule 4: No text yet (STT partial latency) acoustic prior (§7.1.4)
    # =========================================================================
    if analysis is None or len(analysis.tokens) == 0:
        if features.overlap_ms < tuning.classify_min_ms:
            return Verdict(
                label=Label.PENDING,
                confidence=0.5,
                reason="awaiting_text_or_tick",
                overlap_ms=features.overlap_ms,
                decided_at_ns=decided_at_ns,
            )

        # After classify_min_ms:
        if (
            not features.user_speaking
            and features.overlap_ms <= tuning.bc_max_overlap_ms
            and features.rel_loudness_db < tuning.assertive_db
        ):
            return Verdict(
                label=Label.BACKCHANNEL,
                confidence=0.7,
                reason="acoustic_short_speech_ended",
                overlap_ms=features.overlap_ms,
                decided_at_ns=decided_at_ns,
            )

        if features.rel_loudness_db >= tuning.assertive_db and features.user_speaking:
            return Verdict(
                label=Label.INTERRUPT,
                confidence=0.85,
                reason="acoustic_assertive_speaking",
                overlap_ms=features.overlap_ms,
                decided_at_ns=decided_at_ns,
            )

        if features.overlap_ms < tuning.classify_cap_ms:
            return Verdict(
                label=Label.PENDING,
                confidence=0.5,
                reason="acoustic_pending",
                overlap_ms=features.overlap_ms,
                decided_at_ns=decided_at_ns,
            )

    # =========================================================================
    # Rule 5: SIDE_TALK (§7.1.5)
    # =========================================================================
    if is_sidetalk_candidate:
        return Verdict(
            label=Label.SIDE_TALK,
            confidence=0.75,
            reason="sidetalk_quiet_low_conf",
            overlap_ms=features.overlap_ms,
            decided_at_ns=decided_at_ns,
        )

    # =========================================================================
    # Rule 6: Hard cap (classify_cap_ms) fail-safe policy (§7.1.6, INV-L4)
    # =========================================================================
    if features.overlap_ms >= tuning.classify_cap_ms:
        if features.user_speaking:
            label = tuning.failsafe_bias
            reason = "cap_failsafe_user_speaking"
        else:
            label = Label.BACKCHANNEL
            reason = "cap_failsafe_speech_ended"
        return Verdict(
            label=label,
            confidence=0.6,
            reason=reason,
            overlap_ms=features.overlap_ms,
            decided_at_ns=decided_at_ns,
        )

    # Fallthrough before hard cap
    return Verdict(
        label=Label.PENDING,
        confidence=0.5,
        reason="pending_decision",
        overlap_ms=features.overlap_ms,
        decided_at_ns=decided_at_ns,
    )
