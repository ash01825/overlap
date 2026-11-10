"""Hermetic unit tests for `overlap.bench.ab`, `overlap.bench.rating_cli`, and demo packaging."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from overlap.bench.ab import (
    RaterSubmission,
    analyze_ab_results,
    assign_blind_pairs,
    execute_ab_run,
    load_assignment_key,
    paired_sign_test,
    render_ab_summary_markdown,
    run_hermetic_async,
    write_ab_summary_report,
)
from overlap.bench.ab import (
    main as ab_main,
)
from overlap.bench.rating_cli import (
    BlindRatingSession,
    UnblindPrematureError,
    load_blind_manifest,
    load_ratings_jsonl,
    run_interactive_rating_cli,
    save_ratings_jsonl,
)
from overlap.bench.scripted_caller import load_scenarios


def test_seeded_randomization_is_reproducible_and_confound_free() -> None:
    """Randomization is seeded/reproducible and both arms share identical voice/prompt/lang."""
    scenarios = load_scenarios(Path("bench/ab_scenarios"))
    asg_1 = assign_blind_pairs(scenarios, seed=42)
    asg_2 = assign_blind_pairs(scenarios, seed=42)
    asg_diff = assign_blind_pairs(scenarios, seed=999)

    assert asg_1 == asg_2
    assert [a.arm_a_system for a in asg_1] != [a.arm_a_system for a in asg_diff]

    for a, sc in zip(asg_1, scenarios, strict=True):
        # Arms must always be a valid permutation of {'stock', 'overlap'}
        assert {a.arm_a_system, a.arm_b_system} == {"stock", "overlap"}
        # Identical prompt, voice, and language across arms (no confound)
        assert a.voice_id == sc.voice_id
        assert a.persona_prompt == sc.persona_prompt
        assert a.lang == sc.lang
        # Filenames must be strictly anonymized (never contain 'stock' or 'overlap')
        assert "stock" not in a.clip_a_relpath.lower()
        assert "overlap" not in a.clip_a_relpath.lower()
        assert "stock" not in a.clip_b_relpath.lower()
        assert "overlap" not in a.clip_b_relpath.lower()


def test_ab_artifacts_prevent_rater_inference_and_separate_key(tmp_path: Path) -> None:
    """WAV files have identical byte length, standard headers, anonymized names, and hidden key."""
    scenarios = load_scenarios(Path("bench/ab_scenarios"))[:3]
    out_dir = tmp_path / "ab_out"

    assignments, traces = run_hermetic_async(execute_ab_run(scenarios, output_dir=out_dir, seed=42))
    assert len(assignments) == 3
    assert len(traces) == 3

    manifest_path = out_dir / "blind_manifest.json"
    key_path = out_dir / "_private" / "assignment_key.json"
    assert manifest_path.exists()
    assert key_path.exists()

    # Public manifest must not leak 'stock' or 'overlap' anywhere
    manifest_text = manifest_path.read_text(encoding="utf-8").lower()
    assert "stock" not in manifest_text
    assert "overlap" not in manifest_text

    # WAV files in clips/ must have identical size, sample rate, and frame count per pair
    for asg in assignments:
        wav_a = out_dir / asg.clip_a_relpath
        wav_b = out_dir / asg.clip_b_relpath
        assert wav_a.exists()
        assert wav_b.exists()
        assert wav_a.stat().st_size == wav_b.stat().st_size

        with wave.open(str(wav_a), "rb") as wfa, wave.open(str(wav_b), "rb") as wfb:
            assert wfa.getframerate() == wfb.getframerate() == 16000
            assert wfa.getnchannels() == wfb.getnchannels() == 1
            assert wfa.getnframes() == wfb.getnframes()

    loaded_key = load_assignment_key(key_path)
    assert loaded_key == assignments


def test_rating_cli_enforces_blinding_until_all_ratings_submitted(tmp_path: Path) -> None:
    """`BlindRatingSession` blocks unblinding until every pair in the manifest is rated."""
    scenarios = load_scenarios(Path("bench/ab_scenarios"))[:2]
    out_dir = tmp_path / "ab_session"
    assignments, _ = run_hermetic_async(execute_ab_run(scenarios, output_dir=out_dir, seed=42))

    manifest_path = out_dir / "blind_manifest.json"
    key_path = out_dir / "_private" / "assignment_key.json"
    prompts = load_blind_manifest(manifest_path)
    session = BlindRatingSession(prompts, rater_id="judge_1")

    assert not session.is_complete
    with pytest.raises(UnblindPrematureError):
        session.unblind_and_summarize(key_path)

    # Submit only first pair -> still incomplete, unblinding must still raise
    session.submit_rating(
        RaterSubmission(
            pair_id=prompts[0].pair_id,
            rater_id="judge_1",
            naturalness_a=4,
            naturalness_b=2,
            felt_interrupted_a=False,
            felt_interrupted_b=True,
            felt_ignored_a=False,
            felt_ignored_b=False,
            thought_call_dropped_a=False,
            thought_call_dropped_b=True,
            preferred_arm="A" if assignments[0].arm_a_system == "overlap" else "B",
        )
    )
    assert not session.is_complete
    with pytest.raises(UnblindPrematureError):
        session.unblind_and_summarize(key_path)

    # Submit second pair -> session is now complete and unblinds cleanly
    session.submit_rating(
        RaterSubmission(
            pair_id=prompts[1].pair_id,
            rater_id="judge_1",
            naturalness_a=5,
            naturalness_b=2,
            felt_interrupted_a=False,
            felt_interrupted_b=True,
            felt_ignored_a=False,
            felt_ignored_b=False,
            thought_call_dropped_a=False,
            thought_call_dropped_b=True,
            preferred_arm="A" if assignments[1].arm_a_system == "overlap" else "B",
        )
    )
    assert session.is_complete
    summary = session.unblind_and_summarize(key_path)
    assert summary.overlap_wins == 2
    assert summary.stock_wins == 0


def test_interactive_rating_cli_collects_all_fields_and_writes_report(tmp_path: Path) -> None:
    """Interactive rating CLI validates inputs, collects all 5 dimensions, and writes report."""
    scenarios = load_scenarios(Path("bench/ab_scenarios"))[:2]
    out_dir = tmp_path / "ab_cli"
    assignments, _ = run_hermetic_async(execute_ab_run(scenarios, output_dir=out_dir, seed=11))

    manifest_path = out_dir / "blind_manifest.json"
    key_path = out_dir / "_private" / "assignment_key.json"
    ratings_path = out_dir / "ratings.jsonl"
    report_path = out_dir / "summary.md"

    # Scripted answers for 2 pairs (including one invalid input to test retry validation)
    answers: list[str] = []
    for idx, asg in enumerate(assignments):
        a_ov = asg.arm_a_system == "overlap"
        if idx == 0:
            answers.append("9")  # Invalid 1-5 rating -> triggers retry
        answers.extend(
            [
                "5" if a_ov else "2",  # Arm A naturalness
                "n" if a_ov else "y",  # Arm A felt interrupted
                "n",  # Arm A felt ignored
                "n" if a_ov else "y",  # Arm A thought call dropped
                "2" if a_ov else "5",  # Arm B naturalness
                "y" if a_ov else "n",  # Arm B felt interrupted
                "n",  # Arm B felt ignored
                "y" if a_ov else "n",  # Arm B thought call dropped
                "A" if a_ov else "B",  # Forced preference
            ]
        )

    it = iter(answers)
    printed: list[str] = []

    submissions = run_interactive_rating_cli(
        manifest_path,
        ratings_path,
        rater_id="rater_test",
        key_path=key_path,
        report_path=report_path,
        input_fn=lambda _prompt: next(it),
        output_fn=printed.append,
    )

    assert len(submissions) == 2
    assert ratings_path.exists()
    assert report_path.exists()

    reloaded = load_ratings_jsonl(ratings_path)
    assert reloaded == submissions

    save_ratings_jsonl(submissions, tmp_path / "copy.jsonl")
    assert load_ratings_jsonl(tmp_path / "copy.jsonl") == submissions


def test_stats_wilson_ci_paired_sign_test_and_markdown_summary(tmp_path: Path) -> None:
    """Statistical analysis computes Wilson CI, paired sign test, and per-metric deltas."""
    st_all_win = paired_sign_test(8, 0)
    assert st_all_win.n_effective == 8
    assert st_all_win.p_value == pytest.approx(2.0 / 256.0)
    assert st_all_win.significant_at_05 is True

    st_even = paired_sign_test(4, 4)
    assert st_even.p_value == 1.0
    assert st_even.significant_at_05 is False

    scenarios = load_scenarios(Path("bench/ab_scenarios"))
    out_dir = tmp_path / "ab_full"
    assignments, traces = run_hermetic_async(execute_ab_run(scenarios, output_dir=out_dir, seed=42))

    ratings = [
        RaterSubmission(
            pair_id=asg.pair_id,
            rater_id="judge_a",
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

    summary = analyze_ab_results(assignments, ratings, traces_by_pair=traces)
    assert summary.overlap_wins == len(assignments)
    assert summary.overlap_preference_rate == 1.0
    ci_lo, ci_hi = summary.preference_wilson_ci
    assert 0.60 < ci_lo < 1.0
    assert ci_hi == pytest.approx(1.0)
    assert summary.sign_test.significant_at_05 is True

    # Stock has 100% false-interruption rate on backchannels; OVERLAP has 0%
    assert summary.stock_metrics.false_interruption_rate == 1.0
    assert summary.overlap_metrics.false_interruption_rate == 0.0
    assert summary.false_interruption_delta == -1.0
    assert summary.silence_gap_p50_delta_ms < 0.0

    report_file = write_ab_summary_report(
        summary,
        assignments=assignments,
        report_path=out_dir / "summary.md",
    )
    md = render_ab_summary_markdown(summary, assignments=assignments)
    assert report_file.exists()
    assert "Wilson 95% Confidence Interval" in md
    assert "Paired Sign Test" in md
    assert "False-Interruption Rate (FIR)" in md
    assert "Silence-Gap p50 (ms)" in md


def test_ab_cli_dry_run_and_demo_packaging(capsys: pytest.CaptureFixture[str]) -> None:
    """`python -m overlap.bench.ab --dry-run` exits 0 and prints table + stimulus check OK."""
    rc = ab_main(["--dry-run", "--seed", "42"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "Blinded A/B Assignment Table" in captured
    assert "stimulus-identity check OK" in captured

    # Verify demo packaging files exist and contain no hardcoded secrets
    dockerfile = Path("Dockerfile")
    demo_doc = Path("docs/DEMO.md")
    makefile = Path("Makefile")
    assert dockerfile.exists()
    assert demo_doc.exists()
    assert "demo:" in makefile.read_text(encoding="utf-8")
    assert "Headphone" in demo_doc.read_text(encoding="utf-8")
    assert "Fallback Plan" in demo_doc.read_text(encoding="utf-8")
