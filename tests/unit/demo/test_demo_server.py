# ruff: noqa: E402
"""Hermetic unit tests for `overlap.demo` server, session orchestration, and UI asset."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

_repo_root = str(Path(__file__).resolve().parents[3])
_helpers_dir = str(Path(__file__).resolve().parents[2] / "helpers")
for _p in (_repo_root, _helpers_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from virtual import run_virtual  # noqa: E402

from overlap.core.clock import VirtualClock  # noqa: E402
from overlap.core.types import FsmState, SttEvent, Tuning  # noqa: E402
from overlap.demo.server import (  # noqa: E402
    DemoSession,
    load_dotenv_if_present,
    load_static_index_html,
)


def _state(session: DemoSession) -> FsmState:
    return session.engine.state


async def _wait_until_speaking(session: DemoSession, clock: VirtualClock) -> None:
    """Drain microtasks and advance virtual time until agent starts SPEAKING."""
    for _ in range(40):
        await asyncio.sleep(0)
        if _state(session) == FsmState.SPEAKING:
            return
        await clock.advance(10)
    assert _state(session) == FsmState.SPEAKING


def test_static_frontend_matches_desktop_constraints() -> None:
    """Verify bundled desktop UI exists and excludes scenario buttons & benchmark clutter."""
    html = load_static_index_html()
    assert "OVERLAP" in html
    assert "STOCK VAD" in html
    assert "Start Mic" in html
    for lang_code in ("auto", "hi", "en", "ta", "te", "kn", "ml", "bn", "mr", "gu", "pa", "or"):
        assert f'value="{lang_code}"' in html
    # Explicitly forbidden per user requirements:
    assert "235/240" not in html
    assert "96.67%" not in html
    assert "Say 'hmm'" not in html
    assert "Scenario" not in html


def test_load_dotenv_does_not_overwrite_existing(tmp_path: Path, monkeypatch: Any) -> None:
    """`.env` loader populates missing variables and preserves pre-set env values."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TEST_OVERLAP_NEW=\"abc123\"\nTEST_OVERLAP_KEEP='from_file'\n# comment\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TEST_OVERLAP_NEW", raising=False)
    monkeypatch.setenv("TEST_OVERLAP_KEEP", "already_set")

    loaded = load_dotenv_if_present(env_file)
    assert loaded == {"TEST_OVERLAP_NEW": "abc123"}


def test_demo_session_backchannel_vs_interrupt_and_stock_mode() -> None:
    """Verify end-to-end OverlapEngine backchannel continuation, interrupt flush, and Stock mode."""

    async def _run() -> None:
        clock = VirtualClock(start_ns=1_000_000_000)
        events: list[dict[str, Any]] = []

        session = DemoSession(
            send_json=lambda payload: events.append(payload),
            lang="hi",
            mode="overlap",
            clock=clock,
            use_live_apis=False,
        )
        await session.start()
        try:
            assert _state(session) == FsmState.LISTENING

            # 1. Start a real LLM -> TTS turn
            await session.handle_user_text("Explain how full duplex works")
            await _wait_until_speaking(session, clock)
            assert session.engine.mixer.current_gain == 1.0

            # 2. Pure backchannel ("haan") during SPEAKING -> ducks, classifies BACKCHANNEL
            flushes_before = session.transport.flush_count
            await session.handle_user_text("haan")
            await asyncio.sleep(0)

            verdicts = [e for e in events if e.get("type") == "verdict"]
            assert any(v.get("label") == "BACKCHANNEL" for v in verdicts), f"verdicts={verdicts}"
            assert session.transport.flush_count == flushes_before
            assert _state(session) == FsmState.SPEAKING
            assert any(
                e.get("type") == "duck.applied" or e.get("name") == "duck.applied" for e in events
            )

            # 3. True interrupt ("rukiye") during SPEAKING -> classifies INTERRUPT, flushes, yields
            await session.handle_user_text("rukiye")
            for _ in range(10):
                await asyncio.sleep(0)
                if _state(session) == FsmState.LISTENING:
                    break
                await clock.advance(20)

            verdicts_after = [e for e in events if e.get("type") == "verdict"]
            assert any(v.get("label") == "INTERRUPT" for v in verdicts_after)
            assert session.transport.flush_count > flushes_before
            assert _state(session) == FsmState.LISTENING
            assistant_cuts = [
                e
                for e in events
                if e.get("type") == "turn"
                and e.get("role") == "assistant"
                and e.get("interrupted") is True
            ]
            assert assistant_cuts and assistant_cuts[-1].get("discarded_text")

            # 4. Toggle to STOCK mode -> even "haan" immediately cuts & flushes on VadOnset
            events.clear()
            session.set_mode("stock")
            await session.handle_user_text("Tell me another point")
            await _wait_until_speaking(session, clock)

            stock_flushes_before = session.transport.flush_count
            await session.handle_user_text("haan")
            for _ in range(10):
                await asyncio.sleep(0)
                if _state(session) == FsmState.LISTENING:
                    break
                await clock.advance(20)

            stock_verdicts = [e for e in events if e.get("type") == "verdict"]
            assert any(v.get("reason") == "stock_naive_vad_stop" for v in stock_verdicts)
            assert session.transport.flush_count > stock_flushes_before
            assert _state(session) == FsmState.LISTENING

            # 5. Language switching across 10 Indic languages + auto mode
            await session.set_language("ta")
            assert session.lang == "ta"
            assert session.engine._lang == "ta"
            await session.set_language("auto")
            assert session.lang == "auto"
            assert session.engine._lang == "hi"
        finally:
            await session.stop()
            assert session.engine._tuning == Tuning()

    run_virtual(_run())


def test_demo_session_ignores_self_echo_and_stale_stt_during_speaking() -> None:
    """Verify speaker bleed / stale STT during SPEAKING never truncates the agent turn."""

    async def _run() -> None:
        clock = VirtualClock(start_ns=1_000_000_000)
        events: list[dict[str, Any]] = []

        session = DemoSession(
            send_json=lambda payload: events.append(payload),
            lang="en",
            mode="overlap",
            clock=clock,
            use_live_apis=False,
        )
        await session.start()
        try:
            await session.handle_user_text("Give me a description of a tiger")
            await _wait_until_speaking(session, clock)
            assert not session.engine._user_speaking

            # Advance beyond the 600 ms startup guard to exercise echo/unvoiced guards directly
            await clock.advance(700)
            for _ in range(5):
                await asyncio.sleep(0)

            events.clear()
            flushes_before = session.transport.flush_count

            # 1. Agent speaker bleed echoing its own fallback words
            session.engine.handle_stt_event(
                SttEvent(
                    kind="final",
                    text="Got it running in Indian English full-duplex mode",
                    lang="en-IN",
                    conf=0.9,
                    ts_ns=clock.now_ns(),
                )
            )
            # 2. Trailing/unrelated unvoiced STT while user is not speaking
            session.engine.handle_stt_event(
                SttEvent(
                    kind="final",
                    text="description of a tiger",
                    lang="en-IN",
                    conf=0.85,
                    ts_ns=clock.now_ns(),
                )
            )
            for _ in range(10):
                await asyncio.sleep(0)
                await clock.advance(20)

            assert session.transport.flush_count == flushes_before
            assert _state(session) == FsmState.SPEAKING
            assert [e for e in events if e.get("type") == "stt"] == []
            assert [e for e in events if e.get("type") == "turn" and e.get("role") == "user"] == []
        finally:
            await session.stop()

    run_virtual(_run())
