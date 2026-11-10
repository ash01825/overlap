"""Unit tests for ClauseChunker and TurnPipeline runtime modules (§8.3, §10, INV-T1).

Tests coverage:
- ClauseChunker flushes on [।.?!;:] and on comma if >= min_chars, or max_chars/max_tokens fallback.
- First-clause eager flush on comma or early boundary.
- Unpunctuated Indic text flushed cleanly without failing or splitting mid-word.
- Numerals and abbreviations preserved without splitting mid-numeral or mid-word.
- Multilingual chunking: Hindi, Tamil, English, Hinglish.
- Pipeline consumes LlmStream deltas, feeds TtsStream per clause,
  records ClauseAudio with word timings and epochs.
- Normal completion commits the full assistant message.
- On YieldCompleted commits truncate_to_heard(...) text only (verified vs FakeTts ground truth).
- MergeUserTurn prepends cancelled user turn text to the next final user turn.
- Stale-epoch events and late deltas from cancelled turns are dropped.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Sequence

from llm import FakeLlm
from tts import FakeTts
from virtual import run_virtual

from overlap.core.clock import CancelToken, VirtualClock
from overlap.core.types import (
    AbortLlm,
    CancelTts,
    FlushPlayout,
    TruncateHistory,
    Tuning,
    WordTiming,
)
from overlap.ports.llm import LlmPort, LlmStream, Message
from overlap.ports.tts import AudioChunk
from overlap.runtime.supervisor import StreamSupervisor
from overlap.runtime.turn import ClauseChunker, TurnPipeline


class ControlledLlmStream(LlmStream):
    """Fake LLM stream that pauses on an event to simulate in-flight generation."""

    def __init__(
        self,
        deltas: list[str],
        gate: asyncio.Event,
        epoch: int = 1,
        cancel: CancelToken | None = None,
    ) -> None:
        self.epoch = epoch
        self._deltas = deque(deltas)
        self._gate = gate
        self._cancel = cancel
        self._aborted = False

        if cancel is not None:
            cancel.on_cancel(self._handle_abort)

    def _handle_abort(self) -> None:
        self._aborted = True
        self._deltas.clear()
        self._gate.set()

    async def deltas(self) -> AsyncIterator[str]:
        while not self._aborted:
            if self._cancel is not None and self._cancel.cancelled:
                break
            if self._deltas:
                d = self._deltas.popleft()
                yield d
                if not self._gate.is_set():
                    await self._gate.wait()
                continue
            break

    async def abort(self) -> None:
        self._handle_abort()


class ControlledLlm(LlmPort):
    """Fake LLM port producing ControlledLlmStream."""

    def __init__(self, deltas: list[str], gate: asyncio.Event) -> None:
        self.deltas = list(deltas)
        self.gate = gate

    async def stream(
        self,
        messages: Sequence[Message],
        epoch: int = 1,
        cancel: CancelToken | None = None,
    ) -> LlmStream:
        return ControlledLlmStream(
            deltas=list(self.deltas),
            gate=self.gate,
            epoch=epoch,
            cancel=cancel,
        )


# =============================================================================
# 1. ClauseChunker Tests
# =============================================================================


def test_clause_chunker_terminal_punctuation() -> None:
    """ClauseChunker flushes immediately on terminal punctuation [।.?!;:]."""
    chunker = ClauseChunker(min_chars=25, max_chars=120)

    # Devanagari Danda
    clauses = chunker.feed("नमस्ते। ")
    assert clauses == ["नमस्ते।"]

    # Double Danda
    clauses = chunker.feed("शुभ प्रभात॥ ")
    assert clauses == ["शुभ प्रभात॥"]

    # Question mark & Exclamation
    clauses = chunker.feed("क्या हाल है? ")
    assert clauses == ["क्या हाल है?"]

    clauses = chunker.feed("बहुत बढ़िया! ")
    assert clauses == ["बहुत बढ़िया!"]

    # Semicolon & Colon
    clauses = chunker.feed("नोट करें; ")
    assert clauses == ["नोट करें;"]

    clauses = chunker.feed("विवरण: ")
    assert clauses == ["विवरण:"]


def test_clause_chunker_first_clause_eager_flush() -> None:
    """First-clause eager flush on comma; subsequent clauses respect min_chars."""
    chunker = ClauseChunker(min_chars=25, first_min_chars=4, eager_first=True)

    # First clause with comma flushes eagerly despite being only 5 characters
    clauses = chunker.feed("Haan, ")
    assert clauses == ["Haan,"]

    # Second clause with comma shorter than min_chars (25) does NOT flush yet
    clauses = chunker.feed("main sun raha, ")
    assert clauses == []

    # Continuing until terminal punctuation flushes
    clauses = chunker.feed("aap batayein. ")
    assert clauses == ["main sun raha, aap batayein."]


def test_clause_chunker_comma_above_min_chars() -> None:
    """Subsequent clause flushes on comma if >= min_chars."""
    chunker = ClauseChunker(min_chars=20, first_min_chars=4, eager_first=True)

    # First clause
    clauses = chunker.feed("Hello! ")
    assert clauses == ["Hello!"]

    # Second clause with >= 20 chars ending in comma flushes
    text = "I have checked your account details, "
    assert len(text.strip()) >= 20
    clauses = chunker.feed(text)
    assert clauses == ["I have checked your account details,"]


def test_clause_chunker_unpunctuated_indic_fallback() -> None:
    """Failure mode: chunker flushes mid-word or never flushes unpunctuated Indic text."""
    chunker = ClauseChunker(max_chars=40, max_tokens=10)

    text = "नमस्ते आज का मौसम बहुत अच्छा है और हम बाहर जा सकते हैं क्योंकि आज कोई काम नहीं है"
    tokens = text.split()

    all_clauses: list[str] = []
    for token in tokens:
        res = chunker.feed(token + " ")
        all_clauses.extend(res)

    all_clauses.extend(chunker.flush())

    assert len(all_clauses) > 1

    # Verify every clause was split strictly on word boundaries (no partial words)
    reconstructed = " ".join(all_clauses)
    assert reconstructed.split() == tokens

    for clause in all_clauses:
        # None of the words should be sliced mid-word
        for w in clause.split():
            assert w in tokens


def test_clause_chunker_numerals_and_abbreviations_no_split() -> None:
    """Failure mode: chunker flushes mid-numeral or mid-abbreviation."""
    chunker = ClauseChunker(min_chars=25)

    # Decimals and comma in numbers
    clauses = chunker.feed("The value of pi is 3.1415 and count is 10,000 in total. ")
    assert clauses == ["The value of pi is 3.1415 and count is 10,000 in total."]

    # Abbreviations
    clauses = chunker.feed("Dr. Sharma and Mr. Patel visited at 9.30 a.m. today. ")
    assert clauses == ["Dr. Sharma and Mr. Patel visited at 9.30 a.m. today."]


def test_clause_chunker_multilingual_coverage() -> None:
    """Verify Hindi, Tamil, English, and Hinglish chunking."""
    chunker = ClauseChunker(min_chars=20, eager_first=True)

    # Hindi
    chunker.reset()
    clauses = chunker.feed("नमस्ते। आज आप कैसे हैं? ")
    assert clauses == ["नमस्ते।", "आज आप कैसे हैं?"]

    # Tamil
    chunker.reset()
    clauses = chunker.feed("வணக்கம். நீங்கள் எப்படி இருக்கிறீர்கள்? ")
    assert clauses == ["வணக்கம்.", "நீங்கள் எப்படி இருக்கிறீர்கள்?"]

    # English
    chunker.reset()
    clauses = chunker.feed("Good morning! How may I assist you today? ")
    assert clauses == ["Good morning!", "How may I assist you today?"]

    # Hinglish
    chunker.reset()
    clauses = chunker.feed("Haan, bilkul! Main aapki madad kar sakta hoon. ")
    assert clauses == ["Haan,", "bilkul!", "Main aapki madad kar sakta hoon."]


def test_clause_chunker_reset_prevents_leakage() -> None:
    """Cancelled turn's buffered deltas are cleared on reset()."""
    chunker = ClauseChunker(min_chars=25)
    chunker.feed("Incomplete start of old turn ")
    chunker.reset()

    clauses = chunker.feed("Fresh new turn. ")
    assert clauses == ["Fresh new turn."]


# =============================================================================
# 2. TurnPipeline Tests
# =============================================================================


def test_turn_pipeline_normal_completion() -> None:
    """Pipeline consumes deltas, feeds TTS, and on normal completion commits full message."""

    async def _test() -> None:
        llm = FakeLlm(deltas=["Haan, ", "main ", "sun raha hoon."])
        tts = FakeTts(chunk_duration_ms=20)
        pipeline = TurnPipeline(llm=llm, tts=tts, voice_id="v1", lang="hi")

        result = await pipeline.run_turn(user_text="Aap sun rahe hain?", epoch=1)

        assert not result.interrupted
        assert result.full_text == "Haan, main sun raha hoon."

        # History contains both user turn and full assistant reply
        assert len(pipeline.history) == 2
        assert pipeline.history[0] == Message(role="user", content="Aap sun rahe hain?")
        assert pipeline.history[1] == Message(role="assistant", content="Haan, main sun raha hoon.")

        # ClauseAudio was recorded
        assert len(result.clauses) >= 1
        assert any(c.synthesized for c in result.clauses)
        for c in result.clauses:
            if c.synthesized:
                assert c.duration_ms > 0
                assert len(c.words) > 0

    run_virtual(_test())


def test_turn_pipeline_interruption_truncates_to_heard_ground_truth() -> None:
    """On YieldCompleted, pipeline commits truncate_to_heard(...) text only (INV-T1)."""
    clock = VirtualClock()

    async def _test() -> None:
        gate = asyncio.Event()
        # 5 words: Haan (0-20ms), main (20-40ms), sun (40-60ms), raha (60-80ms), hoon. (80-100ms)
        llm = ControlledLlm(deltas=["Haan main sun raha hoon. "], gate=gate)
        tts = FakeTts(chunk_duration_ms=20)
        tuning = Tuning(truncation_marker=" …")
        pipeline = TurnPipeline(llm=llm, tts=tts, tuning=tuning, voice_id="v1", lang="hi")

        # Add user message
        pipeline.add_user_message("Aap sun rahe hain?")

        # Simulate playout interruption at 45 ms
        # Midpoints: Haan=10ms, main=30ms, sun=50ms
        # At 45 ms: Haan (10<=45) and main (30<=45) heard; sun (50>45) not heard!
        token = CancelToken()

        turn_task = asyncio.create_task(pipeline.run_turn(epoch=1, cancel=token))
        for _ in range(5):
            await asyncio.sleep(0)

        # Interruption effect triggers
        pipeline.handle_truncate_history(played_pos_ms=45)
        token.cancel()
        pipeline.handle_yield_completed()

        await turn_task

        # Verify history contains ONLY the truncated heard text, never full reply
        assert len(pipeline.history) == 2
        assert pipeline.history[0] == Message(role="user", content="Aap sun rahe hain?")
        assert pipeline.history[1] == Message(role="assistant", content="Haan main …")

        # Verify assistant turn is NOT committed twice
        assert len([m for m in pipeline.history if m.role == "assistant"]) == 1

    run_virtual(_test(), clock=clock)


def test_turn_pipeline_interruption_at_zero_ms() -> None:
    """Interruption before any speech was heard results in empty or omitted assistant turn."""
    clock = VirtualClock()

    async def _test() -> None:
        gate = asyncio.Event()
        llm = ControlledLlm(deltas=["Namaste, kripya batayein."], gate=gate)
        tts = FakeTts(chunk_duration_ms=20)
        pipeline = TurnPipeline(llm=llm, tts=tts, voice_id="v1", lang="hi")

        pipeline.add_user_message("Hello")
        token = CancelToken()
        turn_task = asyncio.create_task(pipeline.run_turn(epoch=1, cancel=token))

        # Interruption at 0 ms before any deltas
        pipeline.handle_truncate_history(played_pos_ms=0)
        token.cancel()
        pipeline.handle_yield_completed()
        await turn_task

        # Only user message in history; nothing heard was spoken
        assistant_msgs = [m for m in pipeline.history if m.role == "assistant"]
        assert len(assistant_msgs) == 0

    run_virtual(_test(), clock=clock)


def test_turn_pipeline_merge_user_turn() -> None:
    """MergeUserTurn support: cancelled user turn text is prepended to next final."""
    pipeline = TurnPipeline(llm=FakeLlm(), tts=FakeTts())

    # User starts saying "Weather kaisa"
    # Barge in / interruption occurs during thinking: MergeUserTurn emitted
    pipeline.handle_merge_user_turn("Weather kaisa")

    # Next final arrives with the remainder
    pipeline.add_user_message("hai aaj Delhi ka?")

    assert len(pipeline.history) == 1
    assert pipeline.history[0].content == "Weather kaisa hai aaj Delhi ka?"


def test_turn_pipeline_stale_epoch_dropped() -> None:
    """AudioChunk with older epoch is dropped and not recorded."""
    pipeline = TurnPipeline(llm=FakeLlm(), tts=FakeTts())

    # Manually ingest chunk from epoch 0 when active epoch is 2
    stale_chunk = AudioChunk(
        pcm=b"\x00" * 40,
        sample_rate=16000,
        epoch=0,
        words=(WordTiming(word="stale", start_ms=0, end_ms=20),),
    )

    pipeline._current_epoch = 2
    pipeline._handle_audio_chunk(stale_chunk)

    # Stale chunk was dropped
    assert len(pipeline.recorded_clauses) == 0


def test_turn_pipeline_supervisor_integration() -> None:
    """TurnPipeline wires seamlessly into StreamSupervisor effect executor."""
    clock = VirtualClock()

    async def _test() -> None:
        sup = StreamSupervisor(clock=clock)
        gate = asyncio.Event()
        pipeline = TurnPipeline(
            llm=ControlledLlm(deltas=["Haan main sun raha hoon. "], gate=gate),
            tts=FakeTts(chunk_duration_ms=20),
            supervisor=sup,
        )

        pipeline.add_user_message("Kya aap sun rahe hain?")
        turn_task = asyncio.create_task(pipeline.run_turn(epoch=1))
        for _ in range(5):
            await asyncio.sleep(0)

        # Supervisor executes YIELDED effects
        sup.execute_effects(
            (
                FlushPlayout(),
                CancelTts(),
                AbortLlm(),
                TruncateHistory(played_pos_ms=45),
            )
        )

        await turn_task

        # YieldCompleted should have committed truncated history
        assert len(pipeline.history) == 2
        assert pipeline.history[1] == Message(role="assistant", content="Haan main …")

    run_virtual(_test(), clock=clock)
