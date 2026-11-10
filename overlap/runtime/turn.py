"""TurnPipeline: ClauseChunker, streaming to TTS, history commit/truncate (§8.3, §10).

Per ARCHITECTURE §8.3, §9, §10:
- ClauseChunker: flushes on [।.?!;:] and on comma if >= min_chars, or max_chars/max_tokens fallback;
  first-clause eager flush on comma; preserves numerals, abbreviations, and Indic word boundaries.
- TurnPipeline: consumes LlmStream deltas, feeds TtsStream per clause, records ClauseAudio
  with word timings and epochs.
- On normal completion commits the full assistant message.
- On YieldCompleted commits truncate_to_heard(...) text only (verified vs FakeTts ground truth).
- MergeUserTurn: cancelled user turn text is prepended to the next final user turn.
- Stale-epoch events and late deltas from cancelled turns are safely dropped.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from overlap.core.clock import CancelToken
from overlap.core.truncate import (
    ClauseAudio,
    HeardText,
    SpeakingRateEstimator,
    truncate_to_heard,
)
from overlap.core.types import (
    Event,
    Tuning,
    WordTiming,
    YieldCompleted,
)
from overlap.ports.llm import LlmPort, Message
from overlap.ports.tts import AudioChunk, TtsPort
from overlap.runtime.supervisor import StreamSupervisor

# Known abbreviations that end in a dot but should NOT trigger clause flushing
COMMON_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "dr.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
        "sr.",
        "jr.",
        "vs.",
        "etc.",
        "i.e.",
        "e.g.",
        "rs.",
        "approx.",
        "st.",
        "co.",
        "a.m.",
        "p.m.",
        "am.",
        "pm.",
        "inc.",
        "ltd.",
    }
)


class ClauseChunker:
    """Incremental clause chunker for streaming LLM completions.

    Flushes on terminal punctuation [।.?!;:], double danda (॥), and comma if >= min_chars.
    Flushes eagerly on the first clause to minimize Time-to-First-Audio (TTFA).
    Falls back to whitespace word boundaries when reaching max_chars or max_tokens,
    preventing unpunctuated Indic text from buffering indefinitely without breaking words.
    """

    def __init__(
        self,
        min_chars: int = 25,
        max_chars: int = 120,
        max_tokens: int = 25,
        first_min_chars: int = 4,
        eager_first: bool = True,
    ) -> None:
        self.min_chars: int = min_chars
        self.max_chars: int = max_chars
        self.max_tokens: int = max_tokens
        self.first_min_chars: int = first_min_chars
        self.eager_first: bool = eager_first

        self._buffer: str = ""
        self._is_first_clause: bool = True

    @property
    def is_first_clause(self) -> bool:
        """True if the chunker is waiting to flush the first clause of a turn."""
        return self._is_first_clause

    def reset(self) -> None:
        """Reset internal buffer and turn state."""
        self._buffer = ""
        self._is_first_clause = True

    def feed(self, delta: str) -> list[str]:
        """Ingest streaming text delta and return 0 or more completed clauses."""
        if not delta:
            return []

        self._buffer += delta
        flushed: list[str] = []

        while self._buffer:
            boundary_idx = self._find_boundary(self._buffer)
            if boundary_idx is None:
                # Check max_chars / max_tokens fallback
                boundary_idx = self._find_fallback_boundary(self._buffer)
                if boundary_idx is None:
                    break

            clause = self._buffer[:boundary_idx].strip()
            self._buffer = self._buffer[boundary_idx:].lstrip()

            if clause:
                flushed.append(clause)
                self._is_first_clause = False

        return flushed

    def flush(self) -> list[str]:
        """Flush any remaining buffered text as the final clause at stream end."""
        remaining = self._buffer.strip()
        self._buffer = ""
        self._is_first_clause = True
        if remaining:
            return [remaining]
        return []

    def _is_mid_numeral(self, text: str, idx: int) -> bool:
        """Return True if character at idx is a period or comma between digits."""
        ch = text[idx]
        if ch not in (".", ","):
            return False
        if idx > 0 and idx + 1 < len(text):
            if text[idx - 1].isdigit() and text[idx + 1].isdigit():
                return True
        return False

    def _is_abbreviation(self, text: str, idx: int) -> bool:
        """Return True if period at idx is part of a known abbreviation."""
        if text[idx] != ".":
            return False

        # Find start of the current token
        token_start = idx
        while token_start > 0 and not text[token_start - 1].isspace():
            token_start -= 1

        token = text[token_start : idx + 1].lower()
        if token in COMMON_ABBREVIATIONS:
            return True

        # Single uppercase initial (e.g. "A.")
        if len(token) == 2 and text[token_start].isupper():
            return True

        return False

    def _find_boundary(self, text: str) -> int | None:
        """Locate earliest complete clause boundary index in text, or None."""
        # Terminal punctuation characters: danda, double danda, dot, question, etc.
        terminal_chars = set("।.?!;:॥")

        i = 0
        n = len(text)
        while i < n:
            ch = text[i]

            if ch in terminal_chars:
                if self._is_mid_numeral(text, i):
                    i += 1
                    continue

                if ch == "." and i + 1 < n and text[i + 1].isalpha():
                    i += 1
                    continue

                if self._is_abbreviation(text, i):
                    i += 1
                    continue

                # Consume consecutive punctuation (e.g. "...", "?!", "!!")
                boundary = i + 1
                while boundary < n and text[boundary] in terminal_chars:
                    boundary += 1

                # Consume trailing quotes or closing brackets
                while boundary < n and text[boundary] in "\"')]}":
                    boundary += 1

                # If boundary reaches end of text without trailing space/newline,
                # wait for next delta unless it is a strong terminal like danda
                if boundary == n and ch not in ("।", "॥", "!", "?"):
                    return None

                return boundary

            if ch == ",":
                if self._is_mid_numeral(text, i):
                    i += 1
                    continue

                boundary = i + 1
                while boundary < n and text[boundary] in "\"')]}":
                    boundary += 1

                # Determine if comma is eligible to flush
                prefix_len = len(text[:i].strip())
                if self._is_first_clause and self.eager_first:
                    eligible = prefix_len >= self.first_min_chars
                else:
                    eligible = prefix_len >= self.min_chars

                if eligible:
                    # Require whitespace or next character confirming boundary
                    if boundary < n:
                        if text[boundary].isspace():
                            return boundary
                    elif boundary == n and text.endswith(" "):
                        return boundary

            i += 1

        return None

    def _find_fallback_boundary(self, text: str) -> int | None:
        """Find whitespace fallback boundary when max_chars or max_tokens is reached."""
        words = text.split()
        if len(text) < self.max_chars and len(words) < self.max_tokens:
            return None

        # Search backwards from max_chars for last whitespace
        search_limit = min(len(text), self.max_chars)
        last_space = text.rfind(" ", 0, search_limit)

        if last_space != -1 and last_space > 0:
            return last_space + 1

        # If no whitespace before max_chars, find first whitespace after
        next_space = text.find(" ", search_limit)
        if next_space != -1:
            return next_space + 1

        return None


@dataclass(frozen=True, slots=True)
class TurnResult:
    """Summary result of a completed or interrupted conversational turn."""

    full_text: str
    heard_text: str
    clauses: tuple[ClauseAudio, ...]
    interrupted: bool
    epoch: int


class TurnPipeline:
    """Coordinates LLM generation, ClauseChunker, TTS streaming, and history commits.

    Implements ARCHITECTURE §8.3:
    - Feeds LLM deltas into ClauseChunker and sends each clause to TtsStream.
    - Records ClauseAudio (text, duration_ms, word timings) for every clause.
    - On normal completion commits the full assistant message.
    - On YieldCompleted commits truncate_to_heard(...) text only.
    - Prepend cancelled user turn text on MergeUserTurn to the next user final.
    - Enforces epoch hygiene: drops stale-epoch events and cancels gracefully.
    """

    def __init__(
        self,
        llm: LlmPort,
        tts: TtsPort,
        supervisor: StreamSupervisor | None = None,
        voice_id: str = "v1",
        lang: str = "hi",
        tuning: Tuning | None = None,
        rate_est: SpeakingRateEstimator | None = None,
        chunker: ClauseChunker | None = None,
        audio_chunk_handler: Callable[[AudioChunk], Any] | None = None,
    ) -> None:
        self._llm: LlmPort = llm
        self._tts: TtsPort = tts
        self._supervisor: StreamSupervisor | None = supervisor
        self._voice_id: str = voice_id
        self._lang: str = lang
        self._tuning: Tuning = tuning or Tuning()
        self._rate_est: SpeakingRateEstimator = rate_est or SpeakingRateEstimator()
        self._chunker: ClauseChunker = chunker or ClauseChunker()
        self._audio_chunk_handler: Callable[[AudioChunk], Any] | None = audio_chunk_handler

        self._history: list[Message] = []
        self._cancelled_user_text: str = ""

        # Active turn state
        self._current_epoch: int = 0
        self._turn_token: CancelToken = CancelToken()
        self._clauses_sent: list[str] = []
        self._recorded_clauses: list[ClauseAudio] = []
        self._committed: bool = False
        self._interrupted: bool = False
        self._played_pos_ms: int = 0

        # Chunk aggregation state
        self._active_clause_idx: int = 0
        self._active_clause_words: list[WordTiming] = []
        self._active_clause_duration_ms: int = 0

        if supervisor is not None:
            self.attach_supervisor(supervisor)

    @property
    def history(self) -> list[Message]:
        """Committed conversation history."""
        return list(self._history)

    @property
    def chunker(self) -> ClauseChunker:
        """Active clause chunker."""
        return self._chunker

    @property
    def recorded_clauses(self) -> list[ClauseAudio]:
        """Recorded clause audio metadata for the active or last turn."""
        return list(self._recorded_clauses)

    def attach_supervisor(self, supervisor: StreamSupervisor) -> None:
        """Wire this pipeline into StreamSupervisor effect executor and event stream."""
        self._supervisor = supervisor
        supervisor._history_truncator = self.handle_truncate_history
        supervisor._merge_turn_handler = self.handle_merge_user_turn

        prev_on_event = supervisor._on_event

        def _intercept_event(event: Event) -> None:
            if isinstance(event, YieldCompleted):
                self.handle_yield_completed()
            if prev_on_event is not None:
                prev_on_event(event)

        supervisor._on_event = _intercept_event

    def add_user_message(self, text: str) -> Message:
        """Commit a user message to history, prepending cancelled text if MergeUserTurn occurred."""
        cleaned = text.strip()
        if self._cancelled_user_text:
            full_text = f"{self._cancelled_user_text} {cleaned}".strip()
            self._cancelled_user_text = ""
        else:
            full_text = cleaned

        msg = Message(role="user", content=full_text)
        self._history.append(msg)
        return msg

    def add_message(self, message: Message) -> None:
        """Append an arbitrary message to conversation history."""
        self._history.append(message)

    def handle_merge_user_turn(self, text: str) -> None:
        """Handler for MergeUserTurn effect: stores text to prepend to next user turn."""
        if text.strip():
            if self._cancelled_user_text:
                self._cancelled_user_text = f"{self._cancelled_user_text} {text.strip()}".strip()
            else:
                self._cancelled_user_text = text.strip()

    def handle_truncate_history(self, played_pos_ms: int) -> None:
        """Handler for TruncateHistory effect: records played audio position."""
        self._played_pos_ms = max(0, played_pos_ms)

    def handle_yield_completed(self) -> HeardText | None:
        """Handler for YieldCompleted: commits truncate_to_heard(...) text only (INV-T1)."""
        if self._committed:
            return None

        self._mark_unstarted_clauses_dropped()

        heard = truncate_to_heard(
            clauses=self._recorded_clauses,
            played_ms=self._played_pos_ms,
            rate_est=self._rate_est,
            marker=self._tuning.truncation_marker,
            tuning=self._tuning,
        )

        if heard.text:
            self._history.append(Message(role="assistant", content=heard.text))

        self._committed = True
        self._interrupted = True
        return heard

    def _handle_audio_chunk(self, chunk: AudioChunk) -> None:
        """Ingest synthesized audio chunk and aggregate word timings into ClauseAudio."""
        if self._current_epoch == 0:
            self._current_epoch = chunk.epoch
        elif chunk.epoch != self._current_epoch:
            return  # Drop stale-epoch audio chunk

        if self._audio_chunk_handler is not None:
            self._audio_chunk_handler(chunk)

        chunk_dur_ms = int((len(chunk.pcm) / (chunk.sample_rate * 2)) * 1000)

        # Check if this chunk belongs to the next clause
        # In Timbre/FakeTts, each clause starts with word timings at start_ms == 0
        if chunk.words and self._active_clause_words and self._active_clause_duration_ms > 0:
            first_timing = chunk.words[0]
            if first_timing.start_ms == 0:
                self._finalize_active_clause()

        self._active_clause_duration_ms += chunk_dur_ms
        self._active_clause_words.extend(chunk.words)

        if chunk.is_last:
            self._finalize_active_clause()

    def _finalize_active_clause(self) -> None:
        """Package currently accumulated chunks into a completed ClauseAudio record."""
        if self._active_clause_duration_ms > 0 or self._active_clause_words:
            if self._active_clause_idx < len(self._clauses_sent):
                clause_text = self._clauses_sent[self._active_clause_idx]
            else:
                clause_text = " ".join(w.word for w in self._active_clause_words)

            clause_audio = ClauseAudio(
                text=clause_text,
                duration_ms=self._active_clause_duration_ms,
                words=tuple(self._active_clause_words),
                lang=self._lang,
                synthesized=True,
            )
            self._recorded_clauses.append(clause_audio)
            self._active_clause_idx += 1
            self._active_clause_words = []
            self._active_clause_duration_ms = 0

    def _mark_unstarted_clauses_dropped(self) -> None:
        """Mark remaining sent clauses as un-synthesized (duration_ms=0, synthesized=False)."""
        self._finalize_active_clause()
        while self._active_clause_idx < len(self._clauses_sent):
            clause_text = self._clauses_sent[self._active_clause_idx]
            self._recorded_clauses.append(
                ClauseAudio(
                    text=clause_text,
                    duration_ms=0,
                    words=(),
                    lang=self._lang,
                    synthesized=False,
                )
            )
            self._active_clause_idx += 1

    async def run_turn(
        self,
        user_text: str | None = None,
        epoch: int = 1,
        cancel: CancelToken | None = None,
    ) -> TurnResult:
        """Execute a full assistant response turn from LLM streaming through TTS synthesis."""
        if user_text is not None:
            self.add_user_message(user_text)

        # Initialize turn accounting
        self._current_epoch = epoch
        self._turn_token = cancel if cancel is not None else CancelToken()
        self._clauses_sent.clear()
        self._recorded_clauses.clear()
        self._committed = False
        self._interrupted = False
        self._played_pos_ms = 0
        self._active_clause_idx = 0
        self._active_clause_words.clear()
        self._active_clause_duration_ms = 0
        self._chunker.reset()

        # Open streams
        llm_stream = await self._llm.stream(self._history, epoch=epoch, cancel=self._turn_token)
        if self._supervisor is not None:
            self._supervisor.register_llm_stream(llm_stream, cancel=self._turn_token)

        tts_stream = await self._tts.open_stream(
            voice_id=self._voice_id,
            lang=self._lang,
            epoch=epoch,
            cancel=self._turn_token,
        )
        if self._supervisor is not None:
            self._supervisor.register_tts_stream(tts_stream, cancel=self._turn_token)

        async def _consume_llm() -> None:
            try:
                async for delta in llm_stream.deltas():
                    if self._turn_token.cancelled:
                        break
                    clauses = self._chunker.feed(delta)
                    for c in clauses:
                        if self._turn_token.cancelled:
                            break
                        self._clauses_sent.append(c)
                        await tts_stream.send_text(c)

                if not self._turn_token.cancelled:
                    final_clauses = self._chunker.flush()
                    for c in final_clauses:
                        if self._turn_token.cancelled:
                            break
                        self._clauses_sent.append(c)
                        await tts_stream.send_text(c)
                    await tts_stream.finish()
            except asyncio.CancelledError:
                pass

        async def _consume_tts() -> None:
            try:
                async for chunk in tts_stream.chunks():
                    if self._turn_token.cancelled:
                        break
                    self._handle_audio_chunk(chunk)
            except asyncio.CancelledError:
                pass

        llm_task = asyncio.create_task(_consume_llm())
        tts_task = asyncio.create_task(_consume_tts())

        # Link cancellation to tasks
        def _cancel_tasks() -> None:
            llm_task.cancel()
            tts_task.cancel()

        self._turn_token.on_cancel(_cancel_tasks)

        await asyncio.gather(llm_task, tts_task, return_exceptions=True)

        self._mark_unstarted_clauses_dropped()

        full_text = " ".join(c.text for c in self._recorded_clauses if c.text.strip())

        # Normal completion commit
        if not self._committed and not self._interrupted and not self._turn_token.cancelled:
            if full_text:
                self._history.append(Message(role="assistant", content=full_text))
            self._committed = True

        heard_content = ""
        for m in reversed(self._history):
            if m.role == "assistant":
                heard_content = m.content
                break

        return TurnResult(
            full_text=full_text,
            heard_text=heard_content,
            clauses=tuple(self._recorded_clauses),
            interrupted=self._interrupted or self._turn_token.cancelled,
            epoch=self._current_epoch,
        )
