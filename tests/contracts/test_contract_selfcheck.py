"""Self-check tests verifying that contract suites pass valid implementations

and fail (have teeth) when run against broken implementations.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

_helpers_dir = str(Path(__file__).resolve().parent.parent / "helpers")
if _helpers_dir not in sys.path:
    sys.path.insert(0, _helpers_dir)

_contracts_dir = str(Path(__file__).resolve().parent)
if _contracts_dir not in sys.path:
    sys.path.insert(0, _contracts_dir)

from suites import (  # noqa: E402
    LlmContract,
    SttContract,
    TelemetryContract,
    TransportContract,
    TtsContract,
    VadContract,
)
from virtual import run_virtual  # noqa: E402

from overlap.core.clock import CancelToken  # noqa: E402
from overlap.core.types import AudioFrame, SttEvent  # noqa: E402
from overlap.ports.llm import Message  # noqa: E402
from overlap.ports.tts import AudioChunk, AudioClip  # noqa: E402
from overlap.ports.vad import VadResult  # noqa: E402

# =============================================================================
# Valid Dummy Implementations
# =============================================================================


class ValidSttStream:
    def __init__(self, cancel: CancelToken) -> None:
        self._cancel = cancel
        self._closed = False
        self._buffer: list[AudioFrame] = []
        self._max_buffer = 10

    async def send_audio(self, frame: AudioFrame) -> None:
        if self._closed or self._cancel.cancelled:
            return
        # Drop oldest on overflow
        if len(self._buffer) >= self._max_buffer:
            self._buffer.pop(0)
        self._buffer.append(frame)

    async def events(self) -> AsyncIterator[SttEvent]:
        if self._closed or self._cancel.cancelled:
            return
        yield SttEvent(kind="partial", text="test", ts_ns=1000)
        if self._closed or self._cancel.cancelled:
            return
        yield SttEvent(kind="final", text="test done", ts_ns=2000)

    async def close(self) -> None:
        self._closed = True


class ValidSttPort:
    async def open(self, *, lang_hint: str | None = None, cancel: CancelToken) -> ValidSttStream:
        return ValidSttStream(cancel)


class ValidTtsStream:
    def __init__(self, epoch: int, cancel: CancelToken) -> None:
        self._epoch = epoch
        self._cancel = cancel
        self._cancelled = False
        self._finished = False
        self._text = ""

    async def send_text(self, clause: str, *, ssml: bool = False) -> None:
        self._text += clause

    async def finish(self) -> None:
        self._finished = True

    async def chunks(self) -> AsyncIterator[AudioChunk]:
        if self._cancelled or self._cancel.cancelled:
            return
        yield AudioChunk(
            pcm=b"\x00" * 320,
            sample_rate=16000,
            epoch=self._epoch,
            is_last=True,
        )

    async def cancel(self) -> None:
        self._cancelled = True


class ValidTtsPort:
    async def open_stream(
        self, *, voice_id: str, lang: str, epoch: int, cancel: CancelToken
    ) -> ValidTtsStream:
        return ValidTtsStream(epoch, cancel)

    async def synthesize(self, text: str, *, voice_id: str, lang: str) -> AudioClip:
        return AudioClip(pcm=b"\x00" * 640, sample_rate=16000)


class ValidLlmStream:
    def __init__(self, cancel: CancelToken) -> None:
        self._cancel = cancel
        self._aborted = False

    async def deltas(self) -> AsyncIterator[str]:
        if self._aborted or self._cancel.cancelled:
            return
        yield "Hello "
        if self._aborted or self._cancel.cancelled:
            return
        yield "world!"

    async def abort(self) -> None:
        self._aborted = True


class ValidLlmPort:
    async def stream(
        self, messages: Sequence[Message], *, epoch: int, cancel: CancelToken
    ) -> ValidLlmStream:
        return ValidLlmStream(cancel)


class ValidTransport:
    def __init__(self) -> None:
        self._queue_ms = 0

    async def user_frames(self) -> AsyncIterator[AudioFrame]:
        yield AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=0)

    def send_frame(self, frame: AudioFrame) -> None:
        self._queue_ms += 20

    def flush_output(self) -> None:
        self._queue_ms = 0

    def output_queue_ms(self) -> int:
        return self._queue_ms

    def playout_position_ms(self) -> int | None:
        return 0


class ValidVad:
    def process(self, frame: AudioFrame) -> VadResult:
        return VadResult(is_speech=False, confidence=0.0)


class ValidTelemetrySink:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, int, dict[str, Any]]] = []

    def emit(self, name: str, ts_ns: int, **fields: Any) -> None:
        self.emitted.append((name, ts_ns, fields))


# =============================================================================
# Broken Implementations
# =============================================================================


class BrokenSttNonIdempotentCloseStream(ValidSttStream):
    def __init__(self, cancel: CancelToken) -> None:
        super().__init__(cancel)
        self._close_count = 0

    async def close(self) -> None:
        self._close_count += 1
        if self._close_count > 1:
            raise RuntimeError("Double close error")


class BrokenSttOutOfOrderStream(ValidSttStream):
    async def events(self) -> AsyncIterator[SttEvent]:
        # Out of order timestamps
        yield SttEvent(kind="partial", text="first", ts_ns=5000)
        yield SttEvent(kind="final", text="second", ts_ns=1000)


class BrokenSttDeliveryAfterCancelStream(ValidSttStream):
    async def events(self) -> AsyncIterator[SttEvent]:
        # Keeps yielding despite cancellation
        while True:
            await asyncio.sleep(0)
            yield SttEvent(kind="partial", text="zombie", ts_ns=100)


class BrokenSttPortNonIdempotentClose:
    async def open(self, *, lang_hint: str | None = None, cancel: CancelToken) -> Any:
        return BrokenSttNonIdempotentCloseStream(cancel)


class BrokenSttPortOutOfOrder:
    async def open(self, *, lang_hint: str | None = None, cancel: CancelToken) -> Any:
        return BrokenSttOutOfOrderStream(cancel)


class BrokenSttPortDeliveryAfterCancel:
    async def open(self, *, lang_hint: str | None = None, cancel: CancelToken) -> Any:
        return BrokenSttDeliveryAfterCancelStream(cancel)


class BrokenTtsNonIdempotentCancelStream(ValidTtsStream):
    def __init__(self, epoch: int, cancel: CancelToken) -> None:
        super().__init__(epoch, cancel)
        self._cancel_count = 0

    async def cancel(self) -> None:
        self._cancel_count += 1
        if self._cancel_count > 1:
            raise RuntimeError("Double cancel error")


class BrokenTtsDeliveryAfterCancelStream(ValidTtsStream):
    async def chunks(self) -> AsyncIterator[AudioChunk]:
        # Keeps yielding chunks even after cancel
        for _ in range(5):
            await asyncio.sleep(0)
            yield AudioChunk(pcm=b"\x00" * 320, sample_rate=16000, epoch=self._epoch)


class BrokenTtsStaleEpochStream(ValidTtsStream):
    async def chunks(self) -> AsyncIterator[AudioChunk]:
        # Yields wrong epoch (0 instead of requested epoch)
        yield AudioChunk(pcm=b"\x00" * 320, sample_rate=16000, epoch=0)


class BrokenTtsPortNonIdempotentCancel(ValidTtsPort):
    async def open_stream(
        self, *, voice_id: str, lang: str, epoch: int, cancel: CancelToken
    ) -> Any:
        return BrokenTtsNonIdempotentCancelStream(epoch, cancel)


class BrokenTtsPortDeliveryAfterCancel(ValidTtsPort):
    async def open_stream(
        self, *, voice_id: str, lang: str, epoch: int, cancel: CancelToken
    ) -> Any:
        return BrokenTtsDeliveryAfterCancelStream(epoch, cancel)


class BrokenTtsPortStaleEpoch(ValidTtsPort):
    async def open_stream(
        self, *, voice_id: str, lang: str, epoch: int, cancel: CancelToken
    ) -> Any:
        return BrokenTtsStaleEpochStream(epoch, cancel)


class BrokenLlmNonIdempotentAbortStream(ValidLlmStream):
    def __init__(self, cancel: CancelToken) -> None:
        super().__init__(cancel)
        self._abort_count = 0

    async def abort(self) -> None:
        self._abort_count += 1
        if self._abort_count > 1:
            raise RuntimeError("Double abort error")


class BrokenLlmDeliveryAfterAbortStream(ValidLlmStream):
    async def deltas(self) -> AsyncIterator[str]:
        for _ in range(5):
            await asyncio.sleep(0)
            yield "token "


class BrokenLlmPortNonIdempotentAbort(ValidLlmPort):
    async def stream(self, messages: Sequence[Message], *, epoch: int, cancel: CancelToken) -> Any:
        return BrokenLlmNonIdempotentAbortStream(cancel)


class BrokenLlmPortDeliveryAfterAbort(ValidLlmPort):
    async def stream(self, messages: Sequence[Message], *, epoch: int, cancel: CancelToken) -> Any:
        return BrokenLlmDeliveryAfterAbortStream(cancel)


class BrokenTransportAsyncSendFrame:
    # Violation: send_frame must NOT be async
    async def user_frames(self) -> AsyncIterator[AudioFrame]:
        yield AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=0)

    async def send_frame(self, frame: AudioFrame) -> None:
        pass

    def flush_output(self) -> None:
        pass

    def output_queue_ms(self) -> int:
        return 0

    def playout_position_ms(self) -> int | None:
        return 0


class BrokenTransportFlushNoOp:
    # Violation: flush does not reset output queue
    def __init__(self) -> None:
        self._queue_ms = 20

    async def user_frames(self) -> AsyncIterator[AudioFrame]:
        yield AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=0)

    def send_frame(self, frame: AudioFrame) -> None:
        self._queue_ms += 20

    def flush_output(self) -> None:
        pass

    def output_queue_ms(self) -> int:
        return self._queue_ms

    def playout_position_ms(self) -> int | None:
        return 0


class BrokenVadAsyncProcess:
    # Violation: process must NOT be async
    async def process(self, frame: AudioFrame) -> VadResult:
        return VadResult(is_speech=False, confidence=0.0)


class BrokenVadInvalidConfidence:
    def process(self, frame: AudioFrame) -> VadResult:
        # Violation: confidence > 1.0
        return VadResult(is_speech=True, confidence=2.5)


class BrokenTelemetryAsyncEmit:
    # Violation: emit must NOT be async
    async def emit(self, name: str, ts_ns: int, **fields: Any) -> None:
        pass


class BrokenTelemetryRejectsKwargs:
    # Violation: does not accept arbitrary fields
    def emit(self, name: str, ts_ns: int) -> None:
        pass


# =============================================================================
# Valid Implementations Pass Suites
# =============================================================================


def test_stt_contract_passes_valid_impl() -> None:
    run_virtual(SttContract.verify(ValidSttPort()))


def test_tts_contract_passes_valid_impl() -> None:
    run_virtual(TtsContract.verify(ValidTtsPort()))


def test_llm_contract_passes_valid_impl() -> None:
    run_virtual(LlmContract.verify(ValidLlmPort()))


def test_transport_contract_passes_valid_impl() -> None:
    TransportContract.verify(ValidTransport())


def test_vad_contract_passes_valid_impl() -> None:
    VadContract.verify(ValidVad())


def test_telemetry_contract_passes_valid_impl() -> None:
    TelemetryContract.verify(ValidTelemetrySink())


# =============================================================================
# Broken Implementations Fail Suites (Suite Has Teeth)
# =============================================================================


def test_stt_contract_detects_broken_impl() -> None:
    """Suite detects broken Stt implementation."""

    async def _test() -> None:
        with pytest.raises(AssertionError):
            await SttContract.verify(BrokenSttPortNonIdempotentClose())
        with pytest.raises(AssertionError):
            await SttContract.verify(BrokenSttPortOutOfOrder())
        with pytest.raises(AssertionError):
            await SttContract.verify(BrokenSttPortDeliveryAfterCancel())

    run_virtual(_test())


def test_tts_contract_detects_broken_impl() -> None:
    """Suite detects broken Tts implementation."""

    async def _test() -> None:
        with pytest.raises(AssertionError):
            await TtsContract.verify(BrokenTtsPortNonIdempotentCancel())
        with pytest.raises(AssertionError):
            await TtsContract.verify(BrokenTtsPortDeliveryAfterCancel())
        with pytest.raises(AssertionError):
            await TtsContract.verify(BrokenTtsPortStaleEpoch())

    run_virtual(_test())


def test_llm_contract_detects_broken_impl() -> None:
    """Suite detects broken Llm implementation."""

    async def _test() -> None:
        with pytest.raises(AssertionError):
            await LlmContract.verify(BrokenLlmPortNonIdempotentAbort())
        with pytest.raises(AssertionError):
            await LlmContract.verify(BrokenLlmPortDeliveryAfterAbort())

    run_virtual(_test())


def test_transport_contract_detects_broken_impl() -> None:
    """Suite detects broken Transport implementation."""
    with pytest.raises(AssertionError):
        TransportContract.verify(BrokenTransportAsyncSendFrame())
    with pytest.raises(AssertionError):
        TransportContract.verify(BrokenTransportFlushNoOp())


def test_vad_contract_detects_broken_impl() -> None:
    """Suite detects broken Vad implementation."""
    with pytest.raises(AssertionError):
        VadContract.verify(BrokenVadAsyncProcess())
    with pytest.raises(AssertionError):
        VadContract.verify(BrokenVadInvalidConfidence())


def test_telemetry_contract_detects_broken_impl() -> None:
    """Suite detects broken Telemetry implementation."""
    with pytest.raises(AssertionError):
        TelemetryContract.verify(BrokenTelemetryAsyncEmit())
    with pytest.raises(AssertionError):
        TelemetryContract.verify(BrokenTelemetryRejectsKwargs())


# Additional granular tests demonstrating detection of each broken impl
def test_stt_contract_detects_broken_impl_non_idempotent_close() -> None:
    async def _test() -> None:
        with pytest.raises(AssertionError):
            await SttContract().check_idempotent_close(BrokenSttPortNonIdempotentClose())

    run_virtual(_test())


def test_stt_contract_detects_broken_impl_events_out_of_order() -> None:
    async def _test() -> None:
        with pytest.raises(AssertionError):
            await SttContract().check_events_in_order(BrokenSttPortOutOfOrder())

    run_virtual(_test())


def test_tts_contract_detects_broken_impl_cancel_delivery() -> None:
    async def _test() -> None:
        with pytest.raises(AssertionError):
            await TtsContract().check_cancel_stops_delivery(BrokenTtsPortDeliveryAfterCancel())

    run_virtual(_test())


def test_tts_contract_detects_broken_impl_stale_epoch() -> None:
    async def _test() -> None:
        with pytest.raises(AssertionError):
            await TtsContract().check_epoch_propagation(BrokenTtsPortStaleEpoch())

    run_virtual(_test())


def test_transport_contract_detects_broken_impl_async_send_frame() -> None:
    with pytest.raises(AssertionError):
        TransportContract().check_fast_path_non_blocking_sync(BrokenTransportAsyncSendFrame())


def test_vad_contract_detects_broken_impl_async_process() -> None:
    with pytest.raises(AssertionError):
        VadContract().check_fast_path_non_blocking_sync(BrokenVadAsyncProcess())
