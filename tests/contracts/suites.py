"""Reusable, parametrizable contract test suites for all OVERLAP ports.

These suites test that port implementations (both fakes and production adapters)
strictly adhere to the protocols and invariants specified in ARCHITECTURE §5 and §10.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from typing import Any

from overlap.core.clock import CancelToken
from overlap.core.types import AudioFrame, SttEvent
from overlap.ports.llm import LlmPort, Message
from overlap.ports.stt import SttPort
from overlap.ports.telemetry import NullSink, TelemetrySink
from overlap.ports.transport import AudioTransport
from overlap.ports.tts import AudioChunk, AudioClip, TtsPort
from overlap.ports.vad import Vad, VadResult

_helpers_dir = str(Path(__file__).resolve().parent.parent / "helpers")
if _helpers_dir not in sys.path:
    sys.path.insert(0, _helpers_dir)

from virtual import run_virtual  # noqa: E402


def _resolve(obj_or_factory: Any, method_name: str) -> Any:
    if callable(obj_or_factory) and not hasattr(obj_or_factory, method_name):
        return obj_or_factory()
    return obj_or_factory


class SttContract:
    """Reusable contract suite for SttPort and SttStream implementations."""

    def check_fast_path_signatures(self, port: Any) -> None:
        resolved = _resolve(port, "open")
        assert inspect.iscoroutinefunction(resolved.open), (
            "SttPort.open MUST be a coroutine function"
        )

    async def check_idempotent_close(self, port: Any) -> None:
        resolved = _resolve(port, "open")
        token = CancelToken()
        stream = await resolved.open(cancel=token)
        await stream.close()
        # Consecutive close calls must be safe and idempotent
        try:
            await stream.close()
            await stream.close()
        except Exception as exc:
            raise AssertionError(f"SttStream.close() is not idempotent: raised {exc}") from exc

    async def check_events_in_order(self, port: Any) -> None:
        resolved = _resolve(port, "open")
        token = CancelToken()
        stream = await resolved.open(cancel=token)
        prev_ts = -1

        async def _drain() -> None:
            nonlocal prev_ts
            async for evt in stream.events():
                assert isinstance(evt, SttEvent), f"Expected SttEvent, got {type(evt)}"
                if evt.ts_ns > 0 and prev_ts > 0:
                    assert evt.ts_ns >= prev_ts, (
                        f"Stt events arrived out of order: {evt.ts_ns} < {prev_ts}"
                    )
                prev_ts = max(prev_ts, evt.ts_ns)

        try:
            await asyncio.wait_for(_drain(), timeout=0.05)
        except TimeoutError:
            pass
        finally:
            await stream.close()

    async def check_cancel_stops_delivery(self, port: Any) -> None:
        resolved = _resolve(port, "open")

        # 1. Cancel via CancelToken
        token = CancelToken()
        stream = await resolved.open(cancel=token)
        token.cancel()
        await asyncio.sleep(0)

        delivered_after_cancel: list[SttEvent] = []

        async def _drain_token() -> None:
            async for evt in stream.events():
                delivered_after_cancel.append(evt)

        try:
            await asyncio.wait_for(_drain_token(), timeout=0.05)
        except TimeoutError as err:
            raise AssertionError("SttStream.events() hung after CancelToken was cancelled") from err

        assert len(delivered_after_cancel) == 0, (
            f"SttStream delivered {len(delivered_after_cancel)} events after cancellation"
        )

        # 2. Cancel via close()
        token2 = CancelToken()
        stream2 = await resolved.open(cancel=token2)
        await stream2.close()
        await asyncio.sleep(0)

        delivered_after_close: list[SttEvent] = []

        async def _drain_close() -> None:
            async for evt in stream2.events():
                delivered_after_close.append(evt)

        try:
            await asyncio.wait_for(_drain_close(), timeout=0.05)
        except TimeoutError as err:
            raise AssertionError("SttStream.events() hung after close()") from err

        assert len(delivered_after_close) == 0, (
            f"SttStream delivered {len(delivered_after_close)} events after close()"
        )

    async def check_bounded_queue_drop_oldest(self, port: Any) -> None:
        resolved = _resolve(port, "open")
        token = CancelToken()
        stream = await resolved.open(cancel=token)
        try:
            # Enqueue 100 frames rapidly without reading events()
            for i in range(100):
                frame = AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=i * 20_000_000)
                await asyncio.wait_for(stream.send_audio(frame), timeout=0.05)
        except TimeoutError as err:
            raise AssertionError(
                "SttStream.send_audio blocked on full buffer instead of dropping oldest"
            ) from err
        except Exception as exc:
            raise AssertionError(
                f"SttStream.send_audio failed during rapid enqueue: {exc}"
            ) from exc
        finally:
            await stream.close()

    # Reusable pytest methods for subclasses
    def test_fast_path_signatures(self, stt_port: SttPort) -> None:
        self.check_fast_path_signatures(stt_port)

    def test_idempotent_close(self, stt_port: SttPort) -> None:
        run_virtual(self.check_idempotent_close(stt_port))

    def test_events_in_order(self, stt_port: SttPort) -> None:
        run_virtual(self.check_events_in_order(stt_port))

    def test_cancel_stops_delivery(self, stt_port: SttPort) -> None:
        run_virtual(self.check_cancel_stops_delivery(stt_port))

    def test_bounded_queue_drop_oldest(self, stt_port: SttPort) -> None:
        run_virtual(self.check_bounded_queue_drop_oldest(stt_port))

    @classmethod
    async def verify(cls, port: Any) -> None:
        """Run the full contract suite against a port or port factory."""
        suite = cls()
        suite.check_fast_path_signatures(port)
        await suite.check_idempotent_close(port)
        await suite.check_events_in_order(port)
        await suite.check_cancel_stops_delivery(port)
        await suite.check_bounded_queue_drop_oldest(port)

    @classmethod
    def verify_sync(cls, port: Any) -> None:
        """Synchronous wrapper for verify."""
        run_virtual(cls.verify(port))


class TtsContract:
    """Reusable contract suite for TtsPort and TtsStream implementations."""

    def check_fast_path_signatures(self, port: Any) -> None:
        resolved = _resolve(port, "open_stream")
        assert inspect.iscoroutinefunction(resolved.open_stream), (
            "TtsPort.open_stream MUST be a coroutine function"
        )
        assert inspect.iscoroutinefunction(resolved.synthesize), (
            "TtsPort.synthesize MUST be a coroutine function"
        )

    async def check_idempotent_cancel(self, port: Any) -> None:
        resolved = _resolve(port, "open_stream")
        token = CancelToken()
        stream = await resolved.open_stream(voice_id="v1", lang="hi", epoch=1, cancel=token)
        await stream.cancel()
        try:
            await stream.cancel()
            await stream.cancel()
        except Exception as exc:
            raise AssertionError(f"TtsStream.cancel() is not idempotent: raised {exc}") from exc

    async def check_cancel_stops_delivery(self, port: Any) -> None:
        resolved = _resolve(port, "open_stream")
        token = CancelToken()
        stream = await resolved.open_stream(voice_id="v1", lang="hi", epoch=1, cancel=token)
        await stream.send_text("Namaste, kaise hain aap?")
        await stream.cancel()
        # Must stop chunk delivery within 1 loop turn
        await asyncio.sleep(0)

        delivered: list[AudioChunk] = []

        async def _drain() -> None:
            async for chunk in stream.chunks():
                delivered.append(chunk)

        try:
            await asyncio.wait_for(_drain(), timeout=0.05)
        except TimeoutError as err:
            raise AssertionError("TtsStream.chunks() hung after cancel()") from err

        assert len(delivered) == 0, (
            f"TtsStream kept yielding chunks ({len(delivered)}) after cancel()"
        )

    async def check_epoch_propagation(self, port: Any) -> None:
        resolved = _resolve(port, "open_stream")
        expected_epoch = 77
        token = CancelToken()
        stream = await resolved.open_stream(
            voice_id="v1", lang="hi", epoch=expected_epoch, cancel=token
        )
        await stream.send_text("Test clause")
        await stream.finish()

        async def _drain() -> None:
            async for chunk in stream.chunks():
                assert isinstance(chunk, AudioChunk), f"Expected AudioChunk, got {type(chunk)}"
                assert chunk.epoch == expected_epoch, (
                    f"AudioChunk epoch {chunk.epoch} does not match "
                    f"requested epoch {expected_epoch}"
                )

        try:
            await asyncio.wait_for(_drain(), timeout=0.05)
        except TimeoutError:
            pass
        finally:
            await stream.cancel()

    async def check_batch_synthesize(self, port: Any) -> None:
        resolved = _resolve(port, "synthesize")
        clip = await resolved.synthesize("test", voice_id="v1", lang="hi")
        assert isinstance(clip, AudioClip), f"Expected AudioClip, got {type(clip)}"
        assert isinstance(clip.pcm, bytes), f"Expected bytes pcm, got {type(clip.pcm)}"
        assert clip.sample_rate > 0, f"Expected sample_rate > 0, got {clip.sample_rate}"

    # Reusable pytest methods for subclasses
    def test_fast_path_signatures(self, tts_port: TtsPort) -> None:
        self.check_fast_path_signatures(tts_port)

    def test_idempotent_cancel(self, tts_port: TtsPort) -> None:
        run_virtual(self.check_idempotent_cancel(tts_port))

    def test_cancel_stops_delivery(self, tts_port: TtsPort) -> None:
        run_virtual(self.check_cancel_stops_delivery(tts_port))

    def test_epoch_propagation(self, tts_port: TtsPort) -> None:
        run_virtual(self.check_epoch_propagation(tts_port))

    def test_batch_synthesize(self, tts_port: TtsPort) -> None:
        run_virtual(self.check_batch_synthesize(tts_port))

    @classmethod
    async def verify(cls, port: Any) -> None:
        """Run the full contract suite against a port or port factory."""
        suite = cls()
        suite.check_fast_path_signatures(port)
        await suite.check_idempotent_cancel(port)
        await suite.check_cancel_stops_delivery(port)
        await suite.check_epoch_propagation(port)
        await suite.check_batch_synthesize(port)

    @classmethod
    def verify_sync(cls, port: Any) -> None:
        """Synchronous wrapper for verify."""
        run_virtual(cls.verify(port))


class LlmContract:
    """Reusable contract suite for LlmPort and LlmStream implementations."""

    def check_fast_path_signatures(self, port: Any) -> None:
        resolved = _resolve(port, "stream")
        assert inspect.iscoroutinefunction(resolved.stream), (
            "LlmPort.stream MUST be a coroutine function"
        )

    async def check_idempotent_abort(self, port: Any) -> None:
        resolved = _resolve(port, "stream")
        token = CancelToken()
        stream = await resolved.stream(
            [Message(role="user", content="hello")], epoch=1, cancel=token
        )
        await stream.abort()
        try:
            await stream.abort()
            await stream.abort()
        except Exception as exc:
            raise AssertionError(f"LlmStream.abort() is not idempotent: raised {exc}") from exc

    async def check_abort_stops_delivery(self, port: Any) -> None:
        resolved = _resolve(port, "stream")
        token = CancelToken()
        stream = await resolved.stream(
            [Message(role="user", content="hello")], epoch=1, cancel=token
        )
        await stream.abort()
        await asyncio.sleep(0)

        delivered: list[str] = []

        async def _drain() -> None:
            async for delta in stream.deltas():
                delivered.append(delta)

        try:
            await asyncio.wait_for(_drain(), timeout=0.05)
        except TimeoutError as err:
            raise AssertionError("LlmStream.deltas() hung after abort()") from err

        assert len(delivered) == 0, (
            f"LlmStream kept yielding deltas ({len(delivered)}) after abort()"
        )

    async def check_cancel_token_aborts(self, port: Any) -> None:
        resolved = _resolve(port, "stream")
        token = CancelToken()
        stream = await resolved.stream(
            [Message(role="user", content="hello")], epoch=1, cancel=token
        )
        token.cancel()
        await asyncio.sleep(0)

        delivered: list[str] = []

        async def _drain() -> None:
            async for delta in stream.deltas():
                delivered.append(delta)

        try:
            await asyncio.wait_for(_drain(), timeout=0.05)
        except TimeoutError as err:
            raise AssertionError("LlmStream.deltas() hung after CancelToken cancelled") from err

        assert len(delivered) == 0, (
            f"LlmStream kept yielding deltas ({len(delivered)}) after CancelToken cancelled"
        )

    # Reusable pytest methods for subclasses
    def test_fast_path_signatures(self, llm_port: LlmPort) -> None:
        self.check_fast_path_signatures(llm_port)

    def test_idempotent_abort(self, llm_port: LlmPort) -> None:
        run_virtual(self.check_idempotent_abort(llm_port))

    def test_abort_stops_delivery(self, llm_port: LlmPort) -> None:
        run_virtual(self.check_abort_stops_delivery(llm_port))

    def test_cancel_token_aborts(self, llm_port: LlmPort) -> None:
        run_virtual(self.check_cancel_token_aborts(llm_port))

    @classmethod
    async def verify(cls, port: Any) -> None:
        """Run the full contract suite against a port or port factory."""
        suite = cls()
        suite.check_fast_path_signatures(port)
        await suite.check_idempotent_abort(port)
        await suite.check_abort_stops_delivery(port)
        await suite.check_cancel_token_aborts(port)

    @classmethod
    def verify_sync(cls, port: Any) -> None:
        """Synchronous wrapper for verify."""
        run_virtual(cls.verify(port))


class TransportContract:
    """Reusable contract suite for AudioTransport implementations."""

    def check_fast_path_non_blocking_sync(self, transport: Any) -> None:
        resolved = _resolve(transport, "send_frame")
        assert not inspect.iscoroutinefunction(resolved.send_frame), (
            "AudioTransport.send_frame MUST be synchronous and non-blocking"
        )
        assert not inspect.iscoroutinefunction(resolved.flush_output), (
            "AudioTransport.flush_output MUST be synchronous"
        )
        assert not inspect.iscoroutinefunction(resolved.output_queue_ms), (
            "AudioTransport.output_queue_ms MUST be synchronous"
        )
        assert not inspect.iscoroutinefunction(resolved.playout_position_ms), (
            "AudioTransport.playout_position_ms MUST be synchronous"
        )

        frame = AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=0)
        ret = resolved.send_frame(frame)
        assert ret is None, f"send_frame must return None, got {ret!r}"
        assert not inspect.isawaitable(ret), "send_frame returned an awaitable"

    def check_flush_output(self, transport: Any) -> None:
        resolved = _resolve(transport, "flush_output")
        frame = AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=0)
        resolved.send_frame(frame)
        resolved.flush_output()
        queue_ms = resolved.output_queue_ms()
        assert isinstance(queue_ms, int) and queue_ms >= 0, (
            f"output_queue_ms must return non-negative int, got {queue_ms}"
        )
        assert queue_ms == 0, f"Expected output_queue_ms == 0 after flush_output, got {queue_ms}"

    def check_playout_position(self, transport: Any) -> None:
        resolved = _resolve(transport, "playout_position_ms")
        pos = resolved.playout_position_ms()
        assert pos is None or (isinstance(pos, int) and pos >= 0), (
            f"playout_position_ms must be None or non-negative int, got {pos}"
        )

    def check_user_frames(self, transport: Any) -> None:
        resolved = _resolve(transport, "user_frames")
        frames_iter = resolved.user_frames()
        assert hasattr(frames_iter, "__aiter__"), (
            "AudioTransport.user_frames() must return an AsyncIterator"
        )

    # Reusable pytest methods for subclasses
    def test_fast_path_non_blocking_sync(self, transport: AudioTransport) -> None:
        self.check_fast_path_non_blocking_sync(transport)

    def test_flush_output(self, transport: AudioTransport) -> None:
        self.check_flush_output(transport)

    def test_playout_position(self, transport: AudioTransport) -> None:
        self.check_playout_position(transport)

    def test_user_frames(self, transport: AudioTransport) -> None:
        self.check_user_frames(transport)

    @classmethod
    def verify(cls, transport: Any) -> None:
        """Run the full contract suite against a transport or factory."""
        suite = cls()
        suite.check_fast_path_non_blocking_sync(transport)
        suite.check_flush_output(transport)
        suite.check_playout_position(transport)
        suite.check_user_frames(transport)


class VadContract:
    """Reusable contract suite for Vad implementations."""

    def check_fast_path_non_blocking_sync(self, vad: Any) -> None:
        resolved = _resolve(vad, "process")
        assert not inspect.iscoroutinefunction(resolved.process), (
            "Vad.process MUST be synchronous and non-blocking"
        )
        frame = AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=0)
        res = resolved.process(frame)
        assert not inspect.isawaitable(res), "Vad.process returned an awaitable"

    def check_process_result_contract(self, vad: Any) -> None:
        resolved = _resolve(vad, "process")
        frame = AudioFrame(pcm=b"\x00" * 640, sample_rate=16000, ts_ns=0)
        res = resolved.process(frame)
        assert isinstance(res, VadResult), f"Expected VadResult, got {type(res)}"
        assert isinstance(res.is_speech, bool), (
            f"Expected bool is_speech, got {type(res.is_speech)}"
        )
        assert 0.0 <= res.confidence <= 1.0, (
            f"VadResult confidence out of bounds [0.0, 1.0]: {res.confidence}"
        )

    # Reusable pytest methods for subclasses
    def test_fast_path_non_blocking_sync(self, vad: Vad) -> None:
        self.check_fast_path_non_blocking_sync(vad)

    def test_process_result_contract(self, vad: Vad) -> None:
        self.check_process_result_contract(vad)

    @classmethod
    def verify(cls, vad: Any) -> None:
        """Run the full contract suite against a VAD or factory."""
        suite = cls()
        suite.check_fast_path_non_blocking_sync(vad)
        suite.check_process_result_contract(vad)


class TelemetryContract:
    """Reusable contract suite for TelemetrySink implementations."""

    def check_fast_path_non_blocking_sync(self, sink: Any) -> None:
        resolved = _resolve(sink, "emit")
        assert not inspect.iscoroutinefunction(resolved.emit), (
            "TelemetrySink.emit MUST be synchronous and non-blocking"
        )
        try:
            ret = resolved.emit("test.metric", ts_ns=1000, key="value", count=1)
        except TypeError as exc:
            raise AssertionError(f"TelemetrySink.emit must accept arbitrary kwargs: {exc}") from exc
        assert ret is None, f"emit must return None, got {ret!r}"
        assert not inspect.isawaitable(ret), "emit returned an awaitable"

    def check_arbitrary_fields(self, sink: Any) -> None:
        resolved = _resolve(sink, "emit")
        try:
            resolved.emit(
                "test.event",
                ts_ns=2000,
                s="str_val",
                i=42,
                f=3.14,
                b=True,
                n=None,
                d={"k": "v"},
                t=(1, 2),
            )
        except TypeError as exc:
            raise AssertionError(
                f"TelemetrySink.emit rejected arbitrary keyword fields: {exc}"
            ) from exc

    def check_null_sink(self) -> None:
        null_sink = NullSink()
        null_sink.emit("test", 0, foo="bar")

    # Reusable pytest methods for subclasses
    def test_fast_path_non_blocking_sync(self, sink: TelemetrySink) -> None:
        self.check_fast_path_non_blocking_sync(sink)

    def test_arbitrary_fields(self, sink: TelemetrySink) -> None:
        self.check_arbitrary_fields(sink)

    def test_null_sink(self) -> None:
        self.check_null_sink()

    @classmethod
    def verify(cls, sink: Any) -> None:
        """Run the full contract suite against a telemetry sink or factory."""
        suite = cls()
        suite.check_fast_path_non_blocking_sync(sink)
        suite.check_arbitrary_fields(sink)
        suite.check_null_sink()
