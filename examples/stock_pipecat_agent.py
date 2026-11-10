"""Stock Pipecat baseline agent for A/B naturalness evaluation (ARCHITECTURE §12, §13).

Uses the exact same STT, LLM, TTS, and VAD ports as `examples/pipecat_agent.py`,
but implements stock Pipecat stop-on-any-VAD interruption behavior:
- Any VAD speech onset immediately stops agent playout (no 0.3 gain ducking window,
  no backchannel vs interruption classification, no agent backchannels).
- Serves as the A/B control condition against `OverlapProcessor`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from overlap.config import AppConfig, load_config
from overlap.core.clock import Clock
from overlap.core.types import Tuning
from overlap.integrations.pipecat_processor import (
    FrameDirection,
    PipelineTaskParams,
    StockInterruptionProcessor,
)
from overlap.ports.llm import LlmPort
from overlap.ports.stt import SttPort
from overlap.ports.telemetry import TelemetrySink
from overlap.ports.tts import TtsPort
from overlap.ports.vad import Vad


def build_stock_pipecat_processor(
    *,
    vad: Vad,
    stt: SttPort,
    tts: TtsPort,
    llm: LlmPort,
    tuning: Tuning | None = None,
    config: AppConfig | None = None,
    clock: Clock | None = None,
    telemetry: TelemetrySink | None = None,
) -> StockInterruptionProcessor:
    """Create a stock stop-on-any-VAD Pipecat baseline processor sharing the same ports."""
    cfg = config if config is not None else load_config()
    active_tuning = tuning if tuning is not None else cfg.tuning
    return StockInterruptionProcessor(
        vad=vad,
        stt=stt,
        tts=tts,
        llm=llm,
        tuning=active_tuning,
        config=cfg,
        clock=clock,
        telemetry=telemetry,
    )


def create_stock_task_params(tuning: Tuning | None = None) -> PipelineTaskParams:
    """Create `PipelineTaskParams` with stock Pipecat `allow_interruptions=True`."""
    t = tuning if tuning is not None else Tuning()
    return PipelineTaskParams(
        allow_interruptions=True,
        audio_in_sample_rate=t.in_rate_hz,
        audio_out_sample_rate=t.out_rate_hz,
    )


async def run_stock_pipecat_session(
    processor: StockInterruptionProcessor,
    frames: Sequence[Any],
) -> list[tuple[Any, Any]]:
    """Run a sequence of Pipecat frames through `StockInterruptionProcessor`."""
    for frame in frames:
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    return list(processor.emitted_frames)
