"""Pipecat agent example using `OverlapProcessor` as a one-line drop-in (ARCHITECTURE §3, §13).

Demonstrates inserting `OverlapProcessor(tuning=..., config=...)` between transport input
and STT/TTS/transport output in a Pipecat pipeline while disabling Pipecat's built-in
interruption strategy so OVERLAP handles duck-then-decide, heard-text truncation, and
agent backchannels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from overlap.config import AppConfig, load_config
from overlap.core.clock import Clock
from overlap.core.types import Tuning
from overlap.integrations.pipecat_processor import (
    FrameDirection,
    OverlapProcessor,
    PipelineTaskParams,
    disable_pipecat_interruptions,
)
from overlap.ports.llm import LlmPort
from overlap.ports.stt import SttPort
from overlap.ports.telemetry import TelemetrySink
from overlap.ports.tts import TtsPort
from overlap.ports.vad import Vad


def build_pipecat_overlap_processor(
    *,
    tuning: Tuning | None = None,
    config: AppConfig | None = None,
    vad: Vad | None = None,
    stt: SttPort | None = None,
    tts: TtsPort | None = None,
    llm: LlmPort | None = None,
    clock: Clock | None = None,
    telemetry: TelemetrySink | None = None,
) -> OverlapProcessor:
    """Create a configured `OverlapProcessor` ready for insertion into a Pipecat pipeline."""
    cfg = config if config is not None else load_config()
    active_tuning = tuning if tuning is not None else cfg.tuning
    return OverlapProcessor(
        tuning=active_tuning,
        config=cfg,
        vad=vad,
        stt=stt,
        tts=tts,
        llm=llm,
        clock=clock,
        telemetry=telemetry,
    )


def create_overlap_task_params(tuning: Tuning | None = None) -> PipelineTaskParams:
    """Create `PipelineTaskParams` with Pipecat's built-in interruptions disabled."""
    t = tuning if tuning is not None else Tuning()
    params = PipelineTaskParams(
        allow_interruptions=False,
        audio_in_sample_rate=t.in_rate_hz,
        audio_out_sample_rate=t.out_rate_hz,
    )
    disable_pipecat_interruptions(params)
    return params


async def run_pipecat_session(
    processor: OverlapProcessor,
    frames: Sequence[Any],
) -> list[tuple[Any, Any]]:
    """Run a sequence of Pipecat frames through `OverlapProcessor` and return emitted frames."""
    for frame in frames:
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    return list(processor.emitted_frames)
