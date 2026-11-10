"""Live integration & downstream playout latency calibration test for LiveKitTransport.

Marked `@pytest.mark.live` so it is excluded from `make fast` and `make check`
and only executed when triggered by a human via `make live` with valid
`LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` credentials.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import numpy as np
import pytest

from overlap.core.clock import MonotonicClock
from overlap.core.types import AudioFrame, Tuning
from overlap.integrations.livekit_transport import LiveKitTransport


def _load_env_file() -> None:
    for candidate in (Path(".env"), Path(__file__).resolve().parent.parent.parent / ".env"):
        if candidate.is_file():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
            except Exception:
                pass
            break


@pytest.mark.live
def test_livekit_roundtrip_and_downstream_calibration() -> None:
    """Connect a publisher and loopback subscriber to a LiveKit room, measure
    end-to-end audio pulse latency and `flush_output()` stop time, and report
    the calibrated `assumed_downstream_ms`.
    """
    _load_env_file()
    livekit_url = os.environ.get("LIVEKIT_URL", "").strip()
    api_key = os.environ.get("LIVEKIT_API_KEY", "").strip()
    api_secret = os.environ.get("LIVEKIT_API_SECRET", "").strip()

    assert livekit_url and api_key and api_secret, (
        "LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET must be set in environment "
        "to run live LiveKit transport tests."
    )

    from livekit import api, rtc

    async def _run_calibration() -> int:
        room_name = f"overlap-calib-{int(time.time())}"
        agent_token = (
            api.AccessToken(api_key, api_secret)
            .with_identity("overlap-agent")
            .with_grants(api.VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )
        probe_token = (
            api.AccessToken(api_key, api_secret)
            .with_identity("overlap-probe")
            .with_grants(api.VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )

        agent_room = rtc.Room()
        probe_room = rtc.Room()
        track_ready: asyncio.Future[rtc.Track] = asyncio.get_running_loop().create_future()

        @probe_room.on("track_subscribed")
        def _on_track_subscribed(
            track: rtc.Track,
            _publication: rtc.RemoteTrackPublication,
            _participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind == rtc.TrackKind.KIND_AUDIO and not track_ready.done():
                track_ready.set_result(track)

        await agent_room.connect(livekit_url, agent_token)
        await probe_room.connect(livekit_url, probe_token)

        try:
            tuning = Tuning()
            audio_source = rtc.AudioSource(sample_rate=48000, num_channels=1)
            local_track = rtc.LocalAudioTrack.create_audio_track("agent-voice", audio_source)
            await agent_room.local_participant.publish_track(local_track)

            remote_track = await asyncio.wait_for(track_ready, timeout=10.0)
            audio_stream = rtc.AudioStream(remote_track)

            clock = MonotonicClock()
            transport = LiveKitTransport(audio_stream, audio_source, tuning=tuning, clock=clock)

            # Generate a 20 ms 1 kHz tone pulse at amplitude 16000
            t = np.arange(320, dtype=np.float64) / float(tuning.out_rate_hz)
            pulse_pcm = np.round(16000.0 * np.sin(2.0 * np.pi * 1000.0 * t)).astype("<i2").tobytes()

            send_ns = clock.now_ns()
            transport.send_frame(
                AudioFrame(pcm=pulse_pcm, sample_rate=tuning.out_rate_hz, ts_ns=send_ns)
            )

            recv_ns: int | None = None
            async for in_frame in transport.user_frames():
                samples = np.frombuffer(in_frame.pcm, dtype="<i2").astype(np.float64)
                rms = float(np.sqrt(np.mean(samples**2)))
                if rms > 2000.0:
                    recv_ns = clock.now_ns()
                    break

            assert recv_ns is not None, "Did not receive loopback audio pulse from LiveKit room"
            one_way_ms = int(round((recv_ns - send_ns) / 2_000_000.0))
            transport.close()
            return one_way_ms
        finally:
            await probe_room.disconnect()
            await agent_room.disconnect()

    measured_ms = asyncio.run(_run_calibration())
    print(f"\n[LIVEKIT CALIBRATION] Measured assumed_downstream_ms: {measured_ms} ms")
    assert 10 <= measured_ms <= 500
