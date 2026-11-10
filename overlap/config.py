"""Application configuration and environment variable loading.

Per ARCHITECTURE §2:
- config.py: default Tuning + env loading (keys only from env).
- Pure stdlib os.environ loading with defaults.
- Secrets are never hardcoded and read strictly from environment.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from overlap.core.types import Tuning

DEFAULT_TUNING: Tuning = Tuning()


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Configuration settings for OVERLAP engine, ports, and external APIs."""

    gnani_api_key: str = ""
    gnani_api_url: str = "https://api.vachana.ai"
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    voice_id: str = "v1"
    lang: str = "hi"
    tuning: Tuning = field(default_factory=Tuning)


def load_config(env: Mapping[str, str] | None = None) -> AppConfig:
    """Load configuration from environment variables or a provided mapping."""
    source = env if env is not None else os.environ

    return AppConfig(
        gnani_api_key=source.get("GNANI_API_KEY", ""),
        gnani_api_url=source.get("GNANI_API_URL", "https://api.vachana.ai"),
        livekit_url=source.get("LIVEKIT_URL", ""),
        livekit_api_key=source.get("LIVEKIT_API_KEY", ""),
        livekit_api_secret=source.get("LIVEKIT_API_SECRET", ""),
        voice_id=source.get("GNANI_VOICE_ID", "v1"),
        lang=source.get("GNANI_LANG", "hi"),
        tuning=DEFAULT_TUNING,
    )
