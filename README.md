# OVERLAP

Full-duplex conversational audio pipeline with duck-then-decide, truncation to heard text, and agent backchannels on Gnani Prisma → Evon → Timbre.

## Architecture & Layers

OVERLAP enforces strict architectural layering:
- `overlap.core`: Pure data types, state machine, classifier, audio math (no I/O, no async, no clock access except via injected `Clock`).
- `overlap.ports`: Abstract protocol interfaces (`typing.Protocol`).
- `overlap.runtime`: Asyncio orchestration (pipeline, mixer, stream supervisor).
- `overlap.adapters`: Concrete network/model clients (Gnani Prisma/Evon/Timbre, Silero VAD).
- `overlap.integrations`: Transport framework integrations (LiveKit, Pipecat).

## Development Setup

The project requires Python 3.11 inside a conda environment named `gnani`.

```bash
conda activate gnani
pip install -e ".[dev]"
```

## Gates & Commands

- `make fast`: Pure, hermetic, offline test suite (< 15s budget, network sockets disabled).
- `make check`: Full quality gate (`ruff check`, `ruff format --check`, `lint-imports`, `mypy --strict`, `pytest -m "not live"`, `dag_lint validate`).
