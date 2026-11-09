# CLAUDE.md — OVERLAP agent operating manual

You are a coding agent executing **one micro-task at a time** from `TASKS.md`. Project: full-duplex conversation (duck-then-decide, truncation to heard text, agent backchannels) on Gnani Prisma → Evon → Timbre. Design lives in `ARCHITECTURE.md` (numbered sections). **Do not read it whole.**

## 0. Operational overrides (from the human architect; take precedence)

**Environment.** All commands run in conda env `gnani` (Python 3.11). Start every session with `conda activate gnani` and check `python --version` prints 3.11.x. If a shell can't activate, prefix commands with `conda run -n gnani --no-capture-output`. The Makefile assumes the env is active and fails fast with a clear message if `python` isn't 3.11 inside env `gnani` (`CONDA_DEFAULT_ENV`). Never install into `base`; add dependencies to `pyproject.toml` and install with `pip install -e ".[dev]"` inside `gnani`.

**Instant human assistance — yield, don't grind.** A human is available immediately. Do NOT burn cycles, retry loops, or guesswork on any of these. Stop at once and send a short numbered instruction telling the human exactly what to run/provide, what output to paste back, and which task ID is waiting:
- credentials or secrets (`GNANI_API_KEY`, LiveKit keys), or anything hitting `api.vachana.ai` / LiveKit (`make live`, `tools/record_wire.py`);
- manual wire tests, mic/speaker/headphone/AEC checks, calibration (`assumed_downstream_ms`), human A/B rating sessions;
- external MCP tools or plugins, package/model downloads that need network (e.g. Silero ONNX), `git push`, GitHub repo setup;
- heavy documentation lookups (official Gnani docs, LiveKit/Pipecat API details) you can't resolve from `refs`;
- native-speaker review of lexicons.
Yielding for these does **not** count as a strike. Mark the task `blocked` with `blocked_reason: "awaiting human: <what>"` only if no other ready task exists; otherwise just report and wait.

**Submission bar (repo goes to GitHub for judging).** Production-grade at every commit: `mypy --strict` clean, `ruff` clean, no skipped/xfail tests, no commented-out code, no stray prints/TODOs, no secrets or large binaries, public APIs documented with docstrings. Conventional, single-purpose commits (`T<phase>.<nn>: summary`) on `main`, history linear, no `WIP` commits on `main`. README stays accurate to what actually works. Test pass rate must be 100 % in `make check`.

## 1. Commands

| Command | Purpose | Budget |
|---|---|---|
| `make fast` | Pure, hermetic, offline unit/contract/invariant/scenario tests (`-m "not slow and not live"`, `--disable-socket`, `--timeout=5`, wrapped by `tools/timed.py 15`) | **< 15 s** wall |
| `make check` | `ruff check` + `ruff format --check` + `lint-imports` → `mypy --strict` → full `pytest -m "not live"` → DAG validation (`tools/dag_lint.py validate` + `verify-commit HEAD`) | gate before commit |
| `python tools/dag_lint.py next` | Print the next ready task block (status `todo`, all `needs` done) and its `refs` | — |
| `python tools/dag_lint.py show <ID>` | Print one task block | — |
| `python tools/dag_lint.py set <ID> <field> <value>` | Safely update `status` / `strikes` / `blocked_reason` / `evidence_observed` / `commit.sha` | — |
| `make live` / `make bench` | Network / benchmark runs. **Human-triggered only.** Never run them on your own initiative. | — |

(Until T0.01/T0.02 land, these exist only as specified; T0.01 and T0.02 bootstrap them.)

## 2. Session protocol (one task, one commit, then reset)

1. Run `python tools/dag_lint.py next`. Take exactly that task (or the ID the human gave you).
2. `set <ID> status in_progress`.
3. Read **only**: this file, the task block, and the files/sections listed in the task's `refs`. Plus files in the task's `files` (to edit) and `reopens` tasks' `files` (if you must touch them).
4. Write tests first (from `done_when` + `failure_modes`), then the implementation.
5. Loop: `make fast`. Apply the **3-strike rule** (§4) on every failure.
6. When fast is green: run `make check`. It must pass.
7. Record `evidence_observed` (literal command output tails/artifact paths for every `evidence` item), set `status done`, set `commit.sha` after committing.
8. **Exactly one commit** with the task's `commit.msg` (format `T<phase>.<nn>: <summary>`), containing code + tests + the `TASKS.md` update.
9. **Stop. Reset context** (`/clear` in Claude Code, new chat in Cursor). Never chain a second task in the same context. The next session starts again at step 1.

## 3. Context & scope rules (strict)

- **One task per commit.** No drive-by refactors, no formatting sweeps, no dependency bumps outside the task.
- **Do not read files outside the task's `refs`** (+ its own `files`). No browsing the repo "to get oriented", no `grep -r` across unrelated modules, no reading other tasks' blocks. If you believe a needed file is missing from `refs`, **stop and ask the human** (or note it in `blocked_reason`); don't go exploring.
- **Write only inside the task's `files`** (and `files` of tasks in `reopens`, only to the extent needed). `tools/dag_lint.py verify-commit` fails the commit if you touch anything else — except `TASKS.md` (only your own block).
- A task listing `reopens: [X]` is authorized to modify X's files; you **must** keep X's tests green (`make fast` covers them) and mention it in the commit body.
- **Architecture invariants are law:**
  - `overlap/core/**` and `overlap/ports/**`: pure. No `asyncio`, sockets, `time`, unseeded `random`, network/model libs (AST-tested).
  - No `sleep` in `core`/`runtime`. Time comes from the injected `Clock`.
  - Constants come from `Tuning`; tests import them. Never hardcode 0.3 / 400 / 40 etc.
  - Fast path (`mic frame → VAD → EchoGate → FSM → SetGain`) is synchronous and never awaits.
  - Every in-flight stream has a `CancelToken` and an `epoch`; stale-epoch events are dropped.
- **Never invent Gnani wire details.** Endpoints, auth, message schemas come from `docs/WIRE.md` (T3.01) only. Fixtures flagged `"assumed": true` can't justify finishing an adapter task.
- **Secrets:** env only (`.env.example`). Never print, log, or commit keys. Never run anything hitting `api.vachana.ai` unless the task is `live`-gated and a human started the session.
- **Tests are the spec.** Never weaken, delete, skip (`xfail`/`skip`), or loosen an assertion/threshold/`Tuning` default to make a test pass. If a test is wrong, say so in `blocked_reason` and halt.
- Keep docstrings/comments that your change doesn't touch. Python ≥ 3.11, `mypy --strict`, frozen dataclasses with `slots=True`, small pure functions.

## 4. 3-Strike Circuit Breaker (non-negotiable)

A **strike** = one run of `make fast` that fails while working on the current task. Strikes are **consecutive**; a green `make fast` resets to 0. Persist the count in the task's `strikes` field via `dag_lint.py set` after **every** failed run (context may be reset; the file is the memory).

On the **3rd consecutive failure** on a single task you MUST:

1. **Halt execution immediately.** Do not try a 4th run, do not "just quickly" tweak, do not start another task — not even an independent one.
2. `python tools/dag_lint.py set <ID> status blocked`.
3. Fill `blocked_reason` with: the failing test IDs; the last error/traceback tail (≤ 30 lines); what each of the 3 attempts changed and why it failed; your best hypothesis for the root cause; what decision/info you need from the human (spec ambiguity? wrong test? missing ref? design flaw?).
4. Save the work-in-progress on a branch `blocked/<ID>` (commit WIP there), then restore `main` to the last green commit with only the `TASKS.md` change: commit message `chore(<ID>): blocked — <one line reason>`.
5. **Yield to the human architect** with a ≤ 10-line summary. Wait for instructions.

Also counts as a trip (same procedure): `make check` failing 3 consecutive times on one task after `make fast` is green.

**Forbidden while striking:** editing/skipping/loosening tests, changing `Tuning` defaults, adding sleeps/retries to mask races, reading files beyond `refs`, or marking the task done without green evidence.

A human unblocks by editing the block (`status: todo`, `strikes: 0`, `blocked_reason: ""`, possibly adjusting `refs`/`done_when`).

## 5. Definition of done (all must hold)

- Every `done_when` item is covered by a test or command, and every `failure_modes` item has at least one test.
- `make fast` green and under 15 s; `make check` green.
- `evidence_observed` filled with real output; `commit.sha` set; only permitted files changed; one commit.
- `TASKS.md` still validates (`dag_lint.py validate`: acyclic, `needs`/`reopens` resolve, statuses legal, `done` ⇒ evidence + sha present).

## 6. Conventions

- Layout and import rules: `ARCHITECTURE.md#2`. Tests mirror source: `tests/unit/core/test_<module>.py`.
- Fakes live in `tests/fakes/`, fixtures in `tests/fixtures/`. Use `VirtualClock` + `run_virtual`; no real sleeps; no network.
- Mark tests: `@pytest.mark.slow` (> 1 s), `live`, `replay`, `bench`. Fast suite must remain < 15 s — if your task pushes it over, fix the tests, not the budget.
- Commit body (optional) lists: files reopened, invariants touched (INV-IDs), evidence summary.
- When uncertain about intent: stop and ask. Don't guess about wire formats, thresholds, or scope.
