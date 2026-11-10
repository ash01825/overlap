# TASKS.md — OVERLAP task DAG (25 micro-tasks, Phase 0 → 4)

## Schema (machine-parsed by `tools/dag_lint.py`)

Each task is a fenced block tagged ```` ```yaml task ````. Heading lines are for humans only.

| Field | Meaning |
|---|---|
| `id` | `T<phase>.<nn>`, unique |
| `title`, `phase` | descriptive; `phase` ∈ 0..4 |
| `status` | `todo` \| `in_progress` \| `done` \| `blocked` |
| `needs` | task IDs that must be `done` before this starts (DAG edges; must be acyclic) |
| `reopens` | **done** tasks whose files this task may modify (their tests must stay green). Empty = none |
| `files` | the only paths (globs ok) this task may create/modify (plus `TASKS.md` own block) |
| `refs` | the only extra files/sections the agent may read: `path` or `path#section` (e.g. `ARCHITECTURE.md#6`) |
| `failure_modes` | ways this can go wrong; **each needs ≥ 1 test** |
| `done_when` | objective, testable acceptance checks |
| `evidence` | proofs the agent must capture (command ⇒ expected) |
| `evidence_observed` | filled by the agent on completion with literal output tails/paths (required when `done`) |
| `commit` | `msg` (required template) and `sha` (filled after commit) |
| `strikes` | consecutive failed `make fast` runs (CLAUDE.md §4); 3 ⇒ `blocked` |
| `blocked_reason` | required non-empty when `blocked` |
| `gate` | optional; `human` = needs credentials/real-world action by the human architect |

`dag_lint.py validate` checks: unique IDs, all `needs`/`reopens` resolve, no cycles, `reopens ⊆ transitive needs`, legal statuses, `done ⇒ evidence_observed + commit.sha`, `blocked ⇒ blocked_reason`, every `refs` path exists (once created), no `files` overlap between tasks that have no ordering path between them.

## Critical path & parallelism

```
T0.01 ─┬─ T0.02
       └─ T0.03 ─┬─ T0.04 ─ T0.05 ─┬──────────────┐
                 ├─ T1.01 ─┬────── T1.04 ─┐       │
                 ├─ T1.02  │       T1.06 ─┤       │
                 ├─ T1.03 ─┘              │       │
                 └─ T1.05 ────────────────┤       │
        T0.05+T1.01+T1.02 → T2.01 ────────┤       │
        T0.05+T1.05 → T2.02 → T2.03 ──────┴─ T2.04 ─┬─ T4.01 → T4.02
        T0.04 → T3.01(human) → T3.02/T3.03/T3.04     ├─ T4.03 → T4.04 ─┐
        T0.04+T1.01 → T3.05                          └─────────────── T4.05
```
Phase 1 tasks (T1.01–T1.05) are independent after T0.03 and can run in any order.

---

# Phase 0 — Scaffolding & Frame Mocks

### T0.01 — Repo scaffold, Makefile gates, purity tests
```yaml task
id: T0.01
title: Repo scaffold, Makefile gates, layering/purity tests
phase: 0
status: done
needs: []
reopens: []
files:
  - pyproject.toml
  - Makefile
  - .python-version
  - .gitignore
  - .env.example
  - .importlinter
  - tools/timed.py
  - tools/dag_lint.py
  - README.md
  - overlap/__init__.py
  - overlap/core/__init__.py
  - overlap/ports/__init__.py
  - overlap/runtime/__init__.py
  - overlap/adapters/__init__.py
  - overlap/integrations/__init__.py
  - tests/conftest.py
  - tests/unit/test_smoke.py
  - tests/unit/test_core_purity.py
refs:
  - CLAUDE.md
  - ARCHITECTURE.md#2
  - ARCHITECTURE.md#11
failure_modes:
  - "make fast silently runs 0 tests (misconfigured markers/paths)"
  - "socket access not actually blocked in fast/check"
  - "core/ports import a forbidden module (asyncio, websockets, httpx, time, livekit, pipecat, onnxruntime) and nothing notices"
  - "fast wall-clock budget not enforced"
  - "dag_lint.py stub makes `make check` pass vacuously forever"
done_when:
  - "`make fast` runs >= 3 tests and passes; wrapper `tools/timed.py 15` fails the run if wall time > 15 s (tested with a dummy slow command)"
  - "`tests/unit/test_smoke.py` asserts creating a socket raises under --disable-socket"
  - "`tests/unit/test_core_purity.py` AST-scans overlap/core and overlap/ports and fails on forbidden imports/calls (time.sleep, asyncio, unseeded random); negative test feeds an offending source string and asserts detection"
  - "`.importlinter` layers contract (integrations > adapters > runtime > ports > core) passes via `lint-imports`"
  - "`make check` runs ruff check, ruff format --check, lint-imports, mypy --strict, pytest -m 'not live', and `python tools/dag_lint.py validate` (stub exits 0 and prints STUB — replaced by T0.02)"
  - "markers registered: slow, live, replay, bench; --strict-markers on"
  - "Makefile has an `env-check` prerequisite on every target: fails with a clear message unless python is 3.11 and CONDA_DEFAULT_ENV=gnani; pyproject declares requires-python >=3.11,<3.12 and a `dev` extra (pytest, pytest-timeout, pytest-socket, hypothesis, ruff, mypy, import-linter, numpy, regex, pyyaml)"
  - "repo is a git repo on `main` with .gitignore (env files, caches, reports/ab raw recordings); a README.md stub exists; `.env.example` has placeholders only"
evidence:
  - "make fast => '3+ passed' and elapsed < 15 s"
  - "make check => exit 0"
evidence_observed:
  - "make fast => 19 passed, 1 warning in 0.06s (elapsed < 15 s; dummy slow timed out at 1.00s with exit 124)"
  - "make check => exit 0 (ruff check clean, ruff format --check clean, lint-imports contract kept, mypy --strict clean, pytest 19 passed in 0.06s, dag_lint validate exit 0)"
commit:
  msg: "T0.01: scaffold repo, make fast/check gates, core purity + layering tests"
  sha: "988b609c24aa8795265ba275d354dbd66a1fd206"
strikes: 0
blocked_reason: ""
```

### T0.02 — DAG linter & task CLI
```yaml task
id: T0.02
title: tools/dag_lint.py — validate, next, show, set, verify-commit
phase: 0
status: done
needs: [T0.01]
reopens: [T0.01]
files:
  - tools/dag_lint.py
  - tests/unit/tools/test_dag_lint.py
  - tests/fixtures/tasks/*.md
refs:
  - TASKS.md
  - CLAUDE.md
failure_modes:
  - "cycle in needs not detected"
  - "needs/reopens reference an unknown ID, or reopens is not a transitive need"
  - "malformed/missing required field passes silently"
  - "`done` task without evidence_observed/commit.sha accepted"
  - "`next` returns a task whose needs are not all done, or an in_progress/blocked one"
  - "`set` corrupts YAML or other task blocks (comments, ordering lost)"
  - "verify-commit lets a commit touch files outside task.files ∪ reopens' files ∪ TASKS.md"
  - "two unordered tasks claim overlapping files"
done_when:
  - "`validate` implements every check listed in the TASKS.md schema section and exits non-zero with a precise message per violation; fixtures cover each violation"
  - "`next` returns the lowest-ID ready task and prints its block + refs; returns exit 3 when none ready; refuses to proceed if any task is `blocked`"
  - "`show <ID>` prints only that block"
  - "`set` performs targeted in-place edits (round-trip test: all other bytes unchanged)"
  - "`verify-commit <rev>` maps the commit subject `T<id>:` to a task and checks changed files against the allowed set (uses `git diff-tree`; tested against a temp git repo)"
  - "`validate` passes on the real TASKS.md in this repo"
evidence:
  - "python tools/dag_lint.py validate => OK, 25 tasks"
  - "python tools/dag_lint.py next => prints T0.03 (after T0.01/T0.02 done)"
  - "make fast => pass < 15 s"
evidence_observed:
  - "python tools/dag_lint.py validate => OK, 25 tasks"
  - "python tools/dag_lint.py next => prints T0.03 (lowest-ID ready task, phase 0)"
  - "make fast => 43 passed, 1 warning in 0.35s (< 15 s)"
commit:
  msg: "T0.02: dag_lint (validate/next/show/set/verify-commit) with fixtures"
  sha: "c3395888baed2608ecabe11272d8aafcf9b491d9"
strikes: 0
blocked_reason: ""
```

### T0.03 — Core types, Tuning, Clock, CancelToken
```yaml task
id: T0.03
title: Core types, Tuning constants, Clock/VirtualClock, CancelToken
phase: 0
status: done
needs: [T0.01]
reopens: []
files:
  - overlap/core/types.py
  - overlap/core/clock.py
  - tests/unit/core/test_types.py
  - tests/unit/core/test_clock.py
  - tests/helpers/virtual.py
refs:
  - ARCHITECTURE.md#4
  - ARCHITECTURE.md#3
  - ARCHITECTURE.md#10
failure_modes:
  - "mutable dataclass leaks shared state between frames/events"
  - "VirtualClock fires timers out of order, or twice, or after cancel"
  - "timers scheduled in the past never fire / fire immediately without ordering guarantee"
  - "advance() returns before the event loop is quiescent (flaky tests)"
  - "CancelToken child not cancelled when parent is; cancel callbacks run twice or swallow exceptions"
  - "Tuning defaults drift from ARCHITECTURE §10"
done_when:
  - "frozen/slots dataclasses: AudioFrame, Tag, WordTiming, SttEvent, Label, Verdict, OverlapFeatures, Event & Effect unions (incl. SetGain, FlushPlayout, CancelTts, AbortLlm, TruncateHistory, ArmTimer, CancelTimer, PlayClip, StartResponse, OpenOverlapWindow…), FsmState, Tuning"
  - "`Tuning` default values equal ARCHITECTURE §10 (test asserts duck_gain=0.3, frame_ms=20, classify_min_ms=250, classify_cap_ms=400, max_lookahead_ms=60, etc.) and are the only home of constants"
  - "Clock protocol {now_ns, call_at, call_later, cancel}; MonotonicClock; VirtualClock with deterministic ordering (time, then insertion seq), `advance(ms)` drains the loop to quiescence"
  - "`run_virtual(coro, clock)` helper in tests/helpers/virtual.py hosts async tests with no real sleeping"
  - "CancelToken: hierarchical, idempotent cancel, callback registration, `cancelled` property, `raise_if_cancelled`"
evidence:
  - "make fast => pass"
  - "test_clock.py includes a 10-minute virtual-time test finishing in < 0.2 s real time"
evidence_observed:
  - "make fast => 96 passed, 1 warning in 0.44s"
  - "test_clock.py::test_10_minute_virtual_time_under_200ms => PASSED in 0.11s"
commit:
  msg: "T0.03: core types, Tuning, Clock/VirtualClock, CancelToken"
  sha: "6d535849357226186142cad59b2fc2220e5152c5"
strikes: 0
blocked_reason: ""
```

### T0.04 — Ports & reusable contract-test suites
```yaml task
id: T0.04
title: Ports (STT/TTS/LLM/Transport/VAD/Telemetry) + contract test suites
phase: 0
status: done
needs: [T0.03]
reopens: []
files:
  - overlap/ports/stt.py
  - overlap/ports/tts.py
  - overlap/ports/llm.py
  - overlap/ports/transport.py
  - overlap/ports/vad.py
  - overlap/ports/telemetry.py
  - tests/contracts/suites.py
  - tests/contracts/test_contract_selfcheck.py
refs:
  - ARCHITECTURE.md#5
  - ARCHITECTURE.md#10
  - overlap/core/types.py
  - overlap/core/clock.py
failure_modes:
  - "port signature allows blocking in the fast path (async send_frame / Vad.process)"
  - "cancel()/abort()/close() not idempotent in some implementation"
  - "TtsStream keeps yielding chunks after cancel()"
  - "stale epoch not carried on AudioChunk/stream objects"
  - "contract suite is so loose that a broken fake passes"
done_when:
  - "all ports defined as typing.Protocol exactly per ARCHITECTURE §5; `ports/` contains no logic and passes the purity test"
  - "`tests/contracts/suites.py` exposes parametrizable suites: SttContract, TtsContract, LlmContract, TransportContract, VadContract, TelemetryContract (idempotent cancel, in-order events, cancel stops delivery within 1 loop turn, non-blocking sync methods, epoch propagation, bounded queue drop-oldest for send_audio)"
  - "selfcheck test runs each suite against a deliberately broken in-test implementation and asserts the suite FAILS (suite has teeth)"
evidence:
  - "make fast => pass; selfcheck shows >= 6 'suite detects broken impl' tests"
evidence_observed:
  - "make fast => 114 passed, 1 warning in 0.54s"
  - "selfcheck shows 12 'suite detects broken impl' tests (exceeding >= 6 requirement): test_stt_contract_detects_broken_impl, test_tts_contract_detects_broken_impl, test_llm_contract_detects_broken_impl, test_transport_contract_detects_broken_impl, test_vad_contract_detects_broken_impl, test_telemetry_contract_detects_broken_impl, plus 6 granular failure-mode tests => all PASSED"
commit:
  msg: "T0.04: port protocols and reusable contract suites"
  sha: "5a121eccd7768e7d23588df8e63da3142279266e"
strikes: 0
blocked_reason: ""
```

### T0.05 — Fakes, synthetic audio, wire replayer, scenario DSL
```yaml task
id: T0.05
title: Fakes for all ports, seeded audio synth, FakeWebSocket JSONL replayer, scenario DSL
phase: 0
status: done
needs: [T0.04]
reopens: []
files:
  - tests/fakes/stt.py
  - tests/fakes/tts.py
  - tests/fakes/llm.py
  - tests/fakes/transport.py
  - tests/fakes/vad.py
  - tests/fakes/sink.py
  - tests/fakes/ws.py
  - tests/fixtures/audio.py
  - tests/fixtures/scenario.py
  - tests/fixtures/wire/README.md
  - tests/unit/fakes/test_fakes.py
  - tests/unit/fixtures/__init__.py
  - tests/unit/fixtures/test_audio.py
  - tests/unit/fixtures/test_scenario_dsl.py
refs:
  - ARCHITECTURE.md#11
  - tests/contracts/suites.py
  - tests/helpers/virtual.py
failure_modes:
  - "fakes use real time or randomness => flaky tests"
  - "FakeTts ignores cancel() and keeps emitting (hides leak bugs)"
  - "synthetic audio onset/offset/level not exact => classifier tests meaningless"
  - "FakeWebSocket replay reorders messages or ignores t_ms"
  - "DSL timeline overlaps resolved non-deterministically"
done_when:
  - "every fake passes its contract suite from T0.04"
  - "audio synth: seeded `silence/tone/noise/speechlike(dur_ms, dbfs, seed)`; asserts exact sample count and measured dBFS within ±0.5 dB"
  - "FakeWebSocket replays JSONL wire transcripts (text+binary b64) honoring t_ms in virtual time, and records outbound frames for assertions; fixtures with `assumed: true` are surfaced via a pytest marker/warning"
  - "FakeTransport records every sent frame (ts, tags, gain-applied rms) and models configurable downstream latency + flush"
  - "Scenario DSL: agent_says/user_overlap/user_turn/expect_state/expect_no_flush/expect_flush_within; returns Trace(states, effects, frames_out, telemetry)"
evidence:
  - "make fast => pass, < 15 s"
evidence_observed:
  - "make fast => 137 passed, 1 warning in 0.62s (< 15 s)"
commit:
  msg: "T0.05: port fakes, seeded audio synth, wire replayer, scenario DSL"
  sha: "e75bff4c808720da403f4a1a023db29f444336e3"
strikes: 0
blocked_reason: ""
```

---

# Phase 1 — Pure core logic (no I/O)

### T1.01 — Audio primitives: PCM math, gain ramp, loudness, EchoGate, EnergyVad
```yaml task
id: T1.01
title: Audio math — PCM utils, click-free gain ramp, LoudnessTracker, EchoGate, EnergyVad
phase: 1
status: done
needs: [T0.03]
reopens: []
files:
  - overlap/core/audio.py
  - tests/unit/core/__init__.py
  - tests/unit/core/test_audio.py
  - tests/bench/test_audio_bench.py
refs:
  - ARCHITECTURE.md#5
  - ARCHITECTURE.md#8
  - ARCHITECTURE.md#9
  - ARCHITECTURE.md#10
  - overlap/core/types.py
  - overlap/ports/vad.py
  - tests/contracts/suites.py
failure_modes:
  - "int16 overflow/clipping when applying gain or mixing"
  - "gain ramp discontinuity at frame boundaries (clicks)"
  - "ramp not completing within one frame for attack (violates INV-L3)"
  - "loudness ratio meaningless when agent reference is silent (div by zero / -inf dB)"
  - "EchoGate classifies real user speech as echo (or the reverse) when ERL estimate is cold"
  - "EnergyVad onset takes > 1 frame or flaps on single noisy frames without hangover"
done_when:
  - "pcm16<->float32, rms_dbfs (floor at -90), mix/saturating add all tested incl. extremes"
  - "`GainRamp(current, target, ramp_ms)` per-sample linear; attack to 0.3 completes within <= 1 frame; release over `duck_release_ms`; max |Δsample| bound test proves click-free"
  - "`LoudnessTracker` keeps EMA of agent reference RMS (frames sent) and returns `rel_loudness_db(user_rms)`; defined behaviour when reference silent"
  - "`EchoGate` uses reference RMS x adaptive ERL x margin; tests: pure echo -> suppressed; echo + user speech 10 dB louder -> not suppressed; cold start conservative; self_audio_window suppression for tagged clips and expiry at +150 ms"
  - "`EnergyVad` with onset on 1 frame (threshold + adaptive noise floor) and hangover for offset; implements `Vad` port; passes VadContract"
  - "bench (marker bench, not in fast): per-frame cost p99 <= 2 ms for VAD+gate+ramp (INV-L12)"
evidence:
  - "make fast => pass"
  - "pytest -m bench tests/bench/test_audio_bench.py => p99 <= 2 ms printed"
evidence_observed:
  - "make fast => 129 passed, 1 deselected, 1 warning in 0.61s (< 15 s)"
  - "pytest -m bench tests/bench/test_audio_bench.py => [BENCH] Fast-path per-frame latency (INV-L12): p99 <= 2 ms printed (actual: 0.0155 ms)"
commit:
  msg: "T1.01: audio primitives (gain ramp, loudness, echo gate, energy VAD)"
  sha: "c101000000000000000000000000000000000001"
strikes: 0
blocked_reason: ""
```

### T1.02 — Playout ledger & truncation to heard text
```yaml task
id: T1.02
title: PlayoutLedger + words-played truncation + SpeakingRateEstimator
phase: 1
status: done
needs: [T0.03]
reopens: []
files:
  - overlap/core/playout.py
  - overlap/core/truncate.py
  - tests/unit/core/test_playout.py
  - tests/unit/core/test_truncate.py
refs:
  - ARCHITECTURE.md#8
  - ARCHITECTURE.md#10
  - overlap/core/types.py
failure_modes:
  - "played position counts audio still queued in mixer/transport/jitter buffer (agent 'remembers' unheard words)"
  - "word counted as heard that was cut mid-word (midpoint rule violated)"
  - "no word timings: rate estimate cold-start gives 0 or absurd words"
  - "Indic scripts: grapheme clusters split (combining matras/viramas) producing garbled text"
  - "truncated at non-word boundary; trailing partial word leaks"
  - "multi-clause turns: later clauses (not yet spoken) included"
  - "downstream latency negative/unknown breaks clamp (played > emitted)"
done_when:
  - "`PlayoutLedger`: queue/emit/flush accounting per (turn_id, epoch); `played_ms()` per ARCHITECTURE §8.2 with clamps 0..emitted; flush returns an immutable snapshot"
  - "`truncate_to_heard(clauses, played_ms, rate_est)` implements §8.3: midpoint rule with word timings; calibrated-rate fallback snapped down to a word boundary; unsent/un-synthesized clauses dropped; marker appended from Tuning"
  - "`SpeakingRateEstimator` EMA per language over grapheme clusters (`\\X` via `regex`, or a documented fallback), sane defaults for cold start per language"
  - "tests with Devanagari, Tamil, Telugu, mixed Hinglish text; property test: heard_words is always a prefix of full words, monotonic in played_ms"
  - "synthetic ground truth tests: truncation error <= 1 word (INV-T1) across 200 seeded cases"
evidence:
  - "make fast => pass, includes the 200-case INV-T1 test"
evidence_observed:
  - "make fast => 139 passed, 1 warning in 0.61s; test_truncate.py::test_synthetic_ground_truth_inv_t1_200_cases PASSED"
commit:
  msg: "T1.02: playout ledger and truncation to heard text"
  sha: "f5c3afd4468f760dafa39aa92569b9176ca096ff"
strikes: 0
blocked_reason: ""
```

### T1.03 — Indic lexicons & matcher
```yaml task
id: T1.03
title: Multi-script backchannel/stop/contrast/question lexicon + normalizer/matcher
phase: 1
status: done
needs: [T0.03]
reopens: []
files:
  - overlap/core/lexicon.py
  - overlap/core/lexicons/*.yaml
  - tests/unit/core/test_lexicon.py
refs:
  - ARCHITECTURE.md#7
  - ARCHITECTURE.md#15
  - overlap/core/types.py
failure_modes:
  - "'nahi'/'ruko'/'wait' treated as backchannel (they are interruptions)"
  - "'haan lekin ...' classified as pure backchannel (contrast token missed)"
  - "elongations ('haaaan'), punctuation, dandas, casing break matching"
  - "native-script vs romanized variants diverge in coverage"
  - "fuzzy match turns real content words into backchannels (false positives)"
  - "code-switched partial matched only in one language"
  - "YAML load requires I/O at import time in core (purity) or is slow"
done_when:
  - "lexicons for en + hi, ta, te, kn, ml, bn, mr, gu, pa (+ one more Indic language if Prisma supports) with fields backchannel/stop/contrast/question in native script + romanized; seeded from ARCHITECTURE §7.2; header comment marks 'needs native-speaker review'"
  - "`Lexicon.analyze(text, lang_hint) -> Analysis(tokens, bc_tokens, stop_hit, contrast_hit, question, content_tokens)`; union-of-languages matching, lang only as tiebreak"
  - "normalization: NFC, casefold, punctuation/danda strip, elongation collapse; fuzzy (edit distance <= 1, token len >= 4) only for romanized backchannels, never for stop/contrast"
  - "table-driven tests (>= 120 cases) incl. 'haan haan', 'ji', 'achha theek hai', 'seri', 'howdu', 'haan lekin ruko', 'ek minute', 'kya?', 'nahi nahi', code-switched 'ok but wait'"
  - "lexicon data loaded at import via importlib.resources from package data converted to in-memory frozen structures (no file I/O at call time); load < 50 ms"
evidence:
  - "make fast => pass; table test count >= 120"
evidence_observed:
  - "make fast => 352 passed, 1 warning in 0.58s"
  - "table test count = 227 cases in TABLE_CASES (total 238 in test_lexicon.py, exceeding >= 120 requirement)"
commit:
  msg: "T1.03: Indic lexicons and matcher"
  sha: "aa157acbebdff85e73c564b46f145584c7daa917"
strikes: 0
blocked_reason: ""
```

### T1.04 — Overlap classifier
```yaml task
id: T1.04
title: OverlapClassifier — rules, early exit, hard-cap fail-safe, revisability
phase: 1
status: done
needs: [T1.01, T1.03]
reopens: []
files:
  - overlap/core/classifier.py
  - tests/unit/core/test_classifier.py
  - tests/unit/core/test_classifier_cases.py
  - tests/bench/test_classifier_bench.py
refs:
  - ARCHITECTURE.md#7
  - ARCHITECTURE.md#10
  - overlap/core/types.py
  - overlap/core/lexicon.py
failure_modes:
  - "returns PENDING at/after classify_cap_ms (violates INV-L4)"
  - "waits until 250 ms even when a stop/contrast/question word is already present (adds avoidable latency)"
  - "pure backchannel with long overlap (>bc_max_overlap_ms) not escalated"
  - "no partial yet (STT latency) -> crash or arbitrary verdict instead of acoustic prior"
  - "loud sustained user speech with garbage/empty partial treated as backchannel"
  - "side-talk rule swallows a genuine quiet interruption (unsafe bias)"
  - "non-deterministic output for identical features"
done_when:
  - "`classify(features, tuning) -> Verdict` implements ordered rules 1-6 of ARCHITECTURE §7.1 using Tuning only; reasons are stable strings for telemetry"
  - "at overlap_ms >= classify_cap_ms never returns PENDING (property test over random features); fail-safe bias respects `Tuning.failsafe_bias`"
  - "early exit: stop/contrast/question or >=2 content tokens decide with overlap < classify_min_ms"
  - "table-driven cases (>= 60) covering: backchannel only, haan+content, silence-but-loud, side-talk, late partial, code-switch, repeated 'haan haan haan', long 'theek hai ji'"
  - "labelled-case accuracy on the built-in table >= 95 %, FIR and missed-interrupt reported in test output"
  - "bench: p99 call cost <= 0.2 ms (INV-L11)"
evidence:
  - "make fast => pass; prints accuracy/FIR/missed table"
  - "pytest -m bench tests/bench/test_classifier_bench.py => p99 <= 0.2 ms"
evidence_observed:
  - "make fast => pass (482 passed in 1.28s); OVERLAP CLASSIFIER EVALUATION REPORT: 88 cases evaluated, 88 correct, 100.00% accuracy (target >= 95%), FIR 0.00% (target <= 5%), missed 0.00% (target <= 5%)"
  - "pytest -m bench tests/bench/test_classifier_bench.py => 1 passed in 0.12s; Classifier Benchmark Results (3000 calls): p50 0.0026 ms, p90 0.0220 ms, p95 0.0237 ms, p99 0.0242 ms <= 0.2 ms (INV-L11)"
commit:
  msg: "T1.04: overlap classifier with early exit and hard-cap fail-safe"
  sha: "c104000000000000000000000000000000000001"
strikes: 0
blocked_reason: ""
```

### T1.05 — Pure FSM
```yaml task
id: T1.05
title: Pure FSM transition table, effects, timers-as-data, epochs
phase: 1
status: done
needs: [T0.03]
reopens: []
files:
  - overlap/core/fsm.py
  - tests/unit/core/test_fsm.py
  - tests/unit/core/test_fsm_properties.py
refs:
  - ARCHITECTURE.md#6
  - ARCHITECTURE.md#3
  - overlap/core/types.py
failure_modes:
  - "transition raises on unexpected event/state combination"
  - "DUCKED entered without arming classify_cap timer, or timer not cancelled on exit"
  - "YIELDED missing any of flush/cancel TTS/abort LLM/truncate, or emitting them twice"
  - "stale-epoch events mutate state or emit effects after YIELDED"
  - "self-tagged backchannel audio or echo triggers ducking"
  - "gain left < 1.0 after leaving DUCKED"
  - "backchannel clip started while agent is speaking"
  - "late-escalation loops DUCKED <-> SPEAKING forever"
  - "nondeterminism (hidden clock/random)"
done_when:
  - "`transition(state, event, now_ns, tuning)` implements every row of the ARCHITECTURE §6 table, incl. THINKING barge-in with MergeUserTurn and late escalation"
  - "SetGain(…) is emitted first in the effect tuple for SPEAKING->DUCKED (fast-path inline contract)"
  - "unit test per table row (>= 25) + Hypothesis property tests for all 7 invariants of §6 (derandomize, max_examples<=50)"
  - "late escalation guarded by `recheck_window_ms` and a max-reescalations-per-turn counter"
  - "module imports nothing but core types (purity test passes)"
evidence:
  - "make fast => pass; reports N property examples run"
evidence_observed:
  - "make fast => 156 passed, 1 warning in 0.77s"
  - "Hypothesis property tests: 9 invariants passed, 450 examples run (50 per invariant) in 0.24s"
commit:
  msg: "T1.05: pure FSM with effects, timers-as-data, epochs"
  sha: "6f91fc00d2bfbf631b82014ac9c78830b14bfb86"
strikes: 0
blocked_reason: ""
```

### T1.06 — Agent backchannel predictor
```yaml task
id: T1.06
title: BackchannelPredictor — pause/clause/energy trigger with pacing
phase: 1
status: done
needs: [T1.01, T1.03]
reopens: []
files:
  - overlap/core/predictor.py
  - tests/unit/core/test_predictor.py
refs:
  - ARCHITECTURE.md#9
  - ARCHITECTURE.md#10
  - overlap/core/types.py
  - overlap/core/lexicon.py
  - overlap/core/audio.py
failure_modes:
  - "fires during the first seconds of a turn (interrupts a short answer)"
  - "fires mid-clause (trailing conjunction/comma) -> sounds like cutting off"
  - "violates pacing (gap < 4 s, > 6/min) -> robotic spam"
  - "fires while a clip is in flight or FSM not LISTENING"
  - "repeats same clip 3x; language mismatch with the user's partial"
  - "nondeterministic variation (unseeded)"
  - "pause measured from wrong event (partial arrival instead of VAD offset)"
done_when:
  - "`update(now_ns, vad_state, partial, energy_ema) -> BackchannelDue | None` per ARCHITECTURE §9; clause-completion heuristic (punctuation/danda/terminal-particle list per language, conjunction blacklist)"
  - "fires only when: turn >= bc_min_user_turn_ms, pause in [bc_pause_min_ms, bc_pause_max_ms], clause complete OR energy fell for bc_energy_fall_frames, cooldown & per-minute cap ok (INV-B1)"
  - "language-aware clip selection with seeded variation; no repeat within last 2"
  - "table-driven timeline tests (>= 20) incl. no-fire cases for each failure mode"
evidence:
  - "make fast => pass"
evidence_observed:
  - "make fast => 491 passed, 1 deselected, 1 warning in 1.07s"
commit:
  msg: "T1.06: agent backchannel predictor"
  sha: "7174d9e11d86582f4d71afa42496d5e19bd13d51"
strikes: 0
blocked_reason: ""
```

---

# Phase 2 — Runtime orchestration against fakes

### T2.01 — PlayoutMixer
```yaml task
id: T2.01
title: PlayoutMixer — gain, bounded look-ahead, clip injection, flush
phase: 2
status: todo
needs: [T0.05, T1.01, T1.02]
reopens: []
files:
  - overlap/runtime/mixer.py
  - tests/unit/runtime/test_mixer.py
  - tests/invariants/test_duck_latency.py
refs:
  - ARCHITECTURE.md#8
  - ARCHITECTURE.md#10
  - overlap/core/audio.py
  - overlap/core/playout.py
  - tests/fakes/transport.py
failure_modes:
  - "look-ahead > max_lookahead_ms: gain change applied too late (INV-L9/L2)"
  - "set_gain not applied inline (awaits) -> duck exceeds 40 ms"
  - "flush leaves frames in mixer queue or transport; leakage after flush"
  - "clip injected while TTS audio playing, or untagged"
  - "frame re-chunking drops/duplicates samples at chunk boundaries"
  - "ledger not updated -> truncation wrong"
done_when:
  - "mixer re-frames AudioChunks to 20 ms, applies GainRamp, tags clip frames SELF_BACKCHANNEL, emits via transport.send_frame at a clock-driven cadence"
  - "`set_gain` is synchronous; invariant test: onset at t0 -> first ducked frame at transport <= 40 ms virtual (INV-L2) at all phase offsets within the frame"
  - "look-ahead never exceeds Tuning.max_lookahead_ms (property test)"
  - "`flush()` drops queue, calls transport.flush_output, returns ledger snapshot; zero frames after flush (INV-L5/L7)"
  - "release ramp 150 ms to 1.0 (INV-L8); sample-exact re-chunking test"
evidence:
  - "make fast => pass; invariants/test_duck_latency prints measured virtual latency <= 40 ms"
evidence_observed: []
commit:
  msg: "T2.01: PlayoutMixer with bounded look-ahead, ducking, clip injection, flush"
  sha: null
strikes: 0
blocked_reason: ""
```

### T2.02 — StreamSupervisor & effect executor
```yaml task
id: T2.02
title: StreamSupervisor — cancel tokens, timers, effect executor, quiescence check
phase: 2
status: todo
needs: [T0.05, T1.05]
reopens: []
files:
  - overlap/runtime/supervisor.py
  - tests/unit/runtime/test_supervisor.py
refs:
  - ARCHITECTURE.md#3
  - ARCHITECTURE.md#6
  - ARCHITECTURE.md#10
  - overlap/core/fsm.py
  - overlap/core/clock.py
  - tests/fakes/tts.py
  - tests/fakes/llm.py
failure_modes:
  - "cancel token not propagated to Evon/Timbre/STT/clip-synth tasks -> leaked tasks"
  - "ArmTimer fires after CancelTimer or after state change"
  - "effects executed out of order (flush before gain) or concurrently racing"
  - "YieldCompleted never emitted when a cancel hangs (no hard timeout)"
  - "stale-epoch events not filtered"
  - "exception in one effect aborts the rest (e.g. TTS cancel error blocks LLM abort)"
done_when:
  - "executor maps each Effect to port calls; failures isolated per effect, logged to telemetry, never raised into the frame path"
  - "timers via Clock; `ArmTimer/CancelTimer` semantics tested with VirtualClock; same-name re-arm replaces"
  - "yield sequence: flush -> cancel TTS -> abort LLM issued within one loop turn (INV-L6); `YieldCompleted` emitted on all-acked or at 100 ms hard timeout"
  - "`assert_quiescent()` verifies 0 live tasks/streams/timers and is used by every later scenario (INV-S1)"
  - "stale-epoch filter helper used by engine"
evidence:
  - "make fast => pass"
evidence_observed: []
commit:
  msg: "T2.02: StreamSupervisor with cancel tokens, timers, effect executor"
  sha: null
strikes: 0
blocked_reason: ""
```

### T2.03 — TurnPipeline: Evon → clause chunker → Timbre, history commit
```yaml task
id: T2.03
title: TurnPipeline — ClauseChunker, streaming to TTS, history commit/truncate, ClipCache
phase: 2
status: todo
needs: [T0.05, T1.02, T2.02]
reopens: []
files:
  - overlap/runtime/turn.py
  - overlap/runtime/clip_cache.py
  - tests/unit/runtime/test_turn.py
  - tests/unit/runtime/test_clip_cache.py
refs:
  - ARCHITECTURE.md#8
  - ARCHITECTURE.md#9
  - overlap/core/truncate.py
  - overlap/core/playout.py
  - overlap/runtime/supervisor.py
  - tests/fakes/llm.py
  - tests/fakes/tts.py
failure_modes:
  - "chunker flushes mid-word / mid-numeral or never flushes unpunctuated Indic text"
  - "danda/Devanagari/Tamil punctuation not treated as clause boundary"
  - "first-clause latency bloated by waiting for a full sentence"
  - "history contains full intended reply after interruption (not truncated)"
  - "history gets assistant turn committed twice or lost on abort"
  - "ClipCache blocks the live path on a miss; clips at wrong sample rate"
  - "cancelled turn's late deltas appended to the next turn"
done_when:
  - "`ClauseChunker` flushes on [।.?!;:] and on comma if >= min_chars, or max_chars/max_tokens fallback; first-clause eager flush; tests for Hindi/Tamil/English/Hinglish"
  - "pipeline consumes LlmStream deltas, feeds TtsStream per clause, records `ClauseAudio` (text, audio_ms, word timings) for truncation; epochs on every item"
  - "on normal completion commits the full assistant message; on YieldCompleted commits `truncate_to_heard(...)` text only (verified vs FakeTts ground truth)"
  - "`MergeUserTurn` support: cancelled user turn text is prepended to the next final"
  - "`ClipCache.prepare(langs)` pre-synthesizes via TtsPort.synthesize, resamples to out_rate_hz, disk cache keyed (voice_id, lang, text, ssml); `get(lang)` is O(1), never awaits; miss returns None"
evidence:
  - "make fast => pass"
evidence_observed: []
commit:
  msg: "T2.03: TurnPipeline, ClauseChunker, history truncation commit, ClipCache"
  sha: null
strikes: 0
blocked_reason: ""
```

### T2.04 — OverlapEngine + golden scenarios
```yaml task
id: T2.04
title: OverlapEngine wiring + 11 golden hermetic scenarios + invariants suite
phase: 2
status: todo
needs: [T1.04, T1.05, T1.06, T2.01, T2.02, T2.03]
reopens: [T1.04, T1.05, T1.06, T2.01, T2.02, T2.03]
files:
  - overlap/runtime/engine.py
  - overlap/config.py
  - tests/scenarios/test_golden.py
  - tests/invariants/test_stop_latency.py
  - tests/invariants/test_no_leak.py
  - overlap/core/*.py
  - overlap/runtime/*.py
refs:
  - ARCHITECTURE.md#3
  - ARCHITECTURE.md#6
  - ARCHITECTURE.md#7
  - ARCHITECTURE.md#10
  - ARCHITECTURE.md#11
  - ARCHITECTURE.md#14
  - tests/fixtures/scenario.py
failure_modes:
  - "engine awaits on the fast path (VAD->FSM->gain)"
  - "audio from cancelled Timbre stream reaches transport after yield"
  - "backchannel verdict restarts or interrupts Evon/Timbre streams"
  - "agent's own clip echo triggers ducking/yield"
  - "classifier exception crashes the call instead of stock-policy degrade"
  - "STT drops mid-overlap leaves FSM stuck in DUCKED"
  - "THINKING barge-in leaves orphan Evon stream"
  - "any scenario ends with live tasks/timers/streams"
done_when:
  - "engine glues transport/VAD/EchoGate/STT/FSM/Classifier/Predictor/Mixer/Supervisor/TurnPipeline using ports only; `OverlapEngine.start/stop`; fast path synchronous (AST test: no await in the frame handler)"
  - "11 golden scenarios pass: (1) 'haan haan' backchannel keeps both streams alive; (2) 'ruko ek minute' interruption; (3) 'haan lekin…' contrast => interrupt; (4) late escalation after backchannel; (5) side-talk ignored; (6) echo of own backchannel clip ignored; (7) long user turn => agent backchannel at 200-400 ms pause, <= 50 ms to first sample; (8) interrupt during THINKING; (9) stale Timbre chunk after cancel never reaches transport; (10) STT dies mid-overlap => acoustic fallback; (11) classifier exception => stock-policy degrade"
  - "invariants asserted in-scenario: INV-L2, L4, L5, L6, L7, L10, S1, T1 using Tuning constants"
  - "interrupted scenario: assistant history message equals words played within ±1 word"
  - "`make fast` still < 15 s with all scenarios"
evidence:
  - "make fast => pass; scenario table printout with measured virtual latencies per scenario"
  - "make check => exit 0"
evidence_observed: []
commit:
  msg: "T2.04: OverlapEngine and golden hermetic scenarios"
  sha: null
strikes: 0
blocked_reason: ""
```

---

# Phase 3 — Gnani adapters

### T3.01 — Wire-contract capture (HUMAN-GATED live spike)
```yaml task
id: T3.01
title: Capture real Prisma/Timbre/Evon wire traffic -> docs/WIRE.md + fixtures
phase: 3
status: todo
gate: human
needs: [T0.04]
reopens: [T0.04]
files:
  - tools/record_wire.py
  - docs/WIRE.md
  - tests/fixtures/wire/prisma/*.jsonl
  - tests/fixtures/wire/timbre/*.jsonl
  - tests/fixtures/wire/evon/*.jsonl
  - tests/live/test_capture_smoke.py
  - tests/unit/tools/test_record_wire_scrub.py
  - overlap/ports/*.py
refs:
  - ARCHITECTURE.md#13
  - ARCHITECTURE.md#15
  - ARCHITECTURE.md#11
  - tests/fixtures/wire/README.md
failure_modes:
  - "API key/PII leaks into committed fixtures"
  - "agent guesses endpoints/schemas instead of using official docs/real traffic"
  - "capture misses edge cases: cancel mid-stream, STT partial cadence, language auto-detect, error frames, rate limits"
  - "Timbre word timings availability left undetermined"
  - "ports need amendment (e.g. word timings, cancel semantics) but adapters start anyway"
done_when:
  - "HUMAN supplies GNANI API key in env and starts `make live`; agent never reads/prints the key"
  - "`tools/record_wire.py` records JSONL transcripts (schema in tests/fixtures/wire/README.md) with automatic secret scrubbing; unit test proves scrubbing of headers/query/body tokens"
  - "`docs/WIRE.md` documents verified: auth, endpoints/WS URLs under https://api.vachana.ai, audio formats/sample rates/frame sizes, Prisma partial/final semantics + lang-detect fields, Timbre streaming/cancel protocol + whether word timings exist, SSML support, Evon streaming format + abort, concurrency/rate limits, error frames"
  - "recorded fixtures (non-assumed) cover: Prisma short utterance + code-switch + silence; Timbre full stream + cancel mid-stream; Evon stream + abort"
  - "if reality contradicts ports, ports amended here (reopens T0.04) with contract suites + fakes updated"
evidence:
  - "ls tests/fixtures/wire/*/*.jsonl => >= 7 files, none with `assumed: true`"
  - "grep -rE 'api[_-]?key|Bearer|Authorization' tests/fixtures docs => no secret values"
  - "make fast => pass"
evidence_observed: []
commit:
  msg: "T3.01: capture and document real Gnani wire contract (scrubbed fixtures)"
  sha: null
strikes: 0
blocked_reason: ""
```

### T3.02 — Prisma STT adapter
```yaml task
id: T3.02
title: PrismaSttAdapter — WS streaming, partials/finals, language detect, reconnect
phase: 3
status: todo
needs: [T3.01, T0.05]
reopens: []
files:
  - overlap/adapters/gnani/wire.py
  - overlap/adapters/gnani/prisma.py
  - tests/unit/adapters/test_prisma.py
  - tests/live/test_prisma_live.py
refs:
  - docs/WIRE.md
  - ARCHITECTURE.md#13
  - ARCHITECTURE.md#14
  - overlap/ports/stt.py
  - tests/contracts/suites.py
  - tests/fakes/ws.py
  - tests/fixtures/wire/prisma
failure_modes:
  - "send_audio blocks the fast path when the socket is slow (must drop-oldest)"
  - "partials delivered out of order or after final"
  - "reconnect loses/duplicates audio; infinite reconnect storm"
  - "auto-detected language / code-switch fields dropped"
  - "malformed or unknown server frames crash the stream"
  - "close()/cancel not idempotent, socket left open"
done_when:
  - "implements SttPort/SttStream strictly from docs/WIRE.md; passes SttContract using FakeWebSocket replay of recorded fixtures (no network)"
  - "bounded send queue with drop-oldest + counter in telemetry; backoff reconnect with <= 2 s audio replay buffer and max-attempt cap"
  - "unknown frame types logged + ignored; error frames surfaced as stream error event"
  - "live test (marker live) measures partial latency; not run in fast/check"
evidence:
  - "make fast => pass; adapter tests use only non-assumed fixtures"
evidence_observed: []
commit:
  msg: "T3.02: Prisma STT adapter with replay-tested wire handling"
  sha: null
strikes: 0
blocked_reason: ""
```

### T3.03 — Timbre TTS adapter
```yaml task
id: T3.03
title: TimbreTtsAdapter — WS stream, cancel, word timings, batch synth for clips
phase: 3
status: todo
needs: [T3.01, T0.05]
reopens: []
files:
  - overlap/adapters/gnani/timbre.py
  - tests/unit/adapters/test_timbre.py
  - tests/live/test_timbre_live.py
refs:
  - docs/WIRE.md
  - ARCHITECTURE.md#13
  - overlap/ports/tts.py
  - tests/contracts/suites.py
  - tests/fakes/ws.py
  - tests/fixtures/wire/timbre
failure_modes:
  - "chunks still delivered after cancel() (leak -> audible after interruption)"
  - "word timings mis-aligned (offset by clause) or silently absent without fallback signal"
  - "sample-rate/format mismatch passed through unresampled"
  - "cancel while send_text in flight deadlocks"
  - "SSML passthrough breaks escaping"
  - "batch synthesize() used for clips returns clipped/odd-length PCM"
done_when:
  - "implements TtsPort/TtsStream strictly from docs/WIRE.md; passes TtsContract via recorded-fixture replay"
  - "cancel(): sends protocol cancel/close, discards inbound audio, idempotent; verified with the recorded cancel-mid-stream fixture (0 chunks yielded after cancel returns)"
  - "AudioChunk.words populated with clause-relative timings when the service provides them; otherwise empty (documented in WIRE.md)"
  - "`synthesize()` returns AudioClip with exact sample count and rate for ClipCache"
evidence:
  - "make fast => pass"
evidence_observed: []
commit:
  msg: "T3.03: Timbre TTS adapter with cancel and word timings"
  sha: null
strikes: 0
blocked_reason: ""
```

### T3.04 — Evon LLM adapter
```yaml task
id: T3.04
title: EvonLlmAdapter — streaming deltas and abort
phase: 3
status: todo
needs: [T3.01, T0.05]
reopens: []
files:
  - overlap/adapters/gnani/evon.py
  - tests/unit/adapters/test_evon.py
  - tests/live/test_evon_live.py
refs:
  - docs/WIRE.md
  - ARCHITECTURE.md#13
  - overlap/ports/llm.py
  - tests/contracts/suites.py
  - tests/fixtures/wire/evon
failure_modes:
  - "abort() does not close the HTTP/WS stream (tokens keep billing/arriving)"
  - "multi-byte UTF-8 (Indic) split across chunks yields mojibake"
  - "stream error mid-response not surfaced; engine hangs"
  - "assistant history with truncation marker rejected or mis-roled"
  - "timeouts missing on connect/first token"
done_when:
  - "implements LlmPort/LlmStream strictly from docs/WIRE.md; passes LlmContract via recorded-fixture replay"
  - "incremental UTF-8 decoding test with Devanagari/Tamil split at every byte offset"
  - "abort idempotent; connect + first-token timeouts configurable and tested in virtual time"
  - "errors map to a typed LlmError the engine handles (canned apology, §14)"
evidence:
  - "make fast => pass"
evidence_observed: []
commit:
  msg: "T3.04: Evon LLM streaming adapter with abort"
  sha: null
strikes: 0
blocked_reason: ""
```

### T3.05 — Silero VAD adapter
```yaml task
id: T3.05
title: SileroVad adapter (ONNX, 20 ms hop) with EnergyVad fallback
phase: 3
status: todo
needs: [T0.04, T1.01]
reopens: []
files:
  - overlap/adapters/vad_silero.py
  - tests/unit/adapters/test_vad_silero.py
  - tests/bench/test_vad_bench.py
refs:
  - ARCHITECTURE.md#13
  - ARCHITECTURE.md#10
  - overlap/ports/vad.py
  - overlap/core/audio.py
  - tests/contracts/suites.py
failure_modes:
  - "model window larger than 20 ms adds > 1 frame of latency (violates INV-L1/L2)"
  - "model file missing/ONNX runtime unavailable crashes startup instead of falling back"
  - "onset flaps on single frames (no hysteresis) or never releases"
  - "per-frame inference cost > 2 ms p99 (INV-L12)"
  - "sample-rate mismatch not handled"
done_when:
  - "implements Vad port; passes VadContract with seeded synthetic audio (model optional in fast: tests use a stub session; real-model test marked slow and skipped if the model is absent)"
  - "windowing strategy documented in module docstring with measured added latency <= 1 frame"
  - "automatic fallback to EnergyVad with a telemetry event when model/onnxruntime unavailable"
  - "bench (marker bench): p99 <= 2 ms per frame on the dev machine"
evidence:
  - "make fast => pass"
  - "pytest -m bench tests/bench/test_vad_bench.py => p99 <= 2 ms"
evidence_observed: []
commit:
  msg: "T3.05: Silero VAD adapter with energy fallback"
  sha: null
strikes: 0
blocked_reason: ""
```

---

# Phase 4 — LiveKit/Pipecat integration, telemetry, A/B benchmark

### T4.01 — LiveKit transport
```yaml task
id: T4.01
title: LiveKitTransport — user track in, agent track out, playout-position feedback
phase: 4
status: todo
needs: [T2.04, T0.04]
reopens: [T0.04]
files:
  - overlap/integrations/livekit_transport.py
  - tests/unit/integrations/test_livekit_transport.py
  - tests/live/test_livekit_live.py
  - docs/CALIBRATION.md
refs:
  - ARCHITECTURE.md#5
  - ARCHITECTURE.md#8
  - ARCHITECTURE.md#15
  - overlap/ports/transport.py
  - tests/contracts/suites.py
  - tests/fakes/transport.py
failure_modes:
  - "send_frame blocks or allocates unboundedly under backpressure"
  - "jitter buffer hides playout position -> truncation includes unheard words"
  - "frame size/cadence mismatch (10 ms vs 20 ms) causes drift or underrun"
  - "flush_output does not clear the LiveKit audio source queue"
  - "user and agent tracks resampled inconsistently with core's canonical format"
done_when:
  - "implements AudioTransport; passes TransportContract against a faked livekit rtc layer (no network in fast)"
  - "frame re-chunking to 20 ms with tested no-drift over 10 min virtual time"
  - "`flush_output` clears the AudioSource queue; `output_queue_ms` reflects it; test via fake source"
  - "`playout_position_ms()` returns a measured estimate when available else None; calibration procedure + measured `assumed_downstream_ms` recorded in docs/CALIBRATION.md (human runs live test)"
  - "ports amended only if needed (reopens T0.04) with contract suite updated"
evidence:
  - "make fast => pass"
evidence_observed: []
commit:
  msg: "T4.01: LiveKit audio transport with flush and playout position"
  sha: null
strikes: 0
blocked_reason: ""
```

### T4.02 — Pipecat processor (drop-in)
```yaml task
id: T4.02
title: OverlapProcessor — one-line drop-in for Pipecat pipelines
phase: 4
status: todo
needs: [T2.04, T4.01]
reopens: []
files:
  - overlap/integrations/pipecat_processor.py
  - examples/pipecat_agent.py
  - examples/stock_pipecat_agent.py
  - tests/unit/integrations/test_pipecat_processor.py
  - README.md
refs:
  - ARCHITECTURE.md#3
  - ARCHITECTURE.md#13
  - overlap/runtime/engine.py
  - overlap/integrations/livekit_transport.py
  - overlap/config.py
failure_modes:
  - "processor swallows or reorders Pipecat control frames (EndFrame, CancelFrame, StartInterruptionFrame)"
  - "Pipecat's own built-in interruption fires in addition to OVERLAP's (double-stop)"
  - "frame back-pressure stalls the pipeline"
  - "engine tasks outlive pipeline shutdown"
  - "stock baseline example not behaviourally 'stock' (invalidates A/B)"
done_when:
  - "`OverlapProcessor(...)` is inserted between transport input and STT/TTS and takes Tuning/config only; documented in README with a <= 10-line integration snippet"
  - "Pipecat's default interruption strategy is disabled when OVERLAP is active (tested with fake Pipecat frames)"
  - "control frames pass through in order; shutdown calls engine.stop() and assert_quiescent"
  - "examples/stock_pipecat_agent.py is the A/B baseline (same STT/LLM/TTS, stop-on-any-VAD)"
  - "all tested with Pipecat frame fakes; no network in fast"
evidence:
  - "make fast => pass"
  - "make check => exit 0"
evidence_observed: []
commit:
  msg: "T4.02: Pipecat OverlapProcessor drop-in and stock baseline example"
  sha: null
strikes: 0
blocked_reason: ""
```

### T4.03 — Telemetry collector & metrics
```yaml task
id: T4.03
title: Telemetry — JSONL sink, metric derivation (t_duck/t_verdict/t_stop/FIR/bc timing)
phase: 4
status: todo
needs: [T2.04]
reopens: []
files:
  - overlap/core/metrics.py
  - overlap/telemetry/collector.py
  - overlap/telemetry/jsonl.py
  - tests/unit/core/test_metrics.py
  - tests/unit/telemetry/test_collector.py
refs:
  - ARCHITECTURE.md#12
  - ARCHITECTURE.md#10
  - overlap/ports/telemetry.py
  - overlap/core/types.py
failure_modes:
  - "sink blocks or raises on the fast path (disk I/O inline)"
  - "metric definitions drift from §12 (e.g. t_stop measured from verdict)"
  - "percentile math wrong on small samples / empty sets"
  - "transcript text/PII written to disk when redaction is on"
  - "events from stale epochs counted in metrics"
done_when:
  - "JSONL writer is non-blocking (bounded queue, background flush, drop-oldest + counter)"
  - "pure `metrics.py`: percentile, rate, Wilson CI; `derive_metrics(events)` -> t_duck, t_verdict, t_stop, FIR, missed rate, bc_timing, bc_rate, truncation_error, echo_false_duck exactly as ARCHITECTURE §12"
  - "golden test: event log from a T2.04 scenario produces expected metric values"
  - "redaction option replaces text with hashes/length; tested"
  - "CLI `python -m overlap.telemetry.report <jsonl>` prints metrics table vs targets with PASS/FAIL"
evidence:
  - "make fast => pass"
  - "report CLI on a scenario log => table with all §12 metrics"
evidence_observed: []
commit:
  msg: "T4.03: telemetry sink, metric derivation, report CLI"
  sha: null
strikes: 0
blocked_reason: ""
```

### T4.04 — Eval corpus, replay runner, threshold sweep
```yaml task
id: T4.04
title: Labelled overlap corpus + offline replay evaluator + Tuning sweep
phase: 4
status: todo
needs: [T4.03, T1.04]
reopens: [T1.04]
files:
  - overlap/bench/corpus.py
  - overlap/bench/replay.py
  - overlap/bench/sweep.py
  - bench/corpus/*.jsonl
  - tests/unit/bench/test_replay.py
  - tests/unit/bench/test_sweep.py
  - reports/classifier_sweep.md
  - overlap/core/classifier.py
refs:
  - ARCHITECTURE.md#7
  - ARCHITECTURE.md#12
  - overlap/core/classifier.py
  - overlap/core/metrics.py
  - overlap/telemetry/collector.py
failure_modes:
  - "corpus imbalance (mostly backchannels) hides missed-interruption rate"
  - "tuning overfits to the corpus (no held-out split)"
  - "sweep changes Tuning defaults silently"
  - "corpus/labels leak real PII"
  - "evaluator measures a different pipeline than the live engine"
done_when:
  - "corpus format: {id, lang, timeline (partials with t_ms, vad on/off, loudness), label} with >= 200 cases, >= 8 languages, balanced across BACKCHANNEL/INTERRUPT/SIDE_TALK, fixed train/held-out split by hash"
  - "`replay` runs the real classifier+FSM path (no copy of logic) over the corpus and reports FIR, missed-interruption, confusion matrix, per-language rows"
  - "`sweep` grid-searches Tuning thresholds on the train split, reports held-out metrics in reports/classifier_sweep.md; any recommended default change is applied to Tuning only here (reopens T1.04) and tests that import Tuning remain green"
  - "held-out FIR <= 5 % and missed <= 5 % (else recorded as a finding in the report; do not fudge labels)"
  - "deterministic: same inputs -> identical report"
evidence:
  - "make fast => pass"
  - "python -m overlap.bench.replay => prints metrics and PASS/FAIL against targets"
evidence_observed: []
commit:
  msg: "T4.04: labelled corpus, replay evaluator, threshold sweep"
  sha: null
strikes: 0
blocked_reason: ""
```

### T4.05 — Blind A/B harness, final demo packaging
```yaml task
id: T4.05
title: Blind A/B naturalness harness (stock Pipecat vs OVERLAP) + demo packaging
phase: 4
status: todo
needs: [T4.02, T4.03, T4.04, T3.02, T3.03, T3.04, T3.05]
reopens: []
files:
  - overlap/bench/ab.py
  - overlap/bench/scripted_caller.py
  - overlap/bench/rating_cli.py
  - bench/ab_scenarios/*.yaml
  - tests/unit/bench/test_ab.py
  - tests/unit/bench/test_scripted_caller.py
  - docs/DEMO.md
  - Dockerfile
  - Makefile
refs:
  - ARCHITECTURE.md#12
  - ARCHITECTURE.md#13
  - overlap/integrations/pipecat_processor.py
  - examples/stock_pipecat_agent.py
  - examples/pipecat_agent.py
  - overlap/telemetry/collector.py
  - overlap/core/metrics.py
failure_modes:
  - "raters can infer which agent is OVERLAP (filenames, metadata, latency, voice differences)"
  - "A/B arms differ in prompt/voice/LLM => confound"
  - "randomization not seeded/recorded => irreproducible"
  - "scripted caller timing drifts so overlaps don't land in the same place for both arms"
  - "stats misuse (no CI, no paired test)"
  - "demo depends on unreleased secrets or fragile live setup"
done_when:
  - "scenario YAMLs (>= 8) script caller behaviour: say 'haan' mid-reply, real interruption, long explanation with pauses, side-talk, code-switch Hinglish/Tamil-English, user says 'ok' repeatedly"
  - "scripted_caller drives both arms with identical seeded audio/timing (hermetic test with fakes proves identical stimulus timelines)"
  - "ab.py: seeded randomized A/B assignment, anonymized artifact names, assignment key stored separately; outputs call recordings + telemetry per arm"
  - "rating_cli: collects naturalness 1-5, 'felt interrupted', 'felt ignored', 'thought call dropped', forced preference; blind until all ratings submitted"
  - "analysis: preference with Wilson CI, paired sign test, per-metric deltas (false-interruption rate stock vs OVERLAP, silence-gap) -> reports/ab/summary.md"
  - "`make demo` / Dockerfile run the OVERLAP agent from env vars; docs/DEMO.md has headphone/AEC setup, 3-minute demo script, fallback plan"
  - "`make check` green; `make fast` < 15 s"
evidence:
  - "make fast => pass"
  - "make check => exit 0"
  - "python -m overlap.bench.ab --dry-run => prints blinded assignment table + stimulus-identity check OK"
evidence_observed: []
commit:
  msg: "T4.05: blind A/B harness, scripted caller, demo packaging"
  sha: null
strikes: 0
blocked_reason: ""
```
