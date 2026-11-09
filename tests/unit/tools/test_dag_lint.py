"""Unit tests for tools/dag_lint.py.

Covers all failure modes and done_when requirements for T0.02.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "tasks"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REAL_TASKS_MD = REPO_ROOT / "TASKS.md"


def _load_dag_lint() -> ModuleType:
    module_path = REPO_ROOT / "tools" / "dag_lint.py"
    spec = importlib.util.spec_from_file_location("dag_lint", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load dag_lint from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dag_lint"] = mod
    spec.loader.exec_module(mod)
    return mod


dag_lint = _load_dag_lint()


def test_validate_real_tasks_md(capsys: pytest.CaptureFixture[str]) -> None:
    """Validate must pass on the real TASKS.md in this repo."""
    ret = dag_lint.main(["validate", str(REAL_TASKS_MD)])
    out, err = capsys.readouterr()
    assert ret == 0, f"Validation failed on real TASKS.md: {err}\n{out}"
    assert "OK, 25 tasks" in out


def test_validate_minimal_valid(capsys: pytest.CaptureFixture[str]) -> None:
    """Validate passes on a valid minimal DAG."""
    fixture = FIXTURES_DIR / "valid_minimal.md"
    ret = dag_lint.main(["validate", str(fixture)])
    out, _ = capsys.readouterr()
    assert ret == 0
    assert "OK, 2 tasks" in out


@pytest.mark.parametrize(
    ("fixture_name", "expected_err_substring"),
    [
        ("bad_cycle.md", "Cycle detected"),
        ("bad_unknown_need.md", "unknown task"),
        ("bad_unknown_reopen.md", "unknown task"),
        ("bad_reopen_not_transitive.md", "not in transitive needs"),
        ("bad_missing_field.md", "missing required field"),
        ("bad_status.md", "illegal status"),
        ("bad_phase_mismatch.md", "phase mismatch"),
        ("bad_duplicate_id.md", "Duplicate task ID"),
        ("bad_done_missing_sha.md", "done task requires commit.sha"),
        ("bad_done_missing_evidence.md", "done task requires evidence_observed"),
        ("bad_blocked_missing_reason.md", "blocked task requires non-empty blocked_reason"),
        ("bad_ref_nonexistent.md", "ref does not exist"),
        ("bad_ref_done_not_created.md", "created by completed task"),
        ("bad_unordered_overlap.md", "overlapping files"),
    ],
)
def test_validate_failure_modes(
    fixture_name: str,
    expected_err_substring: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validate catches every invalid DAG state with a clear error message."""
    fixture = FIXTURES_DIR / fixture_name
    ret = dag_lint.main(["validate", str(fixture)])
    out, err = capsys.readouterr()
    combined = (out + "\n" + err).lower()
    assert ret != 0, f"Expected non-zero exit for {fixture_name}"
    assert expected_err_substring.lower() in combined, (
        f"Expected {expected_err_substring!r} in output for {fixture_name}.\nOutput:\n{combined}"
    )


def test_next_returns_lowest_id_ready_task(capsys: pytest.CaptureFixture[str]) -> None:
    """next returns the lowest-ID ready task and prints its block + refs."""
    fixture = FIXTURES_DIR / "valid_minimal.md"
    ret = dag_lint.main(["next", str(fixture)])
    out, _ = capsys.readouterr()
    assert ret == 0
    assert "id: T0.02" in out
    assert "Refs:" in out
    assert "README.md" in out


def test_next_refuses_when_task_blocked(capsys: pytest.CaptureFixture[str]) -> None:
    """next refuses to proceed if any task is blocked."""
    fixture = FIXTURES_DIR / "next_blocked.md"
    ret = dag_lint.main(["next", str(fixture)])
    out, err = capsys.readouterr()
    combined = (out + "\n" + err).lower()
    assert ret != 0
    assert "blocked" in combined


def test_next_returns_exit_3_when_none_ready(capsys: pytest.CaptureFixture[str]) -> None:
    """next returns exit code 3 when no task is ready."""
    fixture = FIXTURES_DIR / "next_none_ready.md"
    ret = dag_lint.main(["next", str(fixture)])
    _out, _err = capsys.readouterr()
    assert ret == 3


def test_show_prints_only_specified_block(capsys: pytest.CaptureFixture[str]) -> None:
    """show <ID> prints only that task block."""
    fixture = FIXTURES_DIR / "valid_minimal.md"
    ret = dag_lint.main(["show", "T0.01", str(fixture)])
    out, _ = capsys.readouterr()
    assert ret == 0
    assert "id: T0.01" in out
    assert "First task" in out
    assert "T0.02" not in out


def test_show_unknown_id_fails(capsys: pytest.CaptureFixture[str]) -> None:
    """show fails if task ID does not exist."""
    fixture = FIXTURES_DIR / "valid_minimal.md"
    ret = dag_lint.main(["show", "T9.99", str(fixture)])
    _out, err = capsys.readouterr()
    assert ret != 0
    assert "not found" in err.lower()


def test_set_performs_targeted_inplace_edits_roundtrip(tmp_path: Path) -> None:
    """set performs targeted in-place edits and leaves all other bytes unchanged."""
    fixture = FIXTURES_DIR / "valid_minimal.md"
    original_bytes = fixture.read_bytes()

    test_file = tmp_path / "TASKS.md"
    test_file.write_bytes(original_bytes)

    # 1. Update status
    ret = dag_lint.main(["set", "T0.02", "status", "in_progress", str(test_file)])
    assert ret == 0
    content = test_file.read_text(encoding="utf-8")
    assert "status: in_progress" in content

    # 2. Update strikes
    ret = dag_lint.main(["set", "T0.02", "strikes", "2", str(test_file)])
    assert ret == 0
    content = test_file.read_text(encoding="utf-8")
    assert "strikes: 2" in content

    # 3. Update blocked_reason
    ret = dag_lint.main(["set", "T0.02", "blocked_reason", "failed test", str(test_file)])
    assert ret == 0
    content = test_file.read_text(encoding="utf-8")
    assert 'blocked_reason: "failed test"' in content

    # 4. Update commit.sha
    ret = dag_lint.main(["set", "T0.02", "commit.sha", "deadbeef", str(test_file)])
    assert ret == 0
    content = test_file.read_text(encoding="utf-8")
    assert 'sha: "deadbeef"' in content

    # 5. Revert all edits back to original values
    assert dag_lint.main(["set", "T0.02", "status", "todo", str(test_file)]) == 0
    assert dag_lint.main(["set", "T0.02", "strikes", "0", str(test_file)]) == 0
    assert dag_lint.main(["set", "T0.02", "blocked_reason", "", str(test_file)]) == 0
    assert dag_lint.main(["set", "T0.02", "commit.sha", "null", str(test_file)]) == 0

    reverted_bytes = test_file.read_bytes()
    assert reverted_bytes == original_bytes, "Round-trip set did not preserve all bytes perfectly!"


def test_set_evidence_observed(tmp_path: Path) -> None:
    """set can update evidence_observed field."""
    fixture = FIXTURES_DIR / "valid_minimal.md"
    test_file = tmp_path / "TASKS.md"
    test_file.write_bytes(fixture.read_bytes())

    ret = dag_lint.main(
        ["set", "T0.02", "evidence_observed", "make fast => 10 passed in 0.1s", str(test_file)]
    )
    assert ret == 0
    content = test_file.read_text(encoding="utf-8")
    assert "make fast => 10 passed in 0.1s" in content
    assert "evidence_observed: []" not in content


def test_verify_commit_allowed_vs_forbidden(tmp_path: Path, monkeypatch: Any) -> None:
    """verify-commit checks changed files against allowed set using git diff-tree."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    # Initialize a test git repository
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=repo_dir, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    tasks_content = """# Tasks

```yaml task
id: T0.01
title: Task 1
phase: 0
status: done
needs: []
reopens: []
files:
  - allowed_a.txt
  - allowed_dir/*.py
refs:
  - README.md
failure_modes:
  - "test"
done_when:
  - "test"
evidence:
  - "test"
evidence_observed:
  - "test"
commit:
  msg: "T0.01: first commit"
  sha: "1234"
strikes: 0
blocked_reason: ""
```

```yaml task
id: T0.02
title: Task 2
phase: 0
status: todo
needs: [T0.01]
reopens: [T0.01]
files:
  - allowed_b.txt
refs:
  - README.md
failure_modes:
  - "test"
done_when:
  - "test"
evidence:
  - "test"
evidence_observed: []
commit:
  msg: "T0.02: second commit"
  sha: null
strikes: 0
blocked_reason: ""
```
"""
    tasks_file = repo_dir / "TASKS.md"
    tasks_file.write_text(tasks_content, encoding="utf-8")

    # Initial commit (touching TASKS.md and allowed_a.txt)
    (repo_dir / "allowed_a.txt").write_text("hello", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "T0.01: initial commit"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    # verify-commit on T0.01 initial commit should pass
    ret = dag_lint.verify_commit("HEAD", tasks_path=tasks_file, cwd=repo_dir)
    assert ret == 0

    # Second commit: T0.02 modifies allowed_b.txt and reopens allowed_a.txt
    (repo_dir / "allowed_b.txt").write_text("world", encoding="utf-8")
    (repo_dir / "allowed_a.txt").write_text("modified in reopened", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "T0.02: valid changes in allowed files"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    ret = dag_lint.verify_commit("HEAD", tasks_path=tasks_file, cwd=repo_dir)
    assert ret == 0

    # Third commit: T0.02 touches an unauthorized file
    (repo_dir / "unauthorized.txt").write_text("forbidden", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "T0.02: touching unauthorized file"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    ret = dag_lint.verify_commit("HEAD", tasks_path=tasks_file, cwd=repo_dir)
    assert ret != 0

    # Fourth commit: commit without proper task subject format
    (repo_dir / "allowed_b.txt").write_text("extra", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "WIP commit without task ID"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    ret = dag_lint.verify_commit("HEAD", tasks_path=tasks_file, cwd=repo_dir)
    assert ret != 0
