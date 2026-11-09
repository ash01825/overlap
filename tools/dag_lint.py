#!/usr/bin/env python3
"""Task DAG linter and CLI for OVERLAP.

Implements validate, next, show, set, and verify-commit commands.
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

TASK_BLOCK_RE = re.compile(r"```yaml task\n(.*?)\n```", re.DOTALL)
TASK_ID_RE = re.compile(r"^T[0-4]\.\d{2}$")
COMMIT_SUBJECT_RE = re.compile(r"^(T[0-4]\.\d{2}):")

REQUIRED_FIELDS = [
    "id",
    "title",
    "phase",
    "status",
    "needs",
    "reopens",
    "files",
    "refs",
    "failure_modes",
    "done_when",
    "evidence",
    "evidence_observed",
    "commit",
    "strikes",
    "blocked_reason",
]
LEGAL_STATUSES = {"todo", "in_progress", "done", "blocked"}


@dataclass(slots=True, frozen=True)
class TaskBlock:
    """Represents a parsed task block from TASKS.md."""

    id: str
    data: dict[str, Any]
    raw_text: str
    fenced_text: str
    raw_start: int
    raw_end: int


def parse_task_blocks(content: str) -> list[TaskBlock]:
    """Extract and parse all task blocks from a tasks markdown file."""
    blocks: list[TaskBlock] = []
    for match in TASK_BLOCK_RE.finditer(content):
        raw_text = match.group(1)
        fenced_text = match.group(0)
        raw_start = match.start(1)
        raw_end = match.end(1)

        try:
            parsed = yaml.safe_load(raw_text)
        except Exception as exc:
            parsed = {"_parse_error": str(exc)}

        if not isinstance(parsed, dict):
            parsed = {"_parse_error": "YAML block is not a mapping"}

        task_id = str(parsed.get("id", ""))
        blocks.append(
            TaskBlock(
                id=task_id,
                data=parsed,
                raw_text=raw_text,
                fenced_text=fenced_text,
                raw_start=raw_start,
                raw_end=raw_end,
            )
        )
    return blocks


def build_transitive_needs(tasks: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Compute transitive ancestor IDs for each task."""
    task_map = {t["id"]: t for t in tasks if "id" in t}
    ancestors: dict[str, set[str]] = {}

    for t in tasks:
        tid = t.get("id")
        if not tid or not isinstance(tid, str):
            continue
        visited: set[str] = set()
        frontier = list(t.get("needs", []))
        while frontier:
            cur = frontier.pop()
            if cur in visited:
                continue
            visited.add(cur)
            parent_task = task_map.get(cur)
            if parent_task and isinstance(parent_task.get("needs"), list):
                frontier.extend(parent_task["needs"])
        ancestors[tid] = visited

    return ancestors


def find_cycles(tasks: list[dict[str, Any]]) -> list[list[str]]:
    """Detect cycles in task needs graph using DFS."""
    task_map = {t["id"]: t for t in tasks if "id" in t}
    visited: dict[str, int] = {}  # 0 = unvisited, 1 = visiting, 2 = done
    cycles: list[list[str]] = []

    def dfs(node: str, path: list[str]) -> None:
        visited[node] = 1
        path.append(node)

        t = task_map.get(node)
        if t and isinstance(t.get("needs"), list):
            for neighbor in t["needs"]:
                if not isinstance(neighbor, str) or neighbor not in task_map:
                    continue
                if visited.get(neighbor) == 1:
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
                elif visited.get(neighbor, 0) == 0:
                    dfs(neighbor, path)

        path.pop()
        visited[node] = 2

    for tid in task_map:
        if visited.get(tid, 0) == 0:
            dfs(tid, [])

    return cycles


def patterns_overlap(p1: str, p2: str) -> bool:
    """Check if two file paths or glob patterns overlap."""
    if p1 == p2:
        return True
    if fnmatch.fnmatch(p1, p2) or fnmatch.fnmatch(p2, p1):
        return True
    # If one is a directory prefix
    if p1.rstrip("/") + "/" in p2 or p2.rstrip("/") + "/" in p1:
        return True
    return False


def validate(tasks_path: Path) -> int:
    """Validate tasks against the schema and DAG invariants."""
    if not tasks_path.exists():
        sys.stderr.write(f"Error: Tasks file not found at {tasks_path}\n")
        return 1

    content = tasks_path.read_text(encoding="utf-8")
    blocks = parse_task_blocks(content)
    base_dir = tasks_path.resolve().parent

    errors: list[str] = []

    if not blocks:
        errors.append("No task blocks found in tasks file.")

    task_ids: list[str] = []
    seen_ids: set[str] = set()

    for idx, b in enumerate(blocks):
        data = b.data
        if "_parse_error" in data:
            errors.append(f"Block #{idx + 1}: YAML parse error: {data['_parse_error']}")
            continue

        tid = data.get("id")
        if not tid or not isinstance(tid, str):
            errors.append(f"Block #{idx + 1}: missing or invalid 'id' field.")
            continue

        if not TASK_ID_RE.match(tid):
            errors.append(
                f"Task {tid}: ID format invalid; must match pattern T<phase>.<nn> (e.g. T0.01)"
            )

        if tid in seen_ids:
            errors.append(f"Duplicate task ID: {tid}")
        seen_ids.add(tid)
        task_ids.append(tid)

        # Check required fields
        for rf in REQUIRED_FIELDS:
            if rf not in data:
                errors.append(f"Task {tid}: missing required field '{rf}'")

        # Field types and constraints
        phase = data.get("phase")
        if not isinstance(phase, int) or phase not in range(5):
            errors.append(f"Task {tid}: 'phase' must be integer in 0..4, got {phase!r}")
        elif tid and TASK_ID_RE.match(tid) and int(tid[1]) != phase:
            errors.append(f"Task {tid}: phase mismatch (id indicates {tid[1]}, phase is {phase})")

        status = data.get("status")
        if status not in LEGAL_STATUSES:
            errors.append(
                f"Task {tid}: illegal status {status!r}; must be one of {sorted(LEGAL_STATUSES)}"
            )

        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            errors.append(f"Task {tid}: 'title' must be non-empty string")

        for list_field in [
            "needs",
            "reopens",
            "files",
            "refs",
            "failure_modes",
            "done_when",
            "evidence",
        ]:
            val = data.get(list_field)
            if not isinstance(val, list):
                errors.append(f"Task {tid}: '{list_field}' must be a list")
            elif (
                list_field not in ("needs", "reopens") and len(val) == 0
            ):  # failure_modes, done_when, etc. must not be empty
                errors.append(f"Task {tid}: '{list_field}' must not be empty")
            elif isinstance(val, list):
                for item in val:
                    if not isinstance(item, str) or not item.strip():
                        errors.append(f"Task {tid}: '{list_field}' items must be non-empty strings")

        strikes = data.get("strikes")
        if not isinstance(strikes, int) or strikes < 0:
            errors.append(f"Task {tid}: 'strikes' must be integer >= 0")

        blocked_reason = data.get("blocked_reason")
        if not isinstance(blocked_reason, str):
            errors.append(f"Task {tid}: 'blocked_reason' must be a string")

        commit = data.get("commit")
        if not isinstance(commit, dict) or "msg" not in commit:
            errors.append(f"Task {tid}: 'commit' must be a mapping with 'msg' and 'sha'")

        # Condition checks
        if status == "done":
            ev_obs = data.get("evidence_observed")
            if not isinstance(ev_obs, list) or len(ev_obs) == 0:
                errors.append(
                    f"Task {tid}: done task requires evidence_observed (non-empty list of strings)"
                )
            if not isinstance(commit, dict) or not commit.get("sha"):
                errors.append(f"Task {tid}: done task requires commit.sha (non-empty string)")

        if status == "blocked":
            if not isinstance(blocked_reason, str) or not blocked_reason.strip():
                errors.append(f"Task {tid}: blocked task requires non-empty blocked_reason")

    valid_tasks = [b.data for b in blocks if "_parse_error" not in b.data and "id" in b.data]
    task_map = {t["id"]: t for t in valid_tasks}

    # Needs and reopens resolve
    for t in valid_tasks:
        tid = t["id"]
        for need in t.get("needs", []):
            if need not in task_map:
                errors.append(f"Task {tid}: needs unknown task '{need}'")
        for reopen in t.get("reopens", []):
            if reopen not in task_map:
                errors.append(f"Task {tid}: reopens unknown task '{reopen}'")

    # Cycle check
    cycles = find_cycles(valid_tasks)
    for c in cycles:
        errors.append(f"Cycle detected: {' -> '.join(c)}")

    # Transitive needs & reopens subset check
    ancestors = build_transitive_needs(valid_tasks)
    for t in valid_tasks:
        tid = t["id"]
        t_ancestors = ancestors.get(tid, set())
        for reopen in t.get("reopens", []):
            if reopen in task_map and reopen not in t_ancestors:
                errors.append(
                    f"Task {tid}: reopens '{reopen}', which is not in transitive needs of {tid}"
                )

    # Check refs: every refs path exists (once created)
    for t in valid_tasks:
        tid = t["id"]
        for ref in t.get("refs", []):
            path_part = ref.split("#")[0]
            disk_path = base_dir / path_part
            if disk_path.exists() or Path(path_part).exists():
                continue

            # Check if any task declares creating this path
            matching_creator: dict[str, Any] | None = None
            for other in valid_tasks:
                for f_pat in other.get("files", []):
                    if (
                        f_pat == path_part
                        or fnmatch.fnmatch(path_part, f_pat)
                        or f_pat.startswith(path_part.rstrip("/") + "/")
                        or fnmatch.fnmatch(f_pat, path_part.rstrip("/") + "/*")
                    ):
                        matching_creator = other
                        break
                if matching_creator:
                    break

            if matching_creator is None:
                errors.append(
                    f"Task {tid}: ref does not exist on disk "
                    f"and is not declared in any task files: '{ref}'"
                )
            elif matching_creator.get("status") == "done":
                errors.append(
                    f"Task {tid}: ref '{ref}' is created by completed task "
                    f"{matching_creator['id']}, but does not exist on disk"
                )

    # Check no files overlap between unordered tasks
    for i in range(len(valid_tasks)):
        for j in range(i + 1, len(valid_tasks)):
            t1 = valid_tasks[i]
            t2 = valid_tasks[j]
            id1 = t1["id"]
            id2 = t2["id"]

            has_order = id1 in ancestors.get(id2, set()) or id2 in ancestors.get(id1, set())
            if has_order:
                continue

            for p1 in t1.get("files", []):
                for p2 in t2.get("files", []):
                    if patterns_overlap(p1, p2):
                        errors.append(
                            f"Tasks {id1} and {id2} have no ordering path between them, "
                            f"but claim overlapping files: '{p1}' and '{p2}'"
                        )

    if errors:
        sys.stderr.write(f"Validation failed with {len(errors)} error(s):\n")
        for err in errors:
            sys.stderr.write(f"  - {err}\n")
        return 1

    print(f"OK, {len(valid_tasks)} tasks")
    return 0


def next_task(tasks_path: Path) -> int:
    """Print the lowest-ID ready task block and its refs."""
    if not tasks_path.exists():
        sys.stderr.write(f"Error: Tasks file not found at {tasks_path}\n")
        return 1

    content = tasks_path.read_text(encoding="utf-8")
    blocks = parse_task_blocks(content)

    # Check if any task is blocked
    for b in blocks:
        if b.data.get("status") == "blocked":
            reason = b.data.get("blocked_reason", "")
            sys.stderr.write(f"Refusing to proceed: task {b.id} is blocked (reason: {reason!r})\n")
            return 1

    task_map = {b.id: b for b in blocks if "_parse_error" not in b.data and b.id}

    # Find ready tasks (status == todo and all needs are done)
    ready_blocks: list[TaskBlock] = []
    for b in blocks:
        if b.data.get("status") != "todo":
            continue
        needs = b.data.get("needs", [])
        if all(need in task_map and task_map[need].data.get("status") == "done" for need in needs):
            ready_blocks.append(b)

    if not ready_blocks:
        print("No tasks are ready.")
        return 3

    # Lowest ID first
    ready_blocks.sort(key=lambda x: x.id)
    selected = ready_blocks[0]

    # Print block
    print(selected.fenced_text)
    print("\nRefs:")
    refs = selected.data.get("refs", [])
    for r in refs:
        print(f"  - {r}")

    return 0


def show_task(task_id: str, tasks_path: Path) -> int:
    """Print the fenced block of the specified task ID."""
    if not tasks_path.exists():
        sys.stderr.write(f"Error: Tasks file not found at {tasks_path}\n")
        return 1

    content = tasks_path.read_text(encoding="utf-8")
    blocks = parse_task_blocks(content)

    for b in blocks:
        if b.id == task_id:
            print(b.fenced_text)
            return 0

    sys.stderr.write(f"Error: Task {task_id} not found in {tasks_path}\n")
    return 1


def set_field(task_id: str, field: str, value: str, tasks_path: Path) -> int:
    """Perform targeted in-place edit on a single task's field."""
    if not tasks_path.exists():
        sys.stderr.write(f"Error: Tasks file not found at {tasks_path}\n")
        return 1

    content = tasks_path.read_text(encoding="utf-8")
    blocks = parse_task_blocks(content)

    target_block: TaskBlock | None = None
    for b in blocks:
        if b.id == task_id:
            target_block = b
            break

    if target_block is None:
        sys.stderr.write(f"Error: Task {task_id} not found in {tasks_path}\n")
        return 1

    block_text = target_block.raw_text
    lines = block_text.splitlines(keepends=True)

    new_lines: list[str] = []
    field_modified = False

    if "." in field:
        # e.g. commit.sha
        parent, child = field.split(".", 1)
        in_parent = False
        for line in lines:
            if re.match(rf"^{parent}:\s*$", line):
                in_parent = True
                new_lines.append(line)
                continue
            if in_parent:
                if re.match(r"^\S", line):
                    in_parent = False
                elif re.match(rf"^\s+{child}:", line):
                    indent = line[: len(line) - len(line.lstrip())]
                    if value.lower() in ("null", "none"):
                        new_lines.append(f"{indent}{child}: null\n")
                    else:
                        new_lines.append(f'{indent}{child}: "{value}"\n')
                    field_modified = True
                    continue
            new_lines.append(line)
    elif field == "evidence_observed":
        # Can be evidence_observed: [] or multi-line list
        in_field = False
        for line in lines:
            if re.match(r"^evidence_observed:", line):
                in_field = True
                if not value or value == "[]":
                    new_lines.append("evidence_observed: []\n")
                elif value.startswith("[") and value.endswith("]"):
                    try:
                        items = json.loads(value)
                    except Exception:
                        items = [value]
                    if not items:
                        new_lines.append("evidence_observed: []\n")
                    else:
                        new_lines.append("evidence_observed:\n")
                        for item in items:
                            new_lines.append(f'  - "{item}"\n')
                else:
                    new_lines.append("evidence_observed:\n")
                    new_lines.append(f'  - "{value}"\n')
                field_modified = True
                continue
            if in_field:
                if re.match(r"^\s+-\s", line) or re.match(r"^\s*$", line):
                    continue  # skip old list items
                else:
                    in_field = False
            new_lines.append(line)
    else:
        # Normal top-level field (status, strikes, blocked_reason, etc.)
        for line in lines:
            if re.match(rf"^{field}:", line):
                if field == "strikes":
                    new_lines.append(f"strikes: {int(value)}\n")
                elif field == "blocked_reason":
                    if not value or value == '""':
                        new_lines.append('blocked_reason: ""\n')
                    else:
                        escaped = value.replace('"', '\\"')
                        new_lines.append(f'blocked_reason: "{escaped}"\n')
                elif field == "status":
                    new_lines.append(f"status: {value}\n")
                else:
                    new_lines.append(f"{field}: {value}\n")
                field_modified = True
            else:
                new_lines.append(line)

    if not field_modified:
        sys.stderr.write(f"Error: Field '{field}' not found or modified in task {task_id}\n")
        return 1

    new_block_text = "".join(new_lines)
    if not block_text.endswith("\n") and new_block_text.endswith("\n"):
        new_block_text = new_block_text.rstrip("\n")

    # Reconstruct the file text precisely within the raw text boundaries
    new_content = (
        content[: target_block.raw_start] + new_block_text + content[target_block.raw_end :]
    )

    tasks_path.write_text(new_content, encoding="utf-8")
    return 0


def verify_commit(rev: str, tasks_path: Path, cwd: Path | None = None) -> int:
    """Check that changed files in a commit are within allowed scope for its task."""
    if not tasks_path.exists():
        sys.stderr.write(f"Error: Tasks file not found at {tasks_path}\n")
        return 1

    # Read commit subject
    try:
        proc_subject = subprocess.run(
            ["git", "log", "-1", "--format=%s", rev],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        subject = proc_subject.stdout.strip()
    except Exception as exc:
        sys.stderr.write(f"Error getting commit subject for {rev}: {exc}\n")
        return 1

    match = COMMIT_SUBJECT_RE.match(subject)
    if not match:
        sys.stderr.write(
            "Error: Commit subject does not match task ID format "
            f"'T<phase>.<nn>: <summary>': {subject!r}\n"
        )
        return 1

    task_id = match.group(1)
    content = tasks_path.read_text(encoding="utf-8")
    blocks = parse_task_blocks(content)
    task_map = {b.id: b.data for b in blocks if "_parse_error" not in b.data and b.id}

    if task_id not in task_map:
        sys.stderr.write(f"Error: Task {task_id} not found in {tasks_path}\n")
        return 1

    task = task_map[task_id]
    allowed_patterns: list[str] = list(task.get("files", []))

    # Add files of reopened tasks
    for reopened_id in task.get("reopens", []):
        if reopened_id in task_map:
            allowed_patterns.extend(task_map[reopened_id].get("files", []))

    # TASKS.md itself is always allowed
    allowed_patterns.append("TASKS.md")

    # Get changed files in commit
    try:
        proc_diff = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "--root", rev],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        changed_files = [line.strip() for line in proc_diff.stdout.splitlines() if line.strip()]
    except Exception as exc:
        sys.stderr.write(f"Error getting changed files for commit {rev}: {exc}\n")
        return 1

    forbidden_files: list[str] = []
    for cf in changed_files:
        is_allowed = False
        for pat in allowed_patterns:
            if (
                cf == pat
                or fnmatch.fnmatch(cf, pat)
                or pat.startswith(cf.rstrip("/") + "/")
                or cf.startswith(pat.rstrip("/") + "/")
            ):
                is_allowed = True
                break
        if not is_allowed:
            forbidden_files.append(cf)

    if forbidden_files:
        sys.stderr.write(
            f"verify-commit error: commit {rev} touches files outside "
            f"allowed scope for task {task_id}:\n"
        )
        for ff in forbidden_files:
            sys.stderr.write(f"  - {ff}\n")
        sys.stderr.write(f"Allowed patterns for {task_id}:\n")
        for ap in allowed_patterns:
            sys.stderr.write(f"  - {ap}\n")
        return 1

    print(
        f"Commit {rev} ({task_id}) verified: "
        f"all {len(changed_files)} changed files are within allowed scope."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for dag_lint."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help", "help"):
        print("Usage: dag_lint.py <command> [args...]")
        print("Commands:")
        print("  validate [TASKS_FILE]")
        print("  next [TASKS_FILE]")
        print("  show <ID> [TASKS_FILE]")
        print("  set <ID> <FIELD> <VALUE> [TASKS_FILE]")
        print("  verify-commit <REV> [TASKS_FILE]")
        return 0

    cmd = argv[0]

    if cmd == "validate":
        tasks_file = Path(argv[1]) if len(argv) > 1 else Path("TASKS.md")
        return validate(tasks_file)

    if cmd == "next":
        tasks_file = Path(argv[1]) if len(argv) > 1 else Path("TASKS.md")
        return next_task(tasks_file)

    if cmd == "show":
        if len(argv) < 2:
            sys.stderr.write("Usage: dag_lint.py show <ID> [TASKS_FILE]\n")
            return 1
        task_id = argv[1]
        tasks_file = Path(argv[2]) if len(argv) > 2 else Path("TASKS.md")
        return show_task(task_id, tasks_file)

    if cmd == "set":
        if len(argv) < 4:
            sys.stderr.write("Usage: dag_lint.py set <ID> <FIELD> <VALUE> [TASKS_FILE]\n")
            return 1
        task_id = argv[1]
        field = argv[2]
        value = argv[3]
        tasks_file = Path(argv[4]) if len(argv) > 4 else Path("TASKS.md")
        return set_field(task_id, field, value, tasks_file)

    if cmd == "verify-commit":
        if len(argv) < 2:
            sys.stderr.write("Usage: dag_lint.py verify-commit <REV> [TASKS_FILE]\n")
            return 1
        rev = argv[1]
        tasks_file = Path(argv[2]) if len(argv) > 2 else Path("TASKS.md")
        return verify_commit(rev, tasks_file)

    sys.stderr.write(f"Unknown command: {cmd}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
