#!/usr/bin/env python3
"""Run a command enforcing a strict wall-clock timeout budget.

Usage:
    python tools/timed.py <budget_seconds> <command> [args...]

Exit codes:
    0: Command finished successfully within the budget.
    124: Command exceeded the wall-clock budget.
    Other non-zero: Command failed with that exit code.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time


def run_timed(budget: float, cmd: list[str]) -> int:
    """Run command and enforce wall-clock budget in seconds."""
    start_time = time.monotonic()
    try:
        proc = subprocess.Popen(cmd)
        try:
            return_code = proc.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            elapsed = time.monotonic() - start_time
            sys.stderr.write(
                f"ERROR: Command timed out after {elapsed:.2f}s (budget: {budget:.1f}s): "
                f"{' '.join(cmd)}\n"
            )
            return 124
    except Exception as exc:
        sys.stderr.write(f"ERROR: Failed to execute command {' '.join(cmd)}: {exc}\n")
        return 1

    elapsed = time.monotonic() - start_time
    if elapsed > budget:
        sys.stderr.write(
            f"ERROR: Command completed in {elapsed:.2f}s, exceeding budget of {budget:.1f}s: "
            f"{' '.join(cmd)}\n"
        )
        return 124

    return return_code


def main() -> int:
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: python tools/timed.py <budget_seconds> <command> [args...]\n")
        return 2

    try:
        budget = float(sys.argv[1])
        if budget <= 0:
            raise ValueError("Budget must be positive")
    except ValueError as e:
        sys.stderr.write(f"ERROR: Invalid budget {sys.argv[1]!r}: {e}\n")
        return 2

    if len(sys.argv) == 3 and " " in sys.argv[2]:
        cmd = shlex.split(sys.argv[2])
    else:
        cmd = sys.argv[2:]

    return run_timed(budget, cmd)


if __name__ == "__main__":
    sys.exit(main())
