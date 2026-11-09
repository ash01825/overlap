#!/usr/bin/env python3
"""Task DAG linter and CLI stub for OVERLAP.

NOTE: This stub is bootstrapped in T0.01 and will be fully implemented in T0.02.
"""

from __future__ import annotations

import sys


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "validate":
        print("STUB: tools/dag_lint.py validate (will be replaced by T0.02)")
        return 0

    if cmd == "verify-commit":
        print("STUB: tools/dag_lint.py verify-commit (will be replaced by T0.02)")
        return 0

    if cmd == "next":
        print("STUB: Next task is T0.01 (or T0.02 once T0.01 is committed)")
        return 0

    if cmd in ("show", "set"):
        print(f"STUB: tools/dag_lint.py {cmd} called with args {sys.argv[2:]}")
        return 0

    print(f"STUB: tools/dag_lint.py command {cmd!r} recognized.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
