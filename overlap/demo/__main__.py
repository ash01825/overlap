"""CLI entrypoint for `python -m overlap.demo`."""

from __future__ import annotations

import sys

from overlap.demo.server import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
