"""CLI report generator for OVERLAP telemetry logs (`python -m overlap.telemetry.report <jsonl>`).

Prints all ARCHITECTURE §12 evaluation metrics compared against targets with PASS/FAIL.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from overlap.core.metrics import DerivedMetrics, MetricCheck, derive_metrics, evaluate_targets
from overlap.core.types import Tuning
from overlap.telemetry.jsonl import read_jsonl


def format_report_table(
    metrics_or_events: DerivedMetrics | Sequence[Any],
    tuning: Tuning | None = None,
) -> tuple[str, bool]:
    """Format a markdown/ASCII table of all §12 metrics vs targets and overall PASS/FAIL."""
    if isinstance(metrics_or_events, DerivedMetrics):
        metrics = metrics_or_events
    else:
        metrics = derive_metrics(metrics_or_events)

    checks: tuple[MetricCheck, ...] = evaluate_targets(metrics, tuning=tuning)
    all_passed = all(c.passed for c in checks)

    lines: list[str] = [
        "| Metric | Observed | Target | Status |",
        "|---|---|---|---|",
    ]
    for c in checks:
        status_str = "PASS" if c.passed else "FAIL"
        lines.append(f"| {c.name} | {c.observed_str} | {c.target_str} | {status_str} |")

    overall_str = "PASS" if all_passed else "FAIL"
    lines.append("")
    lines.append(f"OVERALL: {overall_str}")
    return ("\n".join(lines), all_passed)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m overlap.telemetry.report <jsonl>`."""
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("Usage: python -m overlap.telemetry.report <jsonl_path>")
        return 0 if args else 1

    path = Path(args[0])
    if not path.exists():
        sys.stderr.write(f"Error: JSONL file not found: {path}\n")
        return 1

    events = read_jsonl(path)
    table, passed = format_report_table(events)
    print(table)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
