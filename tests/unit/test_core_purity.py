"""AST purity tests for overlap/core and overlap/ports.

Ensures no forbidden modules (asyncio, socket, websockets, httpx, time, random,
livekit, pipecat, onnxruntime), no async/await syntax, and no forbidden calls (e.g. sleep)
appear in pure layers.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

FORBIDDEN_ROOT_MODULES = {
    "aiohttp",
    "asyncio",
    "httpx",
    "livekit",
    "onnxruntime",
    "pipecat",
    "random",
    "requests",
    "socket",
    "time",
    "urllib.request",
    "websockets",
}


class PurityVisitor(ast.NodeVisitor):
    """AST visitor that collects violations of core/ports purity rules."""

    def __init__(self, filename: str = "<source>") -> None:
        self.filename = filename
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_mod = alias.name.split(".")[0]
            if root_mod in FORBIDDEN_ROOT_MODULES:
                self.violations.append(
                    f"{self.filename}:{node.lineno}: Forbidden import '{alias.name}'"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root_mod = node.module.split(".")[0]
            if root_mod in FORBIDDEN_ROOT_MODULES:
                self.violations.append(
                    f"{self.filename}:{node.lineno}: Forbidden import from '{node.module}'"
                )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.violations.append(
            f"{self.filename}:{node.lineno}: Forbidden 'async def {node.name}' in pure layer"
        )
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        self.violations.append(
            f"{self.filename}:{node.lineno}: Forbidden 'await' expression in pure layer"
        )
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.violations.append(
            f"{self.filename}:{node.lineno}: Forbidden 'async with' in pure layer"
        )
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.violations.append(
            f"{self.filename}:{node.lineno}: Forbidden 'async for' in pure layer"
        )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Catch bare sleep() or obj.sleep()
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name == "sleep":
            self.violations.append(
                f"{self.filename}:{node.lineno}: Forbidden call to sleep() in pure layer"
            )
        self.generic_visit(node)


def find_purity_violations(source: str, filename: str = "<source>") -> list[str]:
    """Parse python source and return all purity violations."""
    tree = ast.parse(source, filename=filename)
    visitor = PurityVisitor(filename=filename)
    visitor.visit(tree)
    return visitor.violations


def test_core_and_ports_purity() -> None:
    """Scan all .py files in overlap/core and overlap/ports for AST purity."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    pure_dirs = [repo_root / "overlap" / "core", repo_root / "overlap" / "ports"]

    all_violations: list[str] = []
    py_files_scanned = 0

    for directory in pure_dirs:
        assert directory.exists(), f"Pure directory does not exist: {directory}"
        for py_file in directory.rglob("*.py"):
            py_files_scanned += 1
            source = py_file.read_text(encoding="utf-8")
            violations = find_purity_violations(
                source, filename=str(py_file.relative_to(repo_root))
            )
            all_violations.extend(violations)

    assert py_files_scanned >= 2, (
        f"Expected at least 2 .py files in pure dirs, scanned {py_files_scanned}"
    )
    assert not all_violations, "Found AST purity violations in pure layers:\n" + "\n".join(
        all_violations
    )


@pytest.mark.parametrize(
    "offending_source, expected_fragment",
    [
        ("import asyncio\n", "Forbidden import 'asyncio'"),
        ("from asyncio import Event\n", "Forbidden import from 'asyncio'"),
        ("import socket\n", "Forbidden import 'socket'"),
        ("import websockets\n", "Forbidden import 'websockets'"),
        ("import httpx\n", "Forbidden import 'httpx'"),
        ("import time\n", "Forbidden import 'time'"),
        ("from time import sleep\n", "Forbidden import from 'time'"),
        ("import random\n", "Forbidden import 'random'"),
        ("from random import randint\n", "Forbidden import from 'random'"),
        ("import livekit\n", "Forbidden import 'livekit'"),
        ("import pipecat\n", "Forbidden import 'pipecat'"),
        ("import onnxruntime\n", "Forbidden import 'onnxruntime'"),
        ("async def run_coroutine():\n    pass\n", "Forbidden 'async def run_coroutine'"),
        ("async def run():\n    await fetch()\n", "Forbidden 'await'"),
        ("def do_sleep():\n    sleep(1.0)\n", "Forbidden call to sleep()"),
    ],
)
def test_purity_negative_cases(offending_source: str, expected_fragment: str) -> None:
    """Assert each forbidden import, async construct, or sleep call is detected."""
    violations = find_purity_violations(offending_source)
    assert violations, f"Expected violation for code: {offending_source!r}"
    assert any(expected_fragment in v for v in violations), (
        f"Expected fragment {expected_fragment!r} in violations: {violations}"
    )
