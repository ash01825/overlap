"""Smoke tests for OVERLAP environment and security boundaries."""

from __future__ import annotations

import socket
import sys

import pytest


def test_socket_blocked_under_disable_socket() -> None:
    """Assert creating a network socket is blocked under --disable-socket."""
    with pytest.raises(Exception) as exc_info:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Under pytest-socket with --disable-socket, creating a socket raises SocketBlockedError.
    err_type = type(exc_info.value).__name__
    assert "SocketBlocked" in err_type or "blocked" in str(exc_info.value).lower(), (
        f"Expected socket creation to be blocked by pytest-socket, got {err_type}: {exc_info.value}"
    )


def test_environment_python_version() -> None:
    """Assert the active python version is 3.11."""
    assert sys.version_info[:2] == (3, 11), f"Expected Python 3.11, found {sys.version}"


def test_packages_importable() -> None:
    """Assert all core packages in the layering hierarchy are importable."""
    import overlap
    import overlap.adapters
    import overlap.core
    import overlap.integrations
    import overlap.ports
    import overlap.runtime

    assert overlap is not None
    assert overlap.core is not None
    assert overlap.ports is not None
    assert overlap.runtime is not None
    assert overlap.adapters is not None
    assert overlap.integrations is not None
