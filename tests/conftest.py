"""Skip jailbreak without its detector, trace nothing, and refuse network on request."""

import importlib.util
import socket
from collections.abc import Iterator
from typing import Any, NoReturn, cast

import pytest
from langsmith.utils import get_env_var

_DETECTOR = "src.toolsets.jailbreak_detection"

try:
    _HAS_DETECTOR = importlib.util.find_spec(_DETECTOR) is not None
except ModuleNotFoundError:
    # find_spec imports the parent packages, so a missing src raises instead.
    _HAS_DETECTOR = False

if not _HAS_DETECTOR:
    collect_ignore_glob = ["suites/jailbreak/*", "examples/jailbreak/*"]


def _clear_langsmith_env_cache() -> None:
    """Drop langsmith's cached env reads; mypy sees only get_env_var's overloads."""
    cast(Any, get_env_var).cache_clear()


@pytest.fixture(autouse=True)
def tracing_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Turn LangSmith tracing off unless a test turns it on, so nothing is queued."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    # langsmith reads TRACING_V2 before TRACING, so either would override false.
    monkeypatch.delenv("LANGSMITH_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    _clear_langsmith_env_cache()
    yield
    _clear_langsmith_env_cache()


@pytest.fixture
def network_attempts(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """Refuse every connect and DNS lookup with tracing on, and list the attempts."""
    attempts: list[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> NoReturn:
        """Record a network attempt and refuse it."""
        attempts.append(args)
        raise OSError(f"network disabled in test: {args}")

    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-fake-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    _clear_langsmith_env_cache()
    yield attempts
    _clear_langsmith_env_cache()
