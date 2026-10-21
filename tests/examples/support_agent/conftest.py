"""The offline fixture the support agent's tests share."""

from collections.abc import Iterator
from typing import Any

import pytest
from langsmith import tracing_context

from trajectory_evals.core.runner import build_local_client


@pytest.fixture
def offline(network_attempts: list[Any]) -> Iterator[list[Any]]:
    """Refuse the network with tracing on, tracing to a client that never sends."""
    # The agent's chat model and openevals trace with @traceable, which would queue
    # runs on the default client and send them after the test.
    with tracing_context(client=build_local_client()):
        yield network_attempts
