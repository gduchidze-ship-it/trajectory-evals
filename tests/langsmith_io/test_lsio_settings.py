"""Tests for the agent.toml [langsmith] table and the package's settings."""

from pathlib import Path
from typing import Any

import pytest

from tests.golden.golden_fakes import agent_config
from trajectory_evals.core.agent_config import AgentConfigError
from trajectory_evals.langsmith_io.settings import (
    LangSmithSection,
    langsmith_section,
    settings,
)


def test_an_absent_table_takes_the_configured_defaults(tmp_path: Path) -> None:
    """With no [langsmith], runs read from messages over the default window."""
    section = langsmith_section(agent_config(tmp_path))

    assert section == LangSmithSection(
        project=None,
        dataset_prefix=None,
        namespace=settings.defaults.namespace,
        inputs_key="messages",
        outputs_key="messages",
        since_days=settings.defaults.since_days,
    )


def test_a_table_sets_the_project_and_message_paths(tmp_path: Path) -> None:
    """A sequence-runner agent points the paths into its open state dict."""
    table = {
        "project": "dev-sales",
        "dataset_prefix": "sales",
        "inputs_key": "state.messages",
        "outputs_key": "state.history",
        "since_days": 2,
    }

    section = langsmith_section(agent_config(tmp_path, langsmith=table))

    assert section.model_dump(exclude={"namespace"}) == table


@pytest.mark.parametrize(
    "table",
    [{"projet": "dev-sales"}, {"inputs_key": "state..messages"}, {"since_days": 0}],
    ids=["typo", "empty-path-segment", "no-window"],
)
def test_a_bad_table_names_the_section(tmp_path: Path, table: dict[str, Any]) -> None:
    """An unknown key, a malformed path or an empty window is AgentConfigError."""
    with pytest.raises(AgentConfigError, match=r"\[langsmith\]"):
        langsmith_section(agent_config(tmp_path, langsmith=table))
