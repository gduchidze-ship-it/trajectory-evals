"""Agent folders and a stop_when hook that the simulation tests share."""

from collections.abc import Mapping
from pathlib import Path

from trajectory_evals.core.agent_config import AgentConfig, load_agent
from trajectory_evals.core.types import ChatCompletionMessage

DESCRIPTION = "A support agent for an online bike shop."
GOODBYE = "Goodbye"
STOP_WHEN = "tests.simulation.simulation_test_fakes:said_goodbye"


def agent_folder(
    tmp_path: Path, tables: str = "", files: Mapping[str, str] | None = None
) -> AgentConfig:
    """Load a new agent folder: an [agent] table, then tables, and files beside it."""
    root = tmp_path / "agent"
    root.mkdir()
    agent = f'[agent]\nurl = "http://agent.test"\ndescription = "{DESCRIPTION}"\n'
    (root / "agent.toml").write_text(agent + tables, encoding="utf-8")
    for relative, text in (files or {}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return load_agent(root)


def said_goodbye(transcript: list[ChatCompletionMessage]) -> bool:
    """Return whether the agent's latest message says goodbye: a stop_when hook."""
    return GOODBYE in str(transcript[-1]["content"])


__all__ = ["DESCRIPTION", "GOODBYE", "STOP_WHEN", "agent_folder", "said_goodbye"]
