"""Agent folders and chat-model factories the golden grader tests share."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from tests.fakes import scripted
from trajectory_evals.core.agent_config import AgentConfig, AgentSection

TOOL_CALL = {
    "id": "call-1",
    "type": "function",
    "function": {"name": "lookup_order", "arguments": '{"order_id": "A1"}'},
}


def agent_config(root: Path, **sections: Mapping[str, Any]) -> AgentConfig:
    """Return an agent folder at root whose agent.toml holds sections."""
    return AgentConfig(
        name=root.name,
        root=root,
        agent=AgentSection(url="http://localhost:9999", description="A test agent."),
        bridge=None,
        sections=sections,
    )


class ModelFactory:
    """Hands out a chat model per model name and records each name asked for."""

    def __init__(self, models: Mapping[str, BaseChatModel] | None = None) -> None:
        """Hand out models[name], or a model with no replies for any other name."""
        self.names: list[str] = []
        self._models = dict(models or {})

    def __call__(self, name: str) -> BaseChatModel:
        """Record name and return its model."""
        self.names.append(name)
        return self._models.get(name) or scripted()
