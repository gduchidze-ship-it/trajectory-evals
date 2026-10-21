"""Agent folders and chat-model factories the golden grader tests share."""

from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

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
    """Hands out chat models in order and records each model name asked for."""

    def __init__(self, *models: BaseChatModel) -> None:
        """Hand out models, one per call, in order."""
        self.names: list[str] = []
        self._models = deque(models)

    def __call__(self, name: str) -> BaseChatModel:
        """Record name and return the next model."""
        self.names.append(name)
        return self._models.popleft()
