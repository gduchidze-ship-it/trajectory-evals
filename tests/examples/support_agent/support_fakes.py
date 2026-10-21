"""A chat model scripted per user message, and the support agent it drives."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

from examples.support_agent.responder import factory
from trajectory_evals.connect.bridge import agent_card, build_app
from trajectory_evals.connect.client import A2AAgent
from trajectory_evals.connect.settings import settings
from trajectory_evals.core.agent_config import AgentConfig, load_agent
from trajectory_evals.core.settings import REPO_ROOT

SUPPORT_DIR: Path = REPO_ROOT / "examples" / "support_agent"


class RoutedChatModel(BaseChatModel):
    """Replies by the latest user message: its nth reply on the turn's nth call.

    Concurrent conversations get their own scripts, unlike a shared iterator.
    Records each prompt and accepts any tools.
    """

    routes: Mapping[str, Sequence[AIMessage]]
    prompts: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        """Name the fake for LangChain's tracing."""
        return "routed-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Record the prompt, then reply for its user message and call count."""
        self.prompts.append(messages)
        last_user = max(
            index
            for index, message in enumerate(messages)
            if isinstance(message, HumanMessage)
        )
        calls = sum(isinstance(message, AIMessage) for message in messages[last_user:])
        reply = self.routes[messages[last_user].text][calls]
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(
        self, tools: Sequence[Any], **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Ignore the tools; the script decides which the model calls."""
        return self


def tool_call(name: str, call_id: str, **arguments: Any) -> AIMessage:
    """Return a model reply that calls one tool."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": arguments, "id": call_id}]
    )


def support_config() -> AgentConfig:
    """Return the support agent's folder, loaded."""
    return load_agent(SUPPORT_DIR)


def support_agent(model: BaseChatModel) -> A2AAgent:
    """Return the support agent on model, bridged in process with no sockets."""
    config = support_config()
    assert config.bridge is not None
    responder = factory(config.bridge.options, config, model_factory=lambda _: model)
    card = agent_card(config, settings.bridge.url)
    return A2AAgent.in_process(build_app(responder, card))
