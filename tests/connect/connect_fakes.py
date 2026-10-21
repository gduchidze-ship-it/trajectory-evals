"""A recording responder and an agent folder that the connect tests share."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model

from trajectory_evals.connect.bridge import AgentReply, AgentRequest, Responder
from trajectory_evals.core.agent_config import AgentConfig, AgentSection
from trajectory_evals.core.types import ChatCompletionMessage, ToolMessage

LOOKUP_CALL_ID = "call-lookup"
LOOKUP_RESULT = "in stock"


def lookup_messages(text: str) -> tuple[ChatCompletionMessage, ...]:
    """Return the turn RecordingResponder gives: one lookup call, then the echo."""
    arguments = json.dumps({"query": text})
    function = {"name": "lookup", "arguments": arguments}
    call = {"id": LOOKUP_CALL_ID, "type": "function", "function": function}
    result: ToolMessage = {
        "role": "tool",
        "content": LOOKUP_RESULT,
        "tool_call_id": LOOKUP_CALL_ID,
    }
    return (
        {"role": "assistant", "content": "", "tool_calls": [call]},
        result,
        {"role": "assistant", "content": f"echo: {text}"},
    )


@dataclass
class RecordingResponder:
    """Echoes each message after one lookup tool call, and records each request."""

    apply_overrides: bool = False
    requests: list[AgentRequest] = field(default_factory=list)

    async def __call__(self, request: AgentRequest) -> AgentReply:
        """Record the request and reply with lookup_messages(request.message)."""
        self.requests.append(request)
        return AgentReply(
            text=f"echo: {request.message}",
            messages=lookup_messages(request.message),
            prompt_overrides_applied=self.apply_overrides,
        )


def recording_factory(options: Mapping[str, Any], _config: AgentConfig) -> Responder:
    """Return a RecordingResponder built from [bridge].options."""
    return RecordingResponder(**options)


def chat_model_factory(options: Mapping[str, Any], _config: AgentConfig) -> Responder:
    """Build options["api_key"] as a chat model, as a factory passing a bad option on.

    init_chat_model's error quotes the model string it could not resolve.
    """
    init_chat_model(str(options["api_key"]))
    return RecordingResponder()


def agent_config(name: str = "shop") -> AgentConfig:
    """Return a loaded-looking agent folder with no endpoint, for agent cards."""
    agent = AgentSection(description="A car shop's assistant.")
    return AgentConfig(
        name=name, root=Path(name), agent=agent, bridge=None, sections={}
    )
