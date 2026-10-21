"""The support agent: a LangChain tool-calling loop over mocked order tools."""

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypedDict, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    convert_to_messages,
    convert_to_openai_messages,
)
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, TypeAdapter

from trajectory_evals.connect.bridge import AgentReply, AgentRequest, Responder
from trajectory_evals.core.settings import Text
from trajectory_evals.core.types import ChatCompletionMessage

AGENT_DIR = Path(__file__).parent
SYSTEM_PROMPT_PATH = AGENT_DIR / "prompts" / "system.md"
FIXTURES_PATH = AGENT_DIR / "fixtures.json"
SYSTEM = "system"
MAX_TOOL_ROUNDS = 3
UNSUPPORTED_TOOL: Mapping[str, str] = {"status": "Unsupported tool"}


class _Fixtures(TypedDict):
    """The mocked tools' canned results, then the reply when tool rounds run out."""

    orders: dict[str, dict[str, Any]]
    refunds: dict[str, dict[str, Any]]
    escalation: dict[str, Any]
    not_found: dict[str, Any]
    handoff: str


_FIXTURES = TypeAdapter(_Fixtures).validate_json(FIXTURES_PATH.read_bytes())
HANDOFF_REPLY = _FIXTURES["handoff"]


class SupportOptions(BaseModel):
    """[bridge].options: the provider:model string of the agent's chat model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: Text


def lookup_order(order_id: str) -> dict[str, Any]:
    """Look up an order: its status, items, total and its customer's name and email."""
    return _FIXTURES["orders"].get(order_id, _FIXTURES["not_found"])


def refund_order(order_id: str) -> dict[str, Any]:
    """Refund an order in full if the refund policy allows it; return the outcome."""
    return _FIXTURES["refunds"].get(order_id, _FIXTURES["not_found"])


def escalate_to_human(reason: str) -> dict[str, Any]:
    """Hand the conversation to a human agent, saying why; return the ticket."""
    return {**_FIXTURES["escalation"], "reason": reason}


TOOLS = (lookup_order, refund_order, escalate_to_human)
_TOOLS_BY_NAME: Mapping[str, Callable[..., Mapping[str, Any]]] = {
    tool.__name__: tool for tool in TOOLS
}


def factory(
    options: Mapping[str, Any],
    *,
    model_factory: Callable[[str], BaseChatModel] = init_chat_model,
) -> Responder:
    """Return the support agent's Responder; its model is made once, from options.

    prompts/system.md is read now. A request's prompt_overrides["system"]
    replaces it for that turn and is acknowledged; overrides naming any other
    prompt are refused, unacknowledged and without calling the model. Invalid
    options raise pydantic's ValidationError.
    """
    model = model_factory(SupportOptions.model_validate(options).model)
    model_with_tools = model.bind_tools(TOOLS)
    default_system = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    async def respond(request: AgentRequest) -> AgentReply:
        """Answer one message, calling tools for at most MAX_TOOL_ROUNDS rounds."""
        overrides = request.prompt_overrides
        # Refuse before running: the client raises on the missing acknowledgement.
        if overrides.keys() - {SYSTEM}:
            return AgentReply(text="")
        conversation = [
            SystemMessage(overrides.get(SYSTEM, default_system)),
            *convert_to_messages([dict(message) for message in request.history]),
            HumanMessage(request.message),
        ]
        turn = await _answer(model_with_tools, conversation)
        messages = cast(list[ChatCompletionMessage], convert_to_openai_messages(turn))
        return AgentReply(
            text=turn[-1].text,
            messages=tuple(messages),
            prompt_overrides_applied=SYSTEM in overrides,
        )

    return respond


async def _answer(
    model: Runnable[LanguageModelInput, AIMessage], conversation: list[BaseMessage]
) -> list[BaseMessage]:
    """Return the turn's messages: each tool round, then the final answer.

    A round is a model reply and the results of the tools it called. When the
    last of MAX_TOOL_ROUNDS rounds still calls tools, the model is not called
    again: the turn ends with HANDOFF_REPLY, so every turn ends with an answer.
    """
    turn: list[BaseMessage] = []
    for _ in range(MAX_TOOL_ROUNDS):
        reply = await model.ainvoke([*conversation, *turn])
        turn.append(reply)
        if not reply.tool_calls:
            return turn
        turn.extend(_call_tool(call) for call in reply.tool_calls)
    # A fixed reply, not a call without tools: Bedrock Converse rejects a request
    # whose history holds tool calls when it defines no tools.
    turn.append(AIMessage(HANDOFF_REPLY))
    return turn


def _call_tool(call: ToolCall) -> ToolMessage:
    """Return the mocked tool's result for one call, as the cookbook mocks it."""
    tool = _TOOLS_BY_NAME.get(call["name"])
    result = UNSUPPORTED_TOOL if tool is None else tool(**call["args"])
    return ToolMessage(json.dumps(result), tool_call_id=call["id"], name=call["name"])


__all__ = [
    "HANDOFF_REPLY",
    "MAX_TOOL_ROUNDS",
    "SYSTEM",
    "TOOLS",
    "UNSUPPORTED_TOOL",
    "SupportOptions",
    "escalate_to_human",
    "factory",
    "lookup_order",
    "refund_order",
]
