"""Tests for the support agent's responder, served by the bridge in process."""

import json
import re
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from examples.support_agent.responder import (
    MAX_TOOL_ROUNDS,
    TOOLS,
    UNSUPPORTED_TOOL,
    lookup_order,
    refund_order,
)
from tests.examples.support_agent.support_fakes import (
    SUPPORT_DIR,
    RoutedChatModel,
    support_agent,
    tool_call,
)
from trajectory_evals.connect.settings import settings
from trajectory_evals.core.types import AgentError

REFUND = "Refund order A1001, email dana@example.com."
REFUNDED = "Done: $89.00 is on its way back to your card."


def _round(name: str, call_id: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one tool round on A1001 in OpenAI format: the call, then its result."""
    function = {"name": name, "arguments": json.dumps({"order_id": "A1001"})}
    call = {"type": "function", "id": call_id, "function": function}
    return [
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {
            "role": "tool",
            "name": name,
            "tool_call_id": call_id,
            "content": json.dumps(result),
        },
    ]


@pytest.mark.asyncio
async def test_a_refund_reports_its_tool_calls_and_results(
    offline: list[Any],
) -> None:
    """Each round's call and its mocked result come back in order, then the reply."""
    script = [
        tool_call("lookup_order", "c1", order_id="A1001"),
        tool_call("refund_order", "c2", order_id="A1001"),
        AIMessage(REFUNDED),
    ]
    model = RoutedChatModel(routes={REFUND: script})
    async with support_agent(model) as agent:
        turn = await agent.send(REFUND)

    assert turn.text == REFUNDED
    assert list(turn.messages) == [
        *_round("lookup_order", "c1", lookup_order("A1001")),
        *_round("refund_order", "c2", refund_order("A1001")),
        {"role": "assistant", "content": REFUNDED},
    ]
    system = (SUPPORT_DIR / "prompts" / "system.md").read_text(encoding="utf-8")
    assert model.prompts[0][0].content == system
    assert offline == []


@pytest.mark.asyncio
async def test_a_system_override_is_used_and_acknowledged() -> None:
    """The override replaces system.md, and the client sees it applied."""
    model = RoutedChatModel(routes={"Hi": [AIMessage("Hello.")]})
    async with support_agent(model) as agent:
        turn = await agent.send("Hi", prompt_overrides={"system": "Be terse."})

    assert turn.text == "Hello."
    assert model.prompts[0][0].content == "Be terse."


@pytest.mark.asyncio
async def test_an_override_of_another_prompt_is_refused() -> None:
    """An unknown prompt name is never acknowledged, and the model is not called."""
    model = RoutedChatModel(routes={})
    refused = re.escape(settings.texts.overrides_not_applied)
    async with support_agent(model) as agent:
        with pytest.raises(AgentError, match=refused):
            await agent.send("Hi", prompt_overrides={"tone": "Be rude."})

    assert model.prompts == []


@pytest.mark.asyncio
async def test_an_unknown_tool_gets_the_unsupported_result() -> None:
    """A call to a tool the agent lacks is answered, not raised, like the cookbook."""
    script = [tool_call("cancel_order", "c1", order_id="A1002"), AIMessage("Sorry.")]
    async with support_agent(RoutedChatModel(routes={"Cancel": script})) as agent:
        turn = await agent.send("Cancel")

    assert turn.messages[1]["content"] == json.dumps(UNSUPPORTED_TOOL)


@pytest.mark.asyncio
async def test_a_model_that_keeps_calling_tools_is_made_to_answer() -> None:
    """After MAX_TOOL_ROUNDS tool rounds, one more call gives the final answer."""
    rounds = [
        tool_call("lookup_order", f"c{number}", order_id="A1002")
        for number in range(MAX_TOOL_ROUNDS)
    ]
    model = RoutedChatModel(routes={"Where?": [*rounds, AIMessage("In transit.")]})
    async with support_agent(model) as agent:
        turn = await agent.send("Where?")

    assert len(model.prompts) == MAX_TOOL_ROUNDS + 1
    assert turn.messages[-1] == {"role": "assistant", "content": "In transit."}


@pytest.mark.asyncio
async def test_a_later_turn_sees_the_earlier_tool_results() -> None:
    """The bridge's transcript comes back to the model as LangChain messages."""
    model = RoutedChatModel(
        routes={
            "Where is A1002?": [
                tool_call("lookup_order", "c1", order_id="A1002"),
                AIMessage("It is in transit."),
            ],
            "Thanks": [AIMessage("You're welcome.")],
        }
    )
    async with support_agent(model) as agent:
        first = await agent.send("Where is A1002?")
        await agent.send("Thanks", context_id=first.context_id)

    prompt = model.prompts[-1]
    [tool_result] = [message for message in prompt if isinstance(message, ToolMessage)]
    assert tool_result.tool_call_id == "c1"
    assert json.loads(tool_result.text) == lookup_order("A1002")


def test_every_tool_converts_to_an_openai_tool_schema() -> None:
    """A real chat model binds the plain functions, so each needs a valid schema."""
    names = [convert_to_openai_tool(tool)["function"]["name"] for tool in TOOLS]

    assert names == ["lookup_order", "refund_order", "escalate_to_human"]
