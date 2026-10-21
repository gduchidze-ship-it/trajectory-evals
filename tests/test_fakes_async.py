"""Tests for FakeAgent, the scripted Agent other tests drive."""

import pytest

from tests.fakes import AgentCall, FakeAgent
from trajectory_evals.core.protocols import Agent
from trajectory_evals.core.types import AgentError, AgentTurn, ChatCompletionMessage


@pytest.mark.asyncio
async def test_scripted_replies_come_back_in_order_on_one_context() -> None:
    """A str reply is one assistant message; a new conversation gets a stable id."""
    agent: Agent = FakeAgent("Hi there.", "We open at 9.")

    first = await agent.send("Hello")
    second = await agent.send("Hours?", context_id=first.context_id)

    assert first == AgentTurn(
        text="Hi there.",
        messages=({"role": "assistant", "content": "Hi there."},),
        context_id="fake-context-1",
    )
    assert (second.text, second.context_id) == ("We open at 9.", "fake-context-1")


@pytest.mark.asyncio
async def test_each_new_conversation_gets_the_next_context_id() -> None:
    """Context ids are numbered in the order conversations start."""
    agent = FakeAgent("a", "b")
    turns = [await agent.send("x"), await agent.send("y")]
    assert [turn.context_id for turn in turns] == ["fake-context-1", "fake-context-2"]


@pytest.mark.asyncio
async def test_respond_makes_the_reply_depend_on_the_call() -> None:
    """A respond function sees the text and prompt overrides of each call."""

    def respond(call: AgentCall) -> str:
        """Answer formally only under the formal system prompt."""
        overrides = call.prompt_overrides or {}
        return "Good day." if overrides.get("system") == "Be formal." else "Hey!"

    agent = FakeAgent(respond=respond)

    formal = await agent.send("Hi", prompt_overrides={"system": "Be formal."})
    casual = await agent.send("Hi")

    assert (formal.text, casual.text) == ("Good day.", "Hey!")


@pytest.mark.asyncio
async def test_every_call_is_recorded_as_sent() -> None:
    """Calls keep copies, so later changes to what was sent do not leak in."""
    history: list[ChatCompletionMessage] = [{"role": "user", "content": "Hi"}]
    overrides = {"system": "Be brief."}
    agent = FakeAgent("ok")

    await agent.send(
        "Price?",
        history=history,
        prompt_overrides=overrides,
        context={"channel": "sms"},
    )
    history.append({"role": "assistant", "content": "Hello"})
    overrides["system"] = "changed"

    assert agent.calls == [
        AgentCall(
            text="Price?",
            context_id=None,
            history=({"role": "user", "content": "Hi"},),
            prompt_overrides={"system": "Be brief."},
            context={"channel": "sms"},
        )
    ]


@pytest.mark.asyncio
async def test_a_scripted_agent_turn_is_returned_as_is() -> None:
    """An AgentTurn reply, tool calls included, comes back unchanged."""
    turn = AgentTurn(
        text="Booked.",
        messages=(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "book", "arguments": '{"day": "Sat"}'},
                    }
                ],
            },
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "Booked."},
        ),
        context_id="ctx-9",
    )
    assert await FakeAgent(turn).send("Book Saturday") is turn


@pytest.mark.asyncio
async def test_respond_can_fail_like_an_agent() -> None:
    """An AgentError from respond reaches the caller, and the call is still recorded."""

    def respond(call: AgentCall) -> str:
        """Fail as an unreachable agent would."""
        raise AgentError("agent unreachable")

    agent = FakeAgent(respond=respond)

    with pytest.raises(AgentError, match="agent unreachable"):
        await agent.send("Hello")
    assert [call.text for call in agent.calls] == ["Hello"]


@pytest.mark.asyncio
async def test_running_out_of_replies_fails_the_test() -> None:
    """A send past the end of the script is a test bug, not an agent error."""
    with pytest.raises(AssertionError, match="no scripted reply for 'Hello'"):
        await FakeAgent().send("Hello")


def test_replies_and_respond_together_are_refused() -> None:
    """A FakeAgent takes a script or a respond function, never both."""
    with pytest.raises(TypeError, match="not both"):
        FakeAgent("a", respond=lambda call: "b")
