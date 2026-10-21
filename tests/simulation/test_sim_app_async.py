"""Tests for the app that carries each simulated user message to the agent."""

import asyncio
from typing import Any

import pytest

from tests.fakes import FakeAgent
from trajectory_evals.core.types import AgentTurn, ChatCompletionMessage, ToolMessage
from trajectory_evals.simulation import A2AConversationApp
from trajectory_evals.simulation.settings import settings

STOP = "END_CALL"
THREAD = "thread-1"
STOPPED: ChatCompletionMessage = {
    "role": "assistant",
    "content": settings.prompts.stopped_reply,
}


def _user(content: str | list[dict[str, str]]) -> ChatCompletionMessage:
    """Return a simulated user message as openevals passes it to the app."""
    return {"role": "user", "content": content}


def _app(agent: FakeAgent, **kwargs: Any) -> A2AConversationApp:
    """Return an app over agent that ends a thread on STOP."""
    return A2AConversationApp(agent, stop_token=STOP, **kwargs)


def test_respond_passes_openevals_coroutine_check() -> None:
    """openevals awaits the app only when asyncio reads it as a coroutine function."""
    assert asyncio.iscoroutinefunction(_app(FakeAgent()).respond)


@pytest.mark.asyncio
async def test_stop_token_ends_the_thread_without_calling_the_agent() -> None:
    """A message holding the stop token never reaches the agent or the transcript."""
    agent = FakeAgent()
    app = _app(agent)

    reply = await app.respond(_user(f"Thanks, that's all. {STOP}"), thread_id=THREAD)

    assert reply == STOPPED
    assert agent.calls == []
    assert app.transcript(THREAD) == []
    assert app.ended(THREAD)


@pytest.mark.asyncio
async def test_stop_token_after_a_turn_leaves_the_outputs_as_they_were() -> None:
    """Outputs hold the answered turn; the stop token and placeholder are absent."""
    agent = FakeAgent("Hi there.")
    app = _app(agent)

    await app.respond(_user("Hello"), thread_id=THREAD)
    await app.respond(_user(STOP), thread_id=THREAD)

    assert app.outputs(THREAD) == {
        "text": "Hi there.",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there."},
        ],
        "trajectory_reported": True,
    }
    assert [call.text for call in agent.calls] == ["Hello"]


@pytest.mark.asyncio
async def test_transcript_keeps_tool_messages_but_openevals_gets_the_reply() -> None:
    """The turn's tool call and result enter the transcript as given; one goes out."""
    result: ToolMessage = {"role": "tool", "tool_call_id": "call-1", "content": "x"}
    turn = AgentTurn(
        text="Order 7 shipped.",
        messages=(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup_order", "arguments": '{"id": 7}'},
                    }
                ],
            },
            result,
            {"role": "assistant", "content": "Order 7 shipped."},
        ),
        context_id="ctx-1",
    )
    agent = FakeAgent(turn, "Anything else?")
    app = _app(agent, prompt_overrides={"system": "Be brief."}, context={"ch": "sms"})

    reply = await app.respond(_user("Where is order 7?"), thread_id=THREAD)
    await app.respond(_user("Thanks"), thread_id=THREAD)

    assert reply == {"role": "assistant", "content": "Order 7 shipped."}
    assert app.transcript(THREAD) == [
        {"role": "user", "content": "Where is order 7?"},
        *turn.messages,
        {"role": "user", "content": "Thanks"},
        {"role": "assistant", "content": "Anything else?"},
    ]
    assert [call.context_id for call in agent.calls] == [None, "ctx-1"]
    sent = [(call.prompt_overrides, call.context) for call in agent.calls]
    assert sent == [({"system": "Be brief."}, {"ch": "sms"})] * 2


@pytest.mark.asyncio
async def test_one_turn_without_a_trajectory_marks_the_outputs_unreported() -> None:
    """trajectory_reported is True only if every agent turn reported its trajectory."""
    unreported = AgentTurn(
        text="Hi.",
        messages=({"role": "assistant", "content": "Hi."},),
        context_id="ctx-1",
        trajectory_reported=False,
    )
    app = _app(FakeAgent("Hello.", unreported))

    await app.respond(_user("Hello"), thread_id=THREAD)
    await app.respond(_user("Hi"), thread_id=THREAD)

    assert app.outputs(THREAD)["trajectory_reported"] is False


@pytest.mark.asyncio
async def test_empty_prompt_overrides_are_sent_as_none() -> None:
    """An empty overrides mapping means none, so the agent need not acknowledge it."""
    agent = FakeAgent("ok")
    await _app(agent, prompt_overrides={}).respond(_user("Hi"), thread_id=THREAD)
    assert agent.calls[0].prompt_overrides is None


@pytest.mark.asyncio
async def test_list_content_is_sent_as_its_text() -> None:
    """Content blocks reach the agent as the text of their text blocks."""
    agent = FakeAgent("ok")
    blocks = [{"type": "reasoning", "summary": ""}, {"type": "text", "text": "Hello"}]
    await _app(agent).respond(_user(blocks), thread_id=THREAD)
    assert agent.calls[0].text == "Hello"


@pytest.mark.asyncio
async def test_interleaved_threads_keep_their_own_context_and_transcript() -> None:
    """Two conversations taking turns never share a context id or a message."""
    agent = FakeAgent(respond=lambda call: call.text.upper())
    app = _app(agent)

    for text, thread_id in [("a 1", "a"), ("b 1", "b"), ("a 2", "a"), ("b 2", "b")]:
        await app.respond(_user(text), thread_id=thread_id)

    assert [(call.text, call.context_id) for call in agent.calls] == [
        ("a 1", None),
        ("b 1", None),
        ("a 2", "fake-context-1"),
        ("b 2", "fake-context-2"),
    ]
    for thread_id in ("a", "b"):
        contents = [message["content"] for message in app.transcript(thread_id)]
        first, second = f"{thread_id} 1", f"{thread_id} 2"
        assert contents == [first, first.upper(), second, second.upper()]
    assert app.transcript("unknown") == []
    assert not app.ended("unknown")
