"""Tests for the bridge: a Responder served over A2A, driven by A2AAgent in process."""

import asyncio
import logging
from typing import Any

import pytest

from tests.connect.connect_fakes import (
    RecordingResponder,
    agent_config,
    lookup_messages,
)
from trajectory_evals.connect.bridge import (
    AgentReply,
    AgentRequest,
    agent_card,
    app_from_config,
    build_app,
)
from trajectory_evals.connect.client import A2AAgent
from trajectory_evals.connect.settings import settings
from trajectory_evals.core.agent_config import AgentConfigError
from trajectory_evals.core.types import AgentError, AgentTurn, ChatCompletionMessage

SECRET = "card 4111-1111-1111-1111"
HISTORY: list[ChatCompletionMessage] = [
    {"role": "user", "content": "Do you have a Civic?"},
    {"role": "assistant", "content": "Yes, two."},
]


def _agent(responder: Any) -> A2AAgent:
    """Return an A2AAgent reaching a bridge over responder in process."""
    card = agent_card(agent_config(), settings.bridge.url)
    return A2AAgent.in_process(build_app(responder, card))


def _user(text: str) -> ChatCompletionMessage:
    """Return a user message."""
    return {"role": "user", "content": text}


@pytest.mark.asyncio
async def test_a_turn_returns_the_reply_and_its_trajectory(
    network_attempts: list[Any],
) -> None:
    """The trajectory artifact comes back as the turn's messages, tool calls first."""
    responder = RecordingResponder()
    async with _agent(responder) as agent:
        turn = await agent.send("Price?")

    assert turn == AgentTurn(
        text="echo: Price?",
        messages=lookup_messages("Price?"),
        context_id=turn.context_id,
        trajectory_reported=True,
    )
    assert turn.context_id
    [request] = responder.requests
    assert request == AgentRequest(
        message="Price?",
        history=(),
        context_id=turn.context_id,
        prompt_overrides={},
        context={},
    )
    assert network_attempts == []


@pytest.mark.asyncio
async def test_a_second_turn_sees_the_whole_transcript_before_it() -> None:
    """The bridge keeps each context's transcript, as the SDK keeps none."""
    responder = RecordingResponder()
    async with _agent(responder) as agent:
        first = await agent.send("Hi")
        second = await agent.send("Price?", context_id=first.context_id)

    assert second.context_id == first.context_id
    expected = (_user("Hi"), *lookup_messages("Hi"))
    assert responder.requests[1].history == expected


@pytest.mark.asyncio
async def test_history_seeds_a_new_context_and_is_acknowledged() -> None:
    """History reaches the responder once, then stays in the transcript."""
    responder = RecordingResponder()
    async with _agent(responder) as agent:
        first = await agent.send("Price?", history=HISTORY)
        await agent.send("Colour?", context_id=first.context_id)

    seeded, later = responder.requests
    assert seeded.history == tuple(HISTORY)
    assert later.history == (*HISTORY, _user("Price?"), *lookup_messages("Price?"))


@pytest.mark.asyncio
async def test_applied_overrides_and_context_reach_the_responder() -> None:
    """Overrides and context arrive apart; applied overrides are acknowledged."""
    responder = RecordingResponder(apply_overrides=True)
    overrides = {"system": "Be brief."}
    async with _agent(responder) as agent:
        await agent.send("Hi", prompt_overrides=overrides, context={"channel": "sms"})

    [request] = responder.requests
    assert (request.prompt_overrides, request.context) == (
        overrides,
        {"channel": "sms"},
    )


@pytest.mark.asyncio
async def test_concurrent_conversations_keep_their_own_transcripts() -> None:
    """Interleaved sends on two contexts never mix their transcripts."""
    responder = RecordingResponder()
    async with _agent(responder) as agent:
        first_a, first_b = await asyncio.gather(agent.send("A1"), agent.send("B1"))
        await asyncio.gather(
            agent.send("A2", context_id=first_a.context_id),
            agent.send("B2", context_id=first_b.context_id),
        )

    histories = {request.message: request.history for request in responder.requests}
    assert histories["A2"] == (_user("A1"), *lookup_messages("A1"))
    assert histories["B2"] == (_user("B1"), *lookup_messages("B1"))


@pytest.mark.asyncio
async def test_a_failing_responder_fails_the_task_without_leaking_its_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The client sees the generic message; only the bridge log has the cause."""

    async def failing(request: AgentRequest) -> AgentReply:
        """Fail with text that must not reach the client."""
        raise RuntimeError(SECRET)

    with caplog.at_level(logging.ERROR, logger="trajectory_evals.connect.bridge"):
        async with _agent(failing) as agent:
            with pytest.raises(AgentError) as raised:
                await agent.send("Pay with my card")

    assert settings.texts.responder_failed in str(raised.value)
    assert "TASK_STATE_FAILED" in str(raised.value)
    assert SECRET not in str(raised.value)
    assert SECRET in caplog.text


@pytest.mark.asyncio
async def test_a_reply_json_cannot_hold_fails_generically_and_is_not_kept() -> None:
    """A non-JSON value never reaches the client, nor the next turn's history."""
    requests: list[AgentRequest] = []

    async def unencodable_once(request: AgentRequest) -> AgentReply:
        """Reply with bytes, which JSON cannot hold, on the first turn only."""
        requests.append(request)
        if len(requests) > 1:
            return AgentReply(text="ok")
        message: Any = {"role": "assistant", "content": SECRET.encode()}
        return AgentReply(text="x", messages=(message,))

    async with _agent(unencodable_once) as agent:
        with pytest.raises(AgentError) as raised:
            await agent.send("first")
        await agent.send("second", context_id=requests[0].context_id)

    assert settings.texts.responder_failed in str(raised.value)
    assert SECRET not in str(raised.value)
    assert requests[1].history == ()


@pytest.mark.asyncio
async def test_whole_numbers_reach_the_responder_as_ints() -> None:
    """protobuf turns every number into a float; the bridge turns them back."""
    responder = RecordingResponder()
    context = {"dealer_id": 123, "lead": {"id": 9, "scores": [1, 0.5]}}
    async with _agent(responder) as agent:
        await agent.send("Hi", context=context)

    received = responder.requests[0].context
    assert received == context
    assert type(received["dealer_id"]) is int
    assert type(received["lead"]["scores"][0]) is int


def test_app_from_config_refuses_an_agent_without_a_bridge() -> None:
    """Only a [bridge] table names a responder to serve."""
    with pytest.raises(AgentConfigError, match="agent shop has no \\[bridge\\]"):
        app_from_config(agent_config())


def test_serving_the_card_logs_no_spec_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The card's one skill is the agent, so a2a finds no required field empty."""
    card = agent_card(agent_config(), settings.bridge.url)

    with caplog.at_level(logging.WARNING):
        build_app(RecordingResponder(), card)

    assert "not spec-compliant" not in caplog.text
    [skill] = card.skills
    assert (skill.id, skill.description) == ("shop", "A car shop's assistant.")
