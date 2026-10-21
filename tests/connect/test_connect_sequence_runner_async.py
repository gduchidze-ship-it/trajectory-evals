"""Tests for the sequence-runner Responder, over httpx.MockTransport."""

import json
import re
from collections.abc import Callable, Mapping
from typing import Any, cast

import httpx
import pytest
from pydantic import ValidationError

from tests.connect.connect_fakes import agent_config
from trajectory_evals.connect.bridge import (
    AgentReply,
    AgentRequest,
    agent_card,
    build_app,
)
from trajectory_evals.connect.client import A2AAgent
from trajectory_evals.connect.sequence_runner import factory
from trajectory_evals.connect.settings import settings
from trajectory_evals.core.types import AgentError, ChatCompletionMessage, ToolMessage

TOKEN = "tok-runner"
SECRET = "my card is 4111-1111-1111-1111"
OPTIONS: dict[str, Any] = {
    "url": "http://runner.test",
    "token_env": "RUNNER_TOKEN",
    "sequence_id": "sales_reply",
    "client_key": "C0020177",
    "product_id": "sales_ai",
    "initial_state": {"incoming_message": {"channel": "sms"}},
    "messages_key": "messages",
    "reply_path": "reply.text",
    "tool_calls_path": ["reply._tool_calls", "send_reply_tool_calls"],
}
REPLY_CALLS = {
    "lookup_vehicle": [
        {"arguments": {"vin": "1HG"}, "response": {"price": 21000}, "error": None}
    ],
    "get_hours": [
        {"arguments": {}, "response": "9-5", "error": None},
        {
            "arguments": {"day": "sun"},
            "response": None,
            "error": {"type": "ToolError", "message": "closed"},
        },
    ],
}
SEND_CALLS = {
    "send_sms": [{"arguments": {"to": "+1555"}, "response": "sent", "error": None}]
}
FINAL_STATE = {
    "reply": {"text": "It is $21,000.", "_tool_calls": REPLY_CALLS},
    "send_reply_tool_calls": SEND_CALLS,
}
HISTORY: tuple[ChatCompletionMessage, ...] = (
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
)
CALL_ID = re.compile(r"call_([0-9a-f]{8})_\d+")


def _request(**overrides: Any) -> AgentRequest:
    """Return a request for "Price?" after HISTORY, with fields overridden."""
    fields: dict[str, Any] = {
        "message": "Price?",
        "history": HISTORY,
        "context_id": "ctx-1",
        "prompt_overrides": {},
        "context": {"lead_id": "L-9"},
    }
    return AgentRequest(**{**fields, **overrides})


def _runner(
    state: Any, sent: list[httpx.Request], status: int = 200
) -> httpx.MockTransport:
    """Return a transport that records each request and answers with state."""

    def answer(request: httpx.Request) -> httpx.Response:
        """Record the request and return the final state."""
        sent.append(request)
        return httpx.Response(status, json=state)

    return httpx.MockTransport(answer)


def _call(call_id: str, name: str, arguments: str) -> ChatCompletionMessage:
    """Return an assistant message making one tool call."""
    function = {"name": name, "arguments": arguments}
    call = {"id": call_id, "type": "function", "function": function}
    return {"role": "assistant", "content": "", "tool_calls": [call]}


def _result(call_id: str, content: str) -> ToolMessage:
    """Return the tool result answering call_id."""
    return {"role": "tool", "content": content, "tool_call_id": call_id}


def _tool_call_ids(reply: AgentReply) -> list[str]:
    """Return the tool_call_id of each tool result in the reply, in order."""
    messages = cast(tuple[Mapping[str, Any], ...], reply.messages)
    return [
        message["tool_call_id"] for message in messages if message["role"] == "tool"
    ]


@pytest.fixture
def token(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set the runner token variable and return its value."""
    monkeypatch.setenv("RUNNER_TOKEN", TOKEN)
    return TOKEN


@pytest.mark.asyncio
async def test_a_turn_posts_the_conversation_with_the_bearer_token(token: str) -> None:
    """initial_state merges the static state, the context and the messages."""
    sent: list[httpx.Request] = []
    respond = factory(OPTIONS, transport=_runner(FINAL_STATE, sent))

    await respond(_request())

    [request] = sent
    assert (request.method, str(request.url)) == ("POST", "http://runner.test/execute")
    assert request.headers["Authorization"] == f"Bearer {token}"
    assert json.loads(request.content) == {
        "sequence_id": "sales_reply",
        "client_key": "C0020177",
        "product_id": "sales_ai",
        "initial_state": {
            "incoming_message": {"channel": "sms"},
            "lead_id": "L-9",
            "messages": [*HISTORY, {"role": "user", "content": "Price?"}],
        },
    }


@pytest.mark.asyncio
async def test_the_reply_follows_each_step_s_tool_calls_and_results(
    token: str,
) -> None:
    """Paths in order, calls grouped by tool; an error becomes the result."""
    respond = factory(OPTIONS, transport=_runner(FINAL_STATE, []))

    reply = await respond(_request())

    ids = _tool_call_ids(reply)
    match = CALL_ID.fullmatch(ids[0])
    assert match is not None
    assert ids == [f"call_{match.group(1)}_{index}" for index in range(4)]
    error = json.dumps({"error": {"type": "ToolError", "message": "closed"}})
    assert reply == AgentReply(
        text="It is $21,000.",
        messages=(
            _call(ids[0], "lookup_vehicle", '{"vin": "1HG"}'),
            _result(ids[0], '{"price": 21000}'),
            _call(ids[1], "get_hours", "{}"),
            _result(ids[1], "9-5"),
            _call(ids[2], "get_hours", '{"day": "sun"}'),
            _result(ids[2], error),
            _call(ids[3], "send_sms", '{"to": "+1555"}'),
            _result(ids[3], "sent"),
            {"role": "assistant", "content": "It is $21,000."},
        ),
        prompt_overrides_applied=False,
    )


@pytest.mark.asyncio
async def test_tool_call_ids_never_repeat_across_turns(token: str) -> None:
    """A conversation's transcript holds every turn's calls, so ids must differ."""
    respond = factory(OPTIONS, transport=_runner(FINAL_STATE, []))

    first, second = await respond(_request()), await respond(_request())

    assert set(_tool_call_ids(first)).isdisjoint(_tool_call_ids(second))


@pytest.mark.asyncio
async def test_a_state_without_tool_calls_gives_only_the_reply(token: str) -> None:
    """The runner records _tool_calls only for a step that called a tool."""
    state = {"reply": {"text": "Hello."}}
    respond = factory(OPTIONS, transport=_runner(state, []))

    reply = await respond(_request())

    assert reply.messages == ({"role": "assistant", "content": "Hello."},)


@pytest.mark.asyncio
async def test_overrides_are_refused_before_the_sequence_runs(token: str) -> None:
    """The runner takes no overrides, so a turn with some never reaches it."""
    sent: list[httpx.Request] = []
    respond = factory(OPTIONS, transport=_runner(FINAL_STATE, sent))

    reply = await respond(_request(prompt_overrides={"system": "Be brief."}))

    assert reply == AgentReply(text="")
    assert sent == []


@pytest.mark.asyncio
async def test_a_missing_reply_names_the_path_but_not_the_state(token: str) -> None:
    """The state can hold customer text, so the error names only the path."""
    state = {"reply": {"draft": SECRET}}
    respond = factory(OPTIONS, transport=_runner(state, []))

    with pytest.raises(ValueError, match="no reply text at reply.text") as raised:
        await respond(_request())
    assert SECRET not in str(raised.value)


@pytest.mark.asyncio
async def test_an_http_failure_raises(token: str) -> None:
    """The bridge turns this into its generic failure."""
    respond = factory(OPTIONS, transport=_runner({"detail": "boom"}, [], status=500))
    with pytest.raises(httpx.HTTPStatusError):
        await respond(_request())


@pytest.mark.asyncio
async def test_no_token_variable_sends_no_authorization() -> None:
    """A local runner skips auth, so token_env is optional."""
    sent: list[httpx.Request] = []
    options = {**OPTIONS, "token_env": None}

    await factory(options, transport=_runner(FINAL_STATE, sent))(_request())

    assert "Authorization" not in sent[0].headers


BAD_OPTIONS: dict[str, tuple[Callable[[dict[str, Any]], dict[str, Any]], str]] = {
    "unknown_key": (lambda options: {**options, "urll": "x"}, "urll"),
    "no_reply_path": (
        lambda options: {k: v for k, v in options.items() if k != "reply_path"},
        "reply_path",
    ),
    "tool_calls_path_not_a_list": (
        lambda options: {**options, "tool_calls_path": "reply._tool_calls"},
        "tool_calls_path",
    ),
}


@pytest.mark.parametrize("change, message", BAD_OPTIONS.values(), ids=BAD_OPTIONS)
def test_bad_options_fail_when_the_factory_runs(
    token: str, change: Callable[[dict[str, Any]], dict[str, Any]], message: str
) -> None:
    """Options are checked once, before any turn."""
    with pytest.raises(ValidationError, match=message):
        factory(change(OPTIONS))


def test_a_missing_token_variable_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """The variable is named; the error never holds a value."""
    monkeypatch.delenv("RUNNER_TOKEN", raising=False)
    with pytest.raises(AgentError, match="environment variable RUNNER_TOKEN"):
        factory(OPTIONS)


@pytest.mark.asyncio
async def test_over_the_bridge_tool_calls_arrive_and_overrides_are_refused(
    token: str, network_attempts: list[Any]
) -> None:
    """The client refuses the unapplied overrides; the sequence ran only once."""
    sent: list[httpx.Request] = []
    respond = factory(OPTIONS, transport=_runner(FINAL_STATE, sent))
    card = agent_card(agent_config(), settings.bridge.url)

    async with A2AAgent.in_process(build_app(respond, card)) as agent:
        turn = await agent.send("Price?")
        with pytest.raises(AgentError, match="prompt_overrides"):
            await agent.send("Price?", prompt_overrides={"system": "Be brief."})

    assert [message["role"] for message in turn.messages] == [
        *["assistant", "tool"] * 4,
        "assistant",
    ]
    assert len(sent) == 1
    assert network_attempts == []
