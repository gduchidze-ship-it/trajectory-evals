"""Tests for the sequence-runner Responder, over httpx.MockTransport."""

import json
from collections.abc import Callable
from typing import Any

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
    "tool_calls_path": "reply._tool_calls",
}
TOOL_CALLS = {
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
FINAL_STATE = {"reply": {"text": "It is $21,000.", "_tool_calls": TOOL_CALLS}}
HISTORY: tuple[ChatCompletionMessage, ...] = (
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
)


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
async def test_the_reply_follows_each_recorded_tool_call_and_result(
    token: str,
) -> None:
    """Calls group by tool, in record order; an error becomes the result."""
    respond = factory(OPTIONS, transport=_runner(FINAL_STATE, []))

    reply = await respond(_request())

    error = json.dumps({"error": {"type": "ToolError", "message": "closed"}})
    assert reply == AgentReply(
        text="It is $21,000.",
        messages=(
            _call("call_0", "lookup_vehicle", '{"vin": "1HG"}'),
            _result("call_0", '{"price": 21000}'),
            _call("call_1", "get_hours", "{}"),
            _result("call_1", "9-5"),
            _call("call_2", "get_hours", '{"day": "sun"}'),
            _result("call_2", error),
            {"role": "assistant", "content": "It is $21,000."},
        ),
        prompt_overrides_applied=False,
    )


@pytest.mark.asyncio
async def test_a_state_without_tool_calls_gives_only_the_reply(token: str) -> None:
    """The runner records _tool_calls only for a step that called a tool."""
    state = {"reply": {"text": "Hello."}}
    respond = factory(OPTIONS, transport=_runner(state, []))

    reply = await respond(_request())

    assert reply.messages == ({"role": "assistant", "content": "Hello."},)


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
async def test_url_env_names_where_the_runner_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """url_env reads the url when the factory runs; no token sends no header."""
    monkeypatch.setenv("RUNNER_URL", "http://local.test:8000")
    options = {**OPTIONS, "url": None, "url_env": "RUNNER_URL", "token_env": None}
    sent: list[httpx.Request] = []

    await factory(options, transport=_runner(FINAL_STATE, sent))(_request())

    assert str(sent[0].url) == "http://local.test:8000/execute"
    assert "Authorization" not in sent[0].headers


BAD_OPTIONS: dict[str, tuple[Callable[[dict[str, Any]], dict[str, Any]], str]] = {
    "both_urls": (lambda options: {**options, "url_env": "U"}, "exactly one of url"),
    "no_url": (lambda options: {**options, "url": None}, "exactly one of url"),
    "unknown_key": (lambda options: {**options, "urll": "x"}, "urll"),
    "no_reply_path": (
        lambda options: {k: v for k, v in options.items() if k != "reply_path"},
        "reply_path",
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
    """The runner takes no overrides, so the client refuses an unapplied one."""
    respond = factory(OPTIONS, transport=_runner(FINAL_STATE, []))
    card = agent_card(agent_config(), settings.bridge.url)

    async with A2AAgent.in_process(build_app(respond, card)) as agent:
        turn = await agent.send("Price?")
        with pytest.raises(AgentError, match="prompt_overrides"):
            await agent.send("Price?", prompt_overrides={"system": "Be brief."})

    assert [message["role"] for message in turn.messages] == [
        *["assistant", "tool"] * 3,
        "assistant",
    ]
    assert network_attempts == []
