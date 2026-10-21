"""Tests for A2AAgent: acknowledgements, fallbacks, replay, continuation, config."""

from pathlib import Path
from typing import Any

import httpx
import pytest
from a2a.helpers import new_task, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import Part, TaskState
from starlette.applications import Starlette

from tests.connect.connect_fakes import (
    RecordingResponder,
    agent_config,
    lookup_messages,
)
from trajectory_evals.connect.bridge import agent_card, build_app
from trajectory_evals.connect.client import A2AAgent
from trajectory_evals.connect.settings import settings
from trajectory_evals.core.agent_config import load_agent
from trajectory_evals.core.types import AgentError, AgentTurn, ChatCompletionMessage

TOKEN = "tok-4111-secret"
URL = "http://shop.test/"
URL_AGENT = (
    f'[agent]\nurl = "{URL}"\nauth_token_env = "SHOP_TOKEN"\n'
    'description = "A car shop."\n'
)
BRIDGE_AGENT = (
    '[agent]\ndescription = "A car shop."\n'
    '[bridge]\nresponder = "tests.connect.connect_fakes:recording_factory"\n'
    "options = { apply_overrides = true }\n"
)
HISTORY: list[ChatCompletionMessage] = [
    {"role": "user", "content": "First"},
    {"role": "assistant", "content": "a"},
    {"role": "user", "content": [{"type": "text", "text": "Second"}]},
    {"role": "assistant", "content": "b"},
]


class _TestExecutor(AgentExecutor):
    """An executor base for these tests, which never cancel a task."""

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Fail loudly: no test cancels a task."""
        raise AssertionError("cancel is not under test")


class TextOnlyExecutor(_TestExecutor):
    """Replies with one plain Message, like an agent that knows no convention."""

    def __init__(self) -> None:
        """Start with no messages received."""
        self.received: list[tuple[str | None, str]] = []

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Record the context and text, then reply "ok: <text>"."""
        text = context.get_user_input()
        self.received.append((context.context_id, text))
        reply = new_text_message(
            f"ok: {text}", context_id=context.context_id, task_id=context.task_id
        )
        await event_queue.enqueue_event(reply)


class AskFirstExecutor(_TestExecutor):
    """Asks which car on a new task, and books it when that task is continued."""

    def __init__(self) -> None:
        """Start with no calls; each call records its task id and if it continued."""
        self.calls: list[tuple[str, bool]] = []

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Require input on a new task; complete a continued one."""
        assert context.task_id and context.context_id and context.message
        continued = context.current_task is not None
        self.calls.append((context.task_id, continued))
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if continued:
            booked = Part(text=f"Booked: {context.get_user_input()}")
            await updater.complete(updater.new_agent_message([booked]))
            return
        submitted = TaskState.TASK_STATE_SUBMITTED
        history = [context.message]
        await event_queue.enqueue_event(
            new_task(context.task_id, context.context_id, submitted, history=history)
        )
        question = updater.new_agent_message([Part(text="Which car?")])
        await updater.requires_input(question)


def _executor_agent(executor: AgentExecutor, *, replay: bool = False) -> A2AAgent:
    """Return an A2AAgent reaching a hand-written executor in process."""
    card = agent_card(agent_config(), settings.bridge.url)
    handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card
    )
    routes = [*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")]
    return A2AAgent.in_process(Starlette(routes=routes), replay=replay)


def _bridge_agent(responder: RecordingResponder) -> A2AAgent:
    """Return an A2AAgent reaching a bridge over responder in process."""
    card = agent_card(agent_config(), settings.bridge.url)
    return A2AAgent.in_process(build_app(responder, card))


def _agent_dir(tmp_path: Path, toml: str) -> Path:
    """Write tmp_path/shop/agent.toml holding toml and return the folder."""
    root = tmp_path / "shop"
    root.mkdir()
    (root / "agent.toml").write_text(toml, encoding="utf-8")
    return root


@pytest.mark.asyncio
async def test_unapplied_overrides_raise() -> None:
    """An agent that ran its own prompts must not be scored as the candidate's."""
    async with _bridge_agent(RecordingResponder()) as agent:
        with pytest.raises(AgentError, match="did not acknowledge the prompt_override"):
            await agent.send("Hi", prompt_overrides={"system": "Be brief."})


@pytest.mark.asyncio
async def test_empty_overrides_are_not_sent_and_need_no_acknowledgement() -> None:
    """An empty mapping is the same as None."""
    responder = RecordingResponder()
    async with _bridge_agent(responder) as agent:
        await agent.send("Hi", prompt_overrides={})
    assert responder.requests[0].prompt_overrides == {}


@pytest.mark.asyncio
async def test_a_reserved_context_key_is_refused_before_sending() -> None:
    """A context key named like a wire key would be read as that key."""
    responder = RecordingResponder()
    async with _bridge_agent(responder) as agent:
        with pytest.raises(AgentError, match=r"\['history'\] are reserved"):
            await agent.send("Hi", context={"history": [], "channel": "sms"})
    assert responder.requests == []


@pytest.mark.asyncio
async def test_a_text_only_reply_falls_back_to_one_assistant_message(
    network_attempts: list[Any],
) -> None:
    """Without a trajectory artifact the turn says so, with just the reply text."""
    async with _executor_agent(TextOnlyExecutor()) as agent:
        turn = await agent.send("Hi")

    assert turn == AgentTurn(
        text="ok: Hi",
        messages=({"role": "assistant", "content": "ok: Hi"},),
        context_id=turn.context_id,
        trajectory_reported=False,
    )
    assert network_attempts == []


@pytest.mark.asyncio
async def test_unacknowledged_history_raises_and_points_at_replay() -> None:
    """An agent that ignores metadata history would answer out of context."""
    async with _executor_agent(TextOnlyExecutor()) as agent:
        with pytest.raises(AgentError, match='set history = "replay"'):
            await agent.send("Third", history=HISTORY)


@pytest.mark.asyncio
async def test_replay_sends_prior_user_turns_on_the_same_context() -> None:
    """Replay rebuilds the conversation from the user turns, sending no metadata."""
    executor = TextOnlyExecutor()
    async with _executor_agent(executor, replay=True) as agent:
        turn = await agent.send("Third", history=HISTORY)

    assert [text for _, text in executor.received] == ["First", "Second", "Third"]
    assert {context_id for context_id, _ in executor.received} == {turn.context_id}
    assert turn.text == "ok: Third"


@pytest.mark.asyncio
async def test_input_required_is_continued_on_the_same_task() -> None:
    """The next send continues an INPUT_REQUIRED task; a finished one is not."""
    executor = AskFirstExecutor()
    async with _executor_agent(executor) as agent:
        asked = await agent.send("Book a test drive")
        booked = await agent.send("The Civic", context_id=asked.context_id)
        again = await agent.send("Another", context_id=asked.context_id)

    assert (asked.text, booked.text, again.text) == (
        "Which car?",
        "Booked: The Civic",
        "Which car?",
    )
    (first_task, _), continued, (third_task, third_continued) = executor.calls
    assert continued == (first_task, True)
    assert third_task != first_task and not third_continued


@pytest.mark.asyncio
async def test_from_config_serves_a_bridge_in_process(tmp_path: Path) -> None:
    """With no url, [bridge] is built in process, with its options."""
    config = load_agent(_agent_dir(tmp_path, BRIDGE_AGENT))

    async with A2AAgent.from_config(config) as agent:
        turn = await agent.send("Hi", prompt_overrides={"system": "Be brief."})

    assert turn.messages == lookup_messages("Hi")


@pytest.mark.asyncio
async def test_from_config_sends_the_bearer_token_to_the_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, network_attempts: list[Any]
) -> None:
    """The card fetch and the message both carry the token from auth_token_env."""
    monkeypatch.setenv("SHOP_TOKEN", TOKEN)
    config = load_agent(_agent_dir(tmp_path, URL_AGENT))
    served = httpx.ASGITransport(
        app=build_app(RecordingResponder(), agent_card(config, URL))
    )
    seen: list[tuple[str, str | None]] = []

    async def forward(request: httpx.Request) -> httpx.Response:
        """Record the url and Authorization header, then let the bridge answer."""
        seen.append((str(request.url), request.headers.get("Authorization")))
        return await served.handle_async_request(request)

    real_client = httpx.AsyncClient

    def mocked_client(**kwargs: Any) -> httpx.AsyncClient:
        """Build the client from_config asks for, answering through forward."""
        return real_client(transport=httpx.MockTransport(forward), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", mocked_client)

    async with A2AAgent.from_config(config) as agent:
        turn = await agent.send("Hi")

    assert turn.text == "echo: Hi"
    card_url = URL + ".well-known/agent-card.json"
    assert seen == [(card_url, f"Bearer {TOKEN}"), (URL, f"Bearer {TOKEN}")]
    assert network_attempts == []


def test_from_config_names_a_missing_token_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured token variable that is unset fails before any request."""
    monkeypatch.delenv("SHOP_TOKEN", raising=False)
    config = load_agent(_agent_dir(tmp_path, URL_AGENT))
    with pytest.raises(AgentError, match="environment variable SHOP_TOKEN is not set"):
        A2AAgent.from_config(config)


@pytest.mark.asyncio
async def test_an_unreachable_agent_is_an_agent_error_without_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, network_attempts: list[Any]
) -> None:
    """Transport failures surface as AgentError, and never echo the token."""
    monkeypatch.setenv("SHOP_TOKEN", TOKEN)
    config = load_agent(_agent_dir(tmp_path, URL_AGENT))

    async with A2AAgent.from_config(config) as agent:
        with pytest.raises(AgentError) as raised:
            await agent.send("Hi")

    assert network_attempts
    assert TOKEN not in str(raised.value)
