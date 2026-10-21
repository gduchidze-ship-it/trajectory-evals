"""Tests for A2AAgent: acknowledgements, replies, continuation, safety, config."""

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from a2a.helpers import new_data_part, new_task, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import Message, Part, TaskState
from starlette.applications import Starlette

from tests.connect.connect_fakes import (
    RecordingResponder,
    agent_config,
    lookup_messages,
)
from trajectory_evals.connect.bridge import (
    AgentReply,
    AgentRequest,
    agent_card,
    build_app,
)
from trajectory_evals.connect.client import A2AAgent
from trajectory_evals.connect.settings import settings
from trajectory_evals.core.agent_config import load_agent
from trajectory_evals.core.types import AgentError, AgentTurn, ChatCompletionMessage

TOKEN = "tok-4111-secret"
URL = "http://shop.test/"
CARD_URL = URL + ".well-known/agent-card.json"
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
    """Asks which car on a new task; books, or fails, when that task is continued.

    Every turn adds a trajectory artifact naming the text it answered.
    """

    def __init__(self, *, fail_continuation: bool = False) -> None:
        """Start with no calls; each call records its task id and if it continued."""
        self.fail_continuation = fail_continuation
        self.calls: list[tuple[str, bool]] = []

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Require input on a new task; complete or fail a continued one."""
        assert context.task_id and context.context_id and context.message
        continued = context.current_task is not None
        self.calls.append((context.task_id, continued))
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not continued:
            submitted = TaskState.TASK_STATE_SUBMITTED
            task = new_task(
                context.task_id,
                context.context_id,
                submitted,
                history=[context.message],
            )
            await event_queue.enqueue_event(task)
        text = context.get_user_input()
        turn = {"messages": [{"role": "assistant", "content": f"turn for {text}"}]}
        await updater.add_artifact([new_data_part(turn)], name="trajectory")
        if not continued:
            await updater.requires_input(_says(updater, "Which car?"))
        elif self.fail_continuation:
            await updater.failed(_says(updater, "booking system down"))
        else:
            await updater.complete(_says(updater, f"Booked: {text}"))


class ArtifactExecutor(_TestExecutor):
    """Completes each message with the given named artifacts and the status "done"."""

    def __init__(self, *artifacts: tuple[str, Part]) -> None:
        """Hold the (name, part) of each artifact to send."""
        self.artifacts = artifacts

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Send each artifact, then complete with the status message "done"."""
        assert context.task_id and context.context_id and context.message
        submitted = TaskState.TASK_STATE_SUBMITTED
        task = new_task(
            context.task_id, context.context_id, submitted, history=[context.message]
        )
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        for name, part in self.artifacts:
            await updater.add_artifact([part], name=name)
        await updater.complete(_says(updater, "done"))


def _says(updater: TaskUpdater, text: str) -> Message:
    """Return an agent message holding text."""
    return updater.new_agent_message([Part(text=text)])


def _executor_app(executor: AgentExecutor, url: str = settings.bridge.url) -> Starlette:
    """Return an A2A app answering with a hand-written executor, its card at url."""
    card = agent_card(agent_config(), url)
    handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card
    )
    routes = [*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")]
    return Starlette(routes=routes)


def _executor_agent(executor: AgentExecutor, *, replay: bool = False) -> A2AAgent:
    """Return an A2AAgent reaching a hand-written executor in process."""
    return A2AAgent.in_process(_executor_app(executor), replay=replay)


def _bridge_agent(responder: Any, *, timeout_s: float | None = None) -> A2AAgent:
    """Return an A2AAgent reaching a bridge over responder in process."""
    card = agent_card(agent_config(), settings.bridge.url)
    return A2AAgent.in_process(build_app(responder, card), timeout_s=timeout_s)


def _agent_dir(tmp_path: Path, toml: str) -> Path:
    """Write tmp_path/shop/agent.toml holding toml and return the folder."""
    root = tmp_path / "shop"
    root.mkdir()
    (root / "agent.toml").write_text(toml, encoding="utf-8")
    return root


def _route_clients_to(
    app: Starlette, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, str | None]]:
    """Make each new httpx client answer from app; return each url and bearer."""
    served = httpx.ASGITransport(app=app)
    seen: list[tuple[str, str | None]] = []

    async def forward(request: httpx.Request) -> httpx.Response:
        """Record the url and Authorization header, then let app answer."""
        seen.append((str(request.url), request.headers.get("Authorization")))
        return await served.handle_async_request(request)

    real_client = httpx.AsyncClient

    def routed_client(**kwargs: Any) -> httpx.AsyncClient:
        """Build the client asked for, answering through forward."""
        return real_client(transport=httpx.MockTransport(forward), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", routed_client)
    return seen


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
async def test_an_answer_artifact_is_the_reply_over_the_status_text() -> None:
    """A task that answers in an artifact, with status "done", replies with it."""
    answer = ("", Part(text="The Civic is $21,000"))
    async with _executor_agent(ArtifactExecutor(answer)) as agent:
        turn = await agent.send("Price?")

    assert turn.text == "The Civic is $21,000"
    assert turn.messages == ({"role": "assistant", "content": turn.text},)
    assert not turn.trajectory_reported


MALFORMED_TRAJECTORIES = {
    "no_messages": {"steps": []},
    "not_an_object": ["x"],
    "messages_not_a_list": {"messages": "x"},
    "message_not_an_object": {"messages": ["x"]},
}


@pytest.mark.parametrize(
    "data", MALFORMED_TRAJECTORIES.values(), ids=MALFORMED_TRAJECTORIES
)
@pytest.mark.asyncio
async def test_a_malformed_trajectory_is_an_agent_error(data: Any) -> None:
    """A trajectory artifact must hold {"messages": [message objects]}."""
    executor = ArtifactExecutor(("trajectory", new_data_part(data)))
    async with _executor_agent(executor) as agent:
        with pytest.raises(AgentError, match="trajectory artifact does not hold"):
            await agent.send("Hi")


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
async def test_input_required_is_continued_with_this_turns_trajectory() -> None:
    """The next send continues an INPUT_REQUIRED task; a finished one is not.

    A continued task keeps every turn's artifacts, and the turn is the last one's.
    """
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
    assert booked.messages == ({"role": "assistant", "content": "turn for The Civic"},)
    (first_task, _), continued, (third_task, third_continued) = executor.calls
    assert continued == (first_task, True)
    assert third_task != first_task and not third_continued


@pytest.mark.asyncio
async def test_a_failed_continuation_lets_the_next_send_start_a_new_task() -> None:
    """The failed task id is dropped, so the context stays usable."""
    executor = AskFirstExecutor(fail_continuation=True)
    async with _executor_agent(executor) as agent:
        asked = await agent.send("Book a test drive")
        with pytest.raises(AgentError, match="TASK_STATE_FAILED: booking system down"):
            await agent.send("The Civic", context_id=asked.context_id)
        again = await agent.send("Retry", context_id=asked.context_id)

    assert again.text == "Which car?"
    assert executor.calls[2][1] is False


@pytest.mark.asyncio
async def test_a_slow_agent_times_out_even_in_process() -> None:
    """ASGITransport ignores httpx timeouts, so the exchange carries its own."""

    async def slow(request: AgentRequest) -> AgentReply:
        """Reply long after the client's timeout."""
        await asyncio.sleep(5)
        return AgentReply(text="late")

    async with _bridge_agent(slow, timeout_s=0.05) as agent:
        with pytest.raises(AgentError, match="did not reply within 0.05 seconds"):
            await agent.send("Hi")


@pytest.mark.asyncio
async def test_a_card_with_no_usable_transport_is_an_agent_error() -> None:
    """A card offering only gRPC leaves the JSON-RPC client nothing to use."""
    card = agent_card(agent_config(), settings.bridge.url)
    card.supported_interfaces[0].protocol_binding = "GRPC"
    app = Starlette(routes=create_agent_card_routes(card))

    async with A2AAgent.in_process(app) as agent:
        with pytest.raises(AgentError, match="no compatible transports"):
            await agent.send("Hi")


@pytest.mark.asyncio
async def test_from_config_serves_a_bridge_in_process(tmp_path: Path) -> None:
    """With no url, [bridge] is built in process, with its options."""
    config = load_agent(_agent_dir(tmp_path, BRIDGE_AGENT))

    async with A2AAgent.from_config(config) as agent:
        turn = await agent.send("Hi", prompt_overrides={"system": "Be brief."})

    assert turn.messages == lookup_messages("Hi")


@pytest.mark.parametrize("value", [TOKEN, f"{TOKEN}\r\n"], ids=["plain", "crlf"])
@pytest.mark.asyncio
async def test_from_config_sends_the_bearer_token_to_the_url(
    value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    network_attempts: list[Any],
) -> None:
    """The card fetch and the message carry the token, without a CRLF's \\r."""
    monkeypatch.setenv("SHOP_TOKEN", value)
    config = load_agent(_agent_dir(tmp_path, URL_AGENT))
    app = build_app(RecordingResponder(), agent_card(config, URL))
    seen = _route_clients_to(app, monkeypatch)

    async with A2AAgent.from_config(config) as agent:
        turn = await agent.send("Hi")

    assert turn.text == "echo: Hi"
    assert seen == [(CARD_URL, f"Bearer {TOKEN}"), (URL, f"Bearer {TOKEN}")]
    assert network_attempts == []


@pytest.mark.asyncio
async def test_the_token_is_never_sent_to_another_origin_the_card_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the configured origin sees the token; the error names both origins."""
    monkeypatch.setenv("SHOP_TOKEN", TOKEN)
    config = load_agent(_agent_dir(tmp_path, URL_AGENT))
    responder = RecordingResponder()
    app = build_app(responder, agent_card(config, "http://other.test:8080/"))
    seen = _route_clients_to(app, monkeypatch)

    async with A2AAgent.from_config(config) as agent:
        with pytest.raises(AgentError) as raised:
            await agent.send("Hi")

    message = str(raised.value)
    assert "http://shop.test sends messages to http://other.test:8080" in message
    assert TOKEN not in message
    assert seen == [(CARD_URL, f"Bearer {TOKEN}")]
    assert responder.requests == []


@pytest.mark.parametrize("value", [None, " \r\n"], ids=["unset", "blank"])
def test_from_config_names_a_missing_token_variable(
    value: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unset or blank token variable fails before any request."""
    if value is None:
        monkeypatch.delenv("SHOP_TOKEN", raising=False)
    else:
        monkeypatch.setenv("SHOP_TOKEN", value)
    config = load_agent(_agent_dir(tmp_path, URL_AGENT))
    with pytest.raises(AgentError, match="environment variable SHOP_TOKEN is not set"):
        A2AAgent.from_config(config)
