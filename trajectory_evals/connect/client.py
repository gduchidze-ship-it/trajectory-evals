"""A2AAgent: the Agent for an agent reached over A2A, remote or bridged in process."""

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence, Set
from contextlib import asynccontextmanager
from typing import Any, Self
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, Client, ClientConfig, create_client
from a2a.helpers import get_data_parts, get_message_text, get_text_parts
from a2a.types import (
    AgentCard,
    Artifact,
    Message,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TaskState,
)
from a2a.utils.errors import A2AError
from google.protobuf.json_format import MessageToDict, ParseError
from starlette.applications import Starlette

from trajectory_evals.connect.bridge import (
    APPLIED,
    HISTORY,
    MESSAGES,
    PROMPT_OVERRIDES,
    RESERVED,
    TRAJECTORY,
    app_from_config,
)
from trajectory_evals.connect.settings import AUTHORIZATION, _bearer_headers, settings
from trajectory_evals.core.agent_config import AgentConfig
from trajectory_evals.core.types import AgentError, AgentTurn, ChatCompletionMessage

_INPUT_REQUIRED = TaskState.TASK_STATE_INPUT_REQUIRED
_ANSWERED = (TaskState.TASK_STATE_COMPLETED, _INPUT_REQUIRED)


class A2AAgent:
    """An Agent that sends each message to an A2A agent over JSON-RPC.

    It owns an httpx client: close it with aclose or an async with block.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        url: str,
        *,
        replay: bool = False,
        timeout_s: float | None = None,
    ) -> None:
        """Reach the agent at url through http, which aclose closes.

        replay sends a new conversation's history as earlier user turns on the same
        context, for an agent that cannot take history in message metadata.
        timeout_s bounds each exchange, the first one's card fetch included; None
        leaves only http's own timeouts, which an in-process app ignores.
        """
        self._http = http
        self._url = url
        self._replay = replay
        self._timeout_s = timeout_s
        self._connection: tuple[Client, AgentCard] | None = None
        self._connection_lock = asyncio.Lock()
        self._input_required: dict[str, str] = {}  # context id -> its open task id

    @classmethod
    def from_config(cls, config: AgentConfig) -> Self:
        """Reach [agent].url with its bearer token, or serve [bridge] in process.

        Either way each exchange is bounded by [agent].timeout_s. Raises AgentError
        naming an auth_token_env variable that is unset or blank. A [bridge]
        responder factory's own errors propagate.
        """
        agent = config.agent
        replay, timeout_s = agent.history == "replay", agent.timeout_s
        if agent.url is None:
            app = app_from_config(config)
            return cls.in_process(app, replay=replay, timeout_s=timeout_s)
        headers = _bearer_headers(agent.auth_token_env)
        http = httpx.AsyncClient(headers=headers, timeout=timeout_s)
        return cls(http, agent.url, replay=replay, timeout_s=timeout_s)

    @classmethod
    def in_process(
        cls, app: Starlette, *, replay: bool = False, timeout_s: float | None = None
    ) -> Self:
        """Reach a bridge app in this process, with no sockets.

        timeout_s bounds each exchange, as for __init__; None waits for the app.
        """
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
        return cls(http, settings.bridge.url, replay=replay, timeout_s=timeout_s)

    async def __aenter__(self) -> Self:
        """Return this agent; leaving the block closes it."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the agent."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the httpx client, which the A2A client sends through."""
        await self._http.aclose()

    async def card(self) -> AgentCard:
        """Return the agent card, fetched and checked once, as before the first send.

        Raises AgentError, as send does, when the fetch fails or times out, the
        card is unusable, or it names an origin other than the url's while a
        bearer token is set.
        """
        async with _bounded(self._timeout_s):
            _, card = await self._connect()
        return card

    async def send(
        self,
        text: str,
        *,
        context_id: str | None = None,
        history: Sequence[ChatCompletionMessage] = (),
        prompt_overrides: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> AgentTurn:
        """Send text as one A2A message and return the reply, as Agent.send says.

        Raises AgentError, with the cause's message, when the A2A call or its HTTP
        transport fails or times out; with the state and reply text when the task
        ends other than completed or awaiting input; when the reply holds a
        malformed trajectory artifact; and before the first message, when the
        agent card is unusable or names an origin other than the url's while a
        bearer token is set.
        """
        context = dict(context or {})
        reserved = sorted(RESERVED & context.keys())
        if reserved:
            raise AgentError(settings.texts.reserved_context.format(keys=reserved))
        if context_id is None and history and self._replay:
            context_id = await self._replay_history(history, prompt_overrides, context)
            history = ()
        metadata = _metadata(context_id, history, prompt_overrides, context)
        reply = await self._exchange(text, context_id, metadata)
        # Track the open task before reading the reply can raise, so a failed
        # continuation never leaves a finished task id for the next send.
        if isinstance(reply, Task) and reply.status.state == _INPUT_REQUIRED:
            self._input_required[reply.context_id] = reply.id
        else:
            self._input_required.pop(reply.context_id, None)
        turn, applied = _read_reply(reply)
        _require_applied(RESERVED & metadata.keys(), applied)
        return turn

    async def _replay_history(
        self,
        history: Sequence[ChatCompletionMessage],
        prompt_overrides: Mapping[str, str] | None,
        context: Mapping[str, Any],
    ) -> str | None:
        """Send history's user messages on one new context and return its id."""
        context_id = None
        for message in history:
            if message["role"] == "user":
                turn = await self.send(
                    _content_text(message),
                    context_id=context_id,
                    prompt_overrides=prompt_overrides,
                    context=context,
                )
                context_id = turn.context_id
        return context_id

    async def _exchange(
        self, text: str, context_id: str | None, metadata: dict[str, Any]
    ) -> Task | Message:
        """Send one user message, continuing an open task, and return the reply."""
        task_id = self._input_required.get(context_id, "") if context_id else ""
        message = Message(
            role=Role.ROLE_USER,
            message_id=uuid4().hex,
            parts=[Part(text=text)],
            context_id=context_id or "",
            task_id=task_id,
            metadata=metadata,
        )
        async with _bounded(self._timeout_s):
            client, _ = await self._connect()
            request = SendMessageRequest(message=message)
            (response,) = [event async for event in client.send_message(request)]
        return response.task if response.HasField("task") else response.message

    async def _connect(self) -> tuple[Client, AgentCard]:
        """Return the A2A client and the card it was made for, made on first use."""
        async with self._connection_lock:
            if self._connection is None:
                self._connection = await self._new_connection()
            return self._connection

    async def _new_connection(self) -> tuple[Client, AgentCard]:
        """Fetch and check the agent card, then return a client for it and the card."""
        try:
            card = await A2ACardResolver(self._http, self._url).get_agent_card()
            if AUTHORIZATION in self._http.headers:
                _require_same_origin(self._url, card)
            config = ClientConfig(streaming=False, httpx_client=self._http)
            return await create_client(card, client_config=config), card
        except (ValueError, ParseError) as error:
            raise AgentError(str(error)) from error


@asynccontextmanager
async def _bounded(timeout_s: float | None) -> AsyncIterator[None]:
    """Run the block within timeout_s; an A2A, HTTP or timeout failure is AgentError."""
    deadline = asyncio.timeout(timeout_s)
    timed_out = settings.texts.timed_out.format(timeout_s=timeout_s)
    try:
        async with deadline:
            yield
    except (A2AError, httpx.HTTPError) as error:
        raise AgentError(str(error)) from error
    except TimeoutError as error:
        raise AgentError(timed_out) from error
    # An in-process app can swallow the cancellation and answer anyway.
    if deadline.expired():
        raise AgentError(timed_out)


def _require_same_origin(url: str, card: AgentCard) -> None:
    """Raise AgentError if the card sends messages to an origin other than url's."""
    configured = _origin(url)
    for interface in card.supported_interfaces:
        named = _origin(interface.url)
        if named != configured:
            message = settings.texts.other_origin
            raise AgentError(message.format(configured=configured, named=named))


def _origin(url: str) -> str:
    """Return url's scheme://host[:port], with a default port left out."""
    parsed = httpx.URL(url)
    return f"{parsed.scheme}://{parsed.netloc.decode()}"


def _metadata(
    context_id: str | None,
    history: Sequence[ChatCompletionMessage],
    prompt_overrides: Mapping[str, str] | None,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the message metadata: context, overrides, and a new context's history."""
    metadata = dict(context)
    if prompt_overrides:
        metadata[PROMPT_OVERRIDES] = dict(prompt_overrides)
    if context_id is None and history:
        metadata[HISTORY] = list(history)
    return metadata


def _read_reply(reply: Task | Message) -> tuple[AgentTurn, set[str]]:
    """Return the turn a reply holds and the request keys it says it applied.

    The text is a Message reply's, else the text of the task's answer artifacts
    (any but the trajectory), else its status message's.
    """
    if isinstance(reply, Message):
        status, artifacts = reply, []
    else:
        status, artifacts = reply.status.message, list(reply.artifacts)
        if reply.status.state not in _ANSWERED:
            state = TaskState.Name(reply.status.state)
            message = settings.texts.task_not_completed
            raise AgentError(message.format(state=state, text=get_message_text(status)))
    answers = [
        text
        for artifact in artifacts
        if artifact.name != TRAJECTORY
        for text in get_text_parts(artifact.parts)
    ]
    text = "\n".join(answers) or get_message_text(status)
    trajectory = _trajectory(artifacts)
    fallback: ChatCompletionMessage = {"role": "assistant", "content": text}
    turn = AgentTurn(
        text=text,
        messages=(fallback,) if trajectory is None else tuple(trajectory),
        context_id=reply.context_id,
        trajectory_reported=trajectory is not None,
    )
    return turn, set(MessageToDict(status.metadata).get(APPLIED, []))


def _trajectory(artifacts: Sequence[Artifact]) -> list[ChatCompletionMessage] | None:
    """Return the last trajectory artifact's messages, or None if there is none."""
    # A continued task keeps every turn's artifacts, so the last is this turn's.
    data = [
        value
        for artifact in artifacts
        if artifact.name == TRAJECTORY
        for value in get_data_parts(artifact.parts)
    ]
    if not data:
        return None
    messages = data[-1].get(MESSAGES) if isinstance(data[-1], dict) else None
    if not isinstance(messages, list) or not all(
        isinstance(message, dict) for message in messages
    ):
        raise AgentError(settings.texts.bad_trajectory)
    return messages


def _require_applied(sent: Set[str], applied: Set[str]) -> None:
    """Raise AgentError if the agent did not apply a history or overrides it got."""
    missing = sent - applied
    if HISTORY in missing:
        raise AgentError(settings.texts.history_not_applied)
    if missing:
        raise AgentError(settings.texts.overrides_not_applied)


def _content_text(message: ChatCompletionMessage) -> str:
    """Return a message's text, joining the text blocks of a list content."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content)


__all__ = ["A2AAgent"]
