"""A2AAgent: the Agent for an agent reached over A2A, remote or bridged in process."""

import asyncio
from collections.abc import Mapping, Sequence, Set
from typing import Any, Self
from uuid import uuid4

import httpx
from a2a.client import Client, ClientConfig, create_client
from a2a.helpers import get_data_parts, get_message_text
from a2a.types import Message, Part, Role, SendMessageRequest, Task, TaskState
from a2a.utils.errors import A2AError

# protobuf ships no stubs and pyproject has no google.protobuf override.
from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]
from starlette.applications import Starlette

from trajectory_evals.connect.bridge import app_from_config
from trajectory_evals.connect.settings import _bearer_headers, settings
from trajectory_evals.core.agent_config import AgentConfig
from trajectory_evals.core.types import AgentError, AgentTurn, ChatCompletionMessage

_INPUT_REQUIRED = TaskState.TASK_STATE_INPUT_REQUIRED
_ANSWERED = (TaskState.TASK_STATE_COMPLETED, _INPUT_REQUIRED)


class A2AAgent:
    """An Agent that sends each message to an A2A agent over JSON-RPC.

    It owns an httpx client: close it with aclose or an async with block.
    """

    def __init__(
        self, http: httpx.AsyncClient, url: str, *, replay: bool = False
    ) -> None:
        """Reach the agent at url through http, which aclose closes.

        replay sends a new conversation's history as earlier user turns on the same
        context, for an agent that cannot take history in message metadata.
        """
        self._http = http
        self._url = url
        self._replay = replay
        self._client: Client | None = None
        self._client_lock = asyncio.Lock()
        self._input_required: dict[str, str] = {}  # context id -> its open task id

    @classmethod
    def from_config(cls, config: AgentConfig) -> Self:
        """Reach [agent].url with its bearer token, or serve [bridge] in process.

        Raises AgentError naming an auth_token_env variable that is unset. A
        [bridge] responder factory's own errors propagate.
        """
        agent = config.agent
        replay = agent.history == "replay"
        if agent.url is None:
            return cls.in_process(app_from_config(config), replay=replay)
        headers = _bearer_headers(agent.auth_token_env)
        http = httpx.AsyncClient(headers=headers, timeout=agent.timeout_s)
        return cls(http, agent.url, replay=replay)

    @classmethod
    def in_process(cls, app: Starlette, *, replay: bool = False) -> Self:
        """Reach a bridge app in this process, with no sockets."""
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
        return cls(http, settings.bridge.url, replay=replay)

    async def __aenter__(self) -> Self:
        """Return this agent; leaving the block closes it."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the agent."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the httpx client, which the A2A client sends through."""
        await self._http.aclose()

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
        transport fails, and with the state and reply text when the task ends
        other than completed or awaiting input.
        """
        context = dict(context or {})
        reserved = sorted(settings.wire.reserved & context.keys())
        if reserved:
            raise AgentError(settings.texts.reserved_context.format(keys=reserved))
        if context_id is None and history and self._replay:
            context_id = await self._replay_history(history, prompt_overrides, context)
            history = ()
        metadata = _metadata(context_id, history, prompt_overrides, context)
        reply = await self._exchange(text, context_id, metadata)
        turn, applied = _read_reply(reply)
        _require_applied(settings.wire.reserved & metadata.keys(), applied)
        if isinstance(reply, Task) and reply.status.state == _INPUT_REQUIRED:
            self._input_required[turn.context_id] = reply.id
        else:
            self._input_required.pop(turn.context_id, None)
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
        try:
            client = await self._a2a_client()
            request = SendMessageRequest(message=message)
            (response,) = [event async for event in client.send_message(request)]
        except (A2AError, httpx.HTTPError) as error:
            raise AgentError(str(error)) from error
        return response.task if response.HasField("task") else response.message

    async def _a2a_client(self) -> Client:
        """Return the A2A client, resolving the agent card on first use."""
        async with self._client_lock:
            if self._client is None:
                config = ClientConfig(streaming=False, httpx_client=self._http)
                self._client = await create_client(self._url, client_config=config)
            return self._client


def _metadata(
    context_id: str | None,
    history: Sequence[ChatCompletionMessage],
    prompt_overrides: Mapping[str, str] | None,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the message metadata: context, overrides, and a new context's history."""
    wire = settings.wire
    metadata = dict(context)
    if prompt_overrides:
        metadata[wire.prompt_overrides] = dict(prompt_overrides)
    if context_id is None and history:
        metadata[wire.history] = list(history)
    return metadata


def _read_reply(reply: Task | Message) -> tuple[AgentTurn, set[str]]:
    """Return the turn a reply holds and the request keys it says it applied."""
    wire = settings.wire
    if isinstance(reply, Message):
        status, artifacts = reply, []
    else:
        status, artifacts = reply.status.message, list(reply.artifacts)
        if reply.status.state not in _ANSWERED:
            state = TaskState.Name(reply.status.state)
            message = settings.texts.task_not_completed
            raise AgentError(message.format(state=state, text=get_message_text(status)))
    text = get_message_text(status)
    trajectories = [
        data[wire.messages]
        for artifact in artifacts
        if artifact.name == wire.trajectory
        for data in get_data_parts(artifact.parts)
    ]
    fallback: ChatCompletionMessage = {"role": "assistant", "content": text}
    turn = AgentTurn(
        text=text,
        messages=tuple(trajectories[0]) if trajectories else (fallback,),
        context_id=reply.context_id,
        trajectory_reported=bool(trajectories),
    )
    return turn, set(MessageToDict(status.metadata).get(wire.applied, []))


def _require_applied(sent: Set[str], applied: Set[str]) -> None:
    """Raise AgentError if the agent did not apply a history or overrides it got."""
    missing = sent - applied
    if settings.wire.history in missing:
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
