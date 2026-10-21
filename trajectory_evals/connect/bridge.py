"""An A2A server that wraps a non-A2A agent, given as a Responder."""

import importlib
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from operator import attrgetter
from typing import Any, cast, override

from a2a.helpers import get_message_text, new_data_part, new_task
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    Message,
    Part,
    TaskState,
)
from a2a.utils.constants import (
    DEFAULT_RPC_URL,
    PROTOCOL_VERSION_CURRENT,
    TransportProtocol,
)
from a2a.utils.errors import UnsupportedOperationError

# protobuf ships no stubs and pyproject has no google.protobuf override.
from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]
from pydantic import TypeAdapter
from starlette.applications import Starlette

from trajectory_evals.connect.settings import settings
from trajectory_evals.core.agent_config import AgentConfig, AgentConfigError
from trajectory_evals.core.types import ChatCompletionMessage

logger = logging.getLogger(__name__)

# Metadata is free-form on the wire, so check its shape before a responder sees it.
_HISTORY = TypeAdapter(list[dict[str, Any]])
_PROMPT_OVERRIDES = TypeAdapter(dict[str, str])


@dataclass(frozen=True)
class AgentRequest:
    """One user message for the wrapped agent, with what it needs to answer it.

    history is the conversation before message in OpenAI format: what the client
    seeded the context with, then every earlier turn's user message and reply
    messages. prompt_overrides maps a prompt name to the text to use in its place.
    context is the rest of the message metadata, such as the channel.
    """

    message: str
    history: tuple[ChatCompletionMessage, ...]
    context_id: str
    prompt_overrides: Mapping[str, str]
    context: Mapping[str, Any]


@dataclass(frozen=True)
class AgentReply:
    """The wrapped agent's answer to one AgentRequest.

    messages are this turn's agent messages in OpenAI format, ending with the
    final assistant message; empty means just an assistant message with text.
    prompt_overrides_applied says the agent used the request's prompt_overrides.
    """

    text: str
    messages: tuple[ChatCompletionMessage, ...] = ()
    prompt_overrides_applied: bool = False


type Responder = Callable[[AgentRequest], Awaitable[AgentReply]]


class _ResponderExecutor(AgentExecutor):
    """Answers each message with a Responder and keeps each context's transcript."""

    def __init__(self, responder: Responder) -> None:
        """Hold the responder; no context has a transcript yet."""
        self._responder = responder
        # shortcut: transcripts live as long as the app; evict idle contexts if a
        # long-running serve holds too many.
        self._transcripts: dict[str, tuple[ChatCompletionMessage, ...]] = {}

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Answer one message with its trajectory artifact and what was applied."""
        # The request handler fills all three for every message/send.
        assert context.task_id and context.context_id and context.message
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if context.current_task is None:
            submitted = TaskState.TASK_STATE_SUBMITTED
            task = new_task(
                context.task_id,
                context.context_id,
                submitted,
                history=[context.message],
            )
            await event_queue.enqueue_event(task)
        try:
            request, applied = self._request(context.context_id, context.message)
            reply = await self._responder(request)
        except Exception:
            logger.exception(settings.texts.responder_failed_log, context.context_id)
            failure = Part(text=settings.texts.responder_failed)
            await updater.failed(updater.new_agent_message([failure]))
            return
        await self._report(updater, request, reply, applied)

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Refuse: a responder runs to completion."""
        raise UnsupportedOperationError()

    def _request(
        self, context_id: str, message: Message
    ) -> tuple[AgentRequest, list[str]]:
        """Read message; history seeds only a new context, and is then applied."""
        wire = settings.wire
        metadata = MessageToDict(message.metadata)
        history = metadata.pop(wire.history, None)
        overrides = metadata.pop(wire.prompt_overrides, {})
        transcript = self._transcripts.get(context_id)
        applied: list[str] = []
        if transcript is None and history is not None:
            seeded = _HISTORY.validate_python(history)
            transcript = tuple(cast(list[ChatCompletionMessage], seeded))
            applied.append(wire.history)
        request = AgentRequest(
            message=get_message_text(message),
            history=transcript or (),
            context_id=context_id,
            prompt_overrides=_PROMPT_OVERRIDES.validate_python(overrides),
            context=metadata,
        )
        return request, applied

    async def _report(
        self,
        updater: TaskUpdater,
        request: AgentRequest,
        reply: AgentReply,
        applied: list[str],
    ) -> None:
        """Extend the transcript, then send the trajectory and complete the task."""
        wire = settings.wire
        user: ChatCompletionMessage = {"role": "user", "content": request.message}
        final: ChatCompletionMessage = {"role": "assistant", "content": reply.text}
        messages = reply.messages or (final,)
        self._transcripts[request.context_id] = (*request.history, user, *messages)
        if reply.prompt_overrides_applied:
            applied.append(wire.prompt_overrides)
        trajectory = new_data_part({wire.messages: list(messages)})
        await updater.add_artifact([trajectory], name=wire.trajectory)
        text = Part(text=reply.text)
        message = updater.new_agent_message([text], metadata={wire.applied: applied})
        await updater.complete(message)


def agent_card(config: AgentConfig, url: str) -> AgentCard:
    """Return the card for config's agent served at url, an absolute root url."""
    bridge = settings.bridge
    interface = AgentInterface(
        protocol_binding=TransportProtocol.JSONRPC.value,
        url=url,
        protocol_version=PROTOCOL_VERSION_CURRENT,
    )
    return AgentCard(
        name=config.name,
        description=config.agent.description,
        version=bridge.version,
        default_input_modes=[bridge.media_type],
        default_output_modes=[bridge.media_type],
        capabilities=AgentCapabilities(streaming=False),
        supported_interfaces=[interface],
    )


def build_app(responder: Responder, card: AgentCard) -> Starlette:
    """Return an A2A app serving card and answering each message with responder.

    A responder that raises fails that task with a generic message; the bridge
    logs the exception and never sends its text to the client.
    """
    handler = DefaultRequestHandler(
        agent_executor=_ResponderExecutor(responder),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = create_jsonrpc_routes(handler, DEFAULT_RPC_URL)
    return Starlette(routes=[*create_agent_card_routes(card), *routes])


def app_from_config(config: AgentConfig, url: str = settings.bridge.url) -> Starlette:
    """Return the app for config's [bridge] responder factory, served at url.

    Raises AgentConfigError when config has no [bridge] table; the factory's own
    errors, such as invalid options, propagate.
    """
    bridge = config.bridge
    if bridge is None:
        raise AgentConfigError(settings.texts.no_bridge.format(name=config.name))
    module, attr = bridge.responder.split(":")
    factory = attrgetter(attr)(importlib.import_module(module))
    return build_app(factory(bridge.options), agent_card(config, url))


__all__ = [
    "AgentReply",
    "AgentRequest",
    "Responder",
    "agent_card",
    "app_from_config",
    "build_app",
]
