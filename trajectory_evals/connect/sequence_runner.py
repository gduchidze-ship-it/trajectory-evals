"""A Responder that answers each turn with agent-platform-sequence-runner /execute."""

import json
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import httpx
from pydantic import Field, PositiveFloat

from trajectory_evals.connect.bridge import AgentReply, AgentRequest, Responder
from trajectory_evals.connect.settings import _bearer_headers, settings
from trajectory_evals.core.agent_config import AgentConfig
from trajectory_evals.core.settings import Text, _Section
from trajectory_evals.core.types import ChatCompletionMessage, ToolMessage

_TURN_ID_LENGTH = 8

type _Call = tuple[str, Mapping[str, Any]]


class SequenceRunnerOptions(_Section):
    """[bridge].options for the sequence-runner: where it runs and what to run.

    Each turn's initial_state is initial_state, then the request's context keys,
    then messages_key holding the history and the new user message in OpenAI
    format. reply_path is a dotted path to the reply text in the final state.
    tool_calls_path lists dotted paths to _tool_calls records, any of them
    absent; the runner hoists a step's calls to <step_id>_tool_calls, so a
    multi-step sequence needs one path per step.
    """

    url: Text
    token_env: Text | None = None
    sequence_id: Text
    client_key: Text
    product_id: Text
    initial_state: dict[str, Any] = Field(default_factory=dict)
    messages_key: Text
    reply_path: Text
    tool_calls_path: list[Text] = Field(default_factory=list)
    timeout_s: PositiveFloat = 120


def factory(
    options: Mapping[str, Any],
    _config: AgentConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Responder:
    """Return a Responder that runs the options' sequence once per turn.

    The runner takes no prompt overrides, so a turn that brings any is answered
    with empty text and never acknowledges them, without running the sequence.
    Reads the token variable now; AgentError names it when unset or blank, and
    invalid options raise pydantic's ValidationError. A turn raises
    httpx.HTTPError when the request fails and ValueError when the final state
    has no reply text; the bridge reports both as its generic failure.
    """
    parsed = SequenceRunnerOptions.model_validate(options)
    headers = _bearer_headers(parsed.token_env)

    async def respond(request: AgentRequest) -> AgentReply:
        """Run the sequence on the conversation; return its reply and tool calls."""
        # Refuse before running: the client then raises on the missing
        # acknowledgement, and the sequence never acts on a refused turn.
        if request.prompt_overrides:
            return AgentReply(text="")
        # shortcut: one httpx client per turn, so no connection reuse; hold one
        # client per responder if turn latency ever matters.
        async with httpx.AsyncClient(
            base_url=parsed.url,
            headers=headers,
            timeout=parsed.timeout_s,
            transport=transport,
        ) as http:
            response = await http.post(
                settings.sequence_runner.execute_path, json=_payload(parsed, request)
            )
        response.raise_for_status()
        return _reply(parsed, response.json())

    return respond


def _payload(options: SequenceRunnerOptions, request: AgentRequest) -> dict[str, Any]:
    """Return the /execute body for one turn of the conversation."""
    user: ChatCompletionMessage = {"role": "user", "content": request.message}
    initial_state = {
        **options.initial_state,
        **request.context,
        options.messages_key: [*request.history, user],
    }
    return {
        "sequence_id": options.sequence_id,
        "client_key": options.client_key,
        "product_id": options.product_id,
        "initial_state": initial_state,
    }


def _reply(options: SequenceRunnerOptions, state: Any) -> AgentReply:
    """Return the reply text at reply_path, after the tool calls recorded."""
    text = _at(state, options.reply_path)
    if not isinstance(text, str):
        raise ValueError(settings.texts.no_reply.format(path=options.reply_path))
    calls = [
        (name, record)
        for path in options.tool_calls_path
        for name, records in (_at(state, path) or {}).items()
        for record in records
    ]
    final: ChatCompletionMessage = {"role": "assistant", "content": text}
    return AgentReply(text=text, messages=(*_tool_messages(calls), final))


def _tool_messages(calls: list[_Call]) -> list[ChatCompletionMessage]:
    """Return an assistant tool call and its result per call, ids unique per turn."""
    # _tool_calls groups calls by tool name, so the order across tools is lost.
    turn = uuid4().hex[:_TURN_ID_LENGTH]
    messages: list[ChatCompletionMessage] = []
    for index, (name, record) in enumerate(calls):
        call_id = settings.sequence_runner.tool_call_id.format(turn=turn, index=index)
        arguments = json.dumps(record["arguments"])
        function = {"name": name, "arguments": arguments}
        call = {"id": call_id, "type": "function", "function": function}
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        result: ToolMessage = {
            "role": "tool",
            "content": _result_text(record),
            "tool_call_id": call_id,
        }
        messages.append(result)
    return messages


def _result_text(record: Mapping[str, Any]) -> str:
    """Return a recorded call's response, or its error, as tool message content."""
    error = record.get("error")
    result = record.get("response") if error is None else {"error": error}
    return result if isinstance(result, str) else json.dumps(result)


def _at(state: Any, path: str) -> Any:
    """Return the value at a dotted path through nested objects, or None."""
    for key in path.split("."):
        if not isinstance(state, dict):
            return None
        state = state.get(key)
    return state


__all__ = ["SequenceRunnerOptions", "factory"]
