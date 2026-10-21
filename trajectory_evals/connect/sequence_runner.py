"""A Responder that answers each turn with agent-platform-sequence-runner /execute."""

import json
from collections.abc import Mapping
from typing import Any, Self

import httpx
from pydantic import Field, PositiveFloat, model_validator

from trajectory_evals.connect.bridge import AgentReply, AgentRequest, Responder
from trajectory_evals.connect.settings import _bearer_headers, _require_env, settings
from trajectory_evals.core.settings import Text, _Section
from trajectory_evals.core.types import ChatCompletionMessage, ToolMessage


class SequenceRunnerOptions(_Section):
    """[bridge].options for the sequence-runner: where it runs and what to run.

    Each turn's initial_state is initial_state, then the request's context keys,
    then messages_key holding the history and the new user message in OpenAI
    format. reply_path and tool_calls_path are dotted paths into the final state:
    the reply text, and a step's _tool_calls record, which may be absent.
    """

    url: Text | None = None
    url_env: Text | None = None
    token_env: Text | None = None
    sequence_id: Text
    client_key: Text
    product_id: Text
    initial_state: dict[str, Any] = Field(default_factory=dict)
    messages_key: Text
    reply_path: Text
    tool_calls_path: Text | None = None
    timeout_s: PositiveFloat = 120

    @model_validator(mode="after")
    def _check_one_url(self) -> Self:
        """Reject options that give both or neither of url and url_env."""
        if (self.url is None) == (self.url_env is None):
            raise ValueError(settings.texts.one_url)
        return self


def factory(
    options: Mapping[str, Any], *, transport: httpx.AsyncBaseTransport | None = None
) -> Responder:
    """Return a Responder that runs the options' sequence once per turn.

    The runner takes no prompt overrides, so the reply never acknowledges them.
    Reads the url and token variables now; AgentError names one that is unset.
    """
    parsed = SequenceRunnerOptions.model_validate(options)
    url = parsed.url
    if url is None:
        assert parsed.url_env is not None  # _check_one_url requires one of them
        url = _require_env(parsed.url_env)
    headers = _bearer_headers(parsed.token_env)

    async def respond(request: AgentRequest) -> AgentReply:
        """Run the sequence on the conversation; return its reply and tool calls."""
        # shortcut: one httpx client per turn, so no connection reuse; hold one
        # client per responder if turn latency ever matters.
        async with httpx.AsyncClient(
            base_url=url, headers=headers, timeout=parsed.timeout_s, transport=transport
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
    """Return the reply text at reply_path, after the tool calls it recorded."""
    text = _at(state, options.reply_path)
    if not isinstance(text, str):
        raise ValueError(settings.texts.no_reply.format(path=options.reply_path))
    calls = _at(state, options.tool_calls_path) if options.tool_calls_path else None
    final: ChatCompletionMessage = {"role": "assistant", "content": text}
    return AgentReply(text=text, messages=(*_tool_messages(calls or {}), final))


def _tool_messages(
    calls: Mapping[str, list[Mapping[str, Any]]],
) -> list[ChatCompletionMessage]:
    """Return an assistant tool call and its tool result for each recorded call."""
    # _tool_calls groups calls by tool name, so the order across tools is lost.
    records = [(name, record) for name, records in calls.items() for record in records]
    messages: list[ChatCompletionMessage] = []
    for index, (name, record) in enumerate(records):
        call_id = settings.sequence_runner.tool_call_id.format(index=index)
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
