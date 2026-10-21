"""Typed shapes that agents, datasets, targets and evaluators exchange."""

from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

from openevals.types import (
    ChatCompletionMessage,
    EvaluatorResult,
    SimpleAsyncEvaluator,
    SimpleEvaluator,
)
from openevals.types import ToolArgsMatchMode as ToolArgsMatch
from pydantic import ConfigDict, with_config

TrajectoryMatch = Literal["strict", "unordered", "subset", "superset"]


class AgentError(Exception):
    """The agent failed, or ignored the history or prompt overrides it was sent."""


class ToolMessage(ChatCompletionMessage):
    """A tool result in messages: role "tool", and the id of the call it answers."""

    tool_call_id: str


@dataclass(frozen=True)
class AgentTurn:
    """The agent's reply to one user message.

    text is the final reply text. messages are this turn's agent messages in OpenAI
    format and in order: assistant messages with tool_calls (each
    function.arguments a JSON string), the tool results as ToolMessage, then the
    final assistant message; never the user message. context_id names the
    conversation; send it back to continue that conversation. trajectory_reported
    is False when the agent sent no trajectory artifact: messages is then just
    ({"role": "assistant", "content": text},), and trajectory graders (tools,
    trajectory) return unscored rather than 0.
    """

    text: str
    messages: tuple[ChatCompletionMessage, ...]
    context_id: str
    trajectory_reported: bool = True


class AgentInputs(TypedDict):
    """A target's inputs: the conversation to send, ending with the user message.

    messages are OpenAI-format; the last is the user message to send and any before
    it are the history. context is metadata sent with the message, such as the
    channel.
    """

    messages: list[ChatCompletionMessage]
    context: dict[str, Any]


class AgentOutputs(TypedDict):
    """A target's outputs: the agent's last reply and the full transcript.

    text is the final reply text of the last turn. messages is the whole
    conversation in order: each user message (history included), followed by the
    AgentTurn.messages the agent replied with. trajectory_reported is False when
    the agent sent no trajectory artifact on some turn, and a missing key means
    True; when False, trajectory graders (tools, trajectory) return unscored
    rather than 0.
    """

    text: str
    messages: list[ChatCompletionMessage]
    trajectory_reported: NotRequired[bool]


class DatasetExample(TypedDict):
    """One labelled case ready to mirror into a LangSmith dataset."""

    case_id: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    metadata: dict[str, Any]


@with_config(ConfigDict(extra="allow"))
class Expectations(TypedDict, total=False):
    """What a case expects of the agent; each key present adds one check.

    tools, trajectory, contains and not_contains grade the graded span: for a
    golden case, the messages from the last input user message onward, history
    excluded (outputs.messages[len(inputs.messages) - 1:]); for a simulation, the
    whole conversation.
    output: AgentOutputs.text, the final reply, equals it, ignoring surrounding
    whitespace.
    contains: substrings the concatenated content of every assistant message in
    the span must contain, ignoring case.
    not_contains: substrings that content must not contain, ignoring case, so a
    forbidden claim in any reply fails.
    tools: tool names the agent must call at least once each, in any order, as
    assistant tool_calls in the span; other calls are allowed.
    trajectory: reference messages that openevals' trajectory match compares with
    the span.
    trajectory_match: that comparison's trajectory mode; "strict" when absent.
    tool_args_match: that comparison's tool-arguments mode; "exact" when absent.
    criteria: rubric lines one judge grades over the full transcript as a single
    pass or fail, passing only when every line holds. An absent key is not graded;
    validation keeps unknown keys, such as a suite's own label.
    """

    output: str
    contains: list[str]
    not_contains: list[str]
    tools: list[str]
    trajectory: list[ChatCompletionMessage]
    trajectory_match: TrajectoryMatch
    tool_args_match: ToolArgsMatch
    criteria: list[str]


__all__ = [
    "AgentError",
    "AgentInputs",
    "AgentOutputs",
    "AgentTurn",
    "ChatCompletionMessage",
    "DatasetExample",
    "EvaluatorResult",
    "Expectations",
    "SimpleAsyncEvaluator",
    "SimpleEvaluator",
    "ToolArgsMatch",
    "ToolMessage",
    "TrajectoryMatch",
]
