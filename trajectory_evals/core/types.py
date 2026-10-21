"""Typed shapes that agents, datasets, targets and evaluators exchange."""

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from openevals.types import (
    ChatCompletionMessage,
    EvaluatorResult,
    SimpleAsyncEvaluator,
    SimpleEvaluator,
)

TrajectoryMatch = Literal["strict", "unordered", "subset", "superset"]
ToolArgsMatch = Literal["exact", "ignore", "subset", "superset"]


class AgentError(Exception):
    """The agent failed, or ignored the history or prompt overrides it was sent."""


@dataclass(frozen=True)
class AgentTurn:
    """The agent's reply to one user message.

    text is the final reply text. messages are this turn's agent messages in OpenAI
    format and in order: assistant messages with tool_calls (each
    function.arguments a JSON string), the tool results, then the final assistant
    message; never the user message. An agent that reports no tool calls gives
    just ({"role": "assistant", "content": text},). context_id names the
    conversation; send it back to continue that conversation.
    """

    text: str
    messages: tuple[ChatCompletionMessage, ...]
    context_id: str


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
    AgentTurn.messages the agent replied with.
    """

    text: str
    messages: list[ChatCompletionMessage]


class DatasetExample(TypedDict):
    """One labelled case ready to mirror into a LangSmith dataset."""

    case_id: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    metadata: dict[str, Any]


class Expectations(TypedDict, total=False):
    """What a case expects of the agent; each key present adds one check.

    output: AgentOutputs.text equals it, ignoring surrounding whitespace.
    contains: substrings AgentOutputs.text must contain, ignoring case.
    not_contains: substrings AgentOutputs.text must not contain, ignoring case.
    tools: tool names the agent must call at least once each, in any order, as
    assistant tool_calls in AgentOutputs.messages; other calls are allowed.
    trajectory: reference messages that openevals' trajectory match compares with
    the whole AgentOutputs.messages, user messages included.
    trajectory_match: that comparison's trajectory mode; "strict" when absent.
    tool_args_match: that comparison's tool-arguments mode; "exact" when absent.
    criteria: rubric lines one judge grades over the full transcript as a single
    pass or fail, passing only when every line holds. An absent key is not graded.
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
    "TrajectoryMatch",
]
