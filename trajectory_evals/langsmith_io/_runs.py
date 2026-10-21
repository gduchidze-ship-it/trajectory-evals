"""A project's root runs, and the conversation each one's inputs and outputs hold."""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from langchain_core.messages import convert_to_openai_messages
from langsmith.types import Run, RunSelectField

from trajectory_evals.core.settings import settings as core_settings
from trajectory_evals.core.types import ChatCompletionMessage
from trajectory_evals.golden.cases import GoldenCase
from trajectory_evals.golden.graders import _text
from trajectory_evals.golden.settings import settings as golden_settings
from trajectory_evals.langsmith_io.protocols import LangSmithClient
from trajectory_evals.langsmith_io.settings import settings

# What convert_to_openai_messages raises for a value that is not messages.
_NOT_MESSAGES = (ValueError, TypeError, NotImplementedError)


@dataclass(frozen=True)
class Conversation:
    """A root run's messages through the last user message, and the agent's reply.

    messages end with a user message with text; reply holds the messages the
    run's outputs have after their last user message, all of them if none.
    """

    messages: tuple[ChatCompletionMessage, ...]
    reply: tuple[ChatCompletionMessage, ...]

    @property
    def text(self) -> str:
        """Return the text of the reply's last assistant message, "" if none."""
        assistant = golden_settings.data.assistant_role
        replies = [message for message in self.reply if message["role"] == assistant]
        return _text(replies[-1]["content"]) if replies else ""


def check_limit(limit: int) -> None:
    """Raise ValueError for a run limit below 1, before anything is called."""
    if limit < 1:
        message = core_settings.texts.at_least_one
        raise ValueError(message.format(name="limit", value=limit))


async def query_root_runs(
    client: LangSmithClient,
    project: str,
    *,
    since: datetime,
    limit: int,
    selects: Sequence[RunSelectField],
    **conditions: Any,
) -> AsyncIterator[Run]:
    """Yield up to limit root runs of the named project started since since.

    limit must already have passed check_limit. conditions are further
    runs.query arguments, such as filter or has_error. Raises
    LangSmithNotFoundError for an unknown project, and a langsmith.APIError
    subclass when the query fails.
    """
    project_id = str((await client.aread_project(project_name=project)).id)
    runs = client.runs.query(
        project_ids=[project_id],
        is_root=True,
        min_start_time=since,
        selects=list(selects),
        page_size=min(limit, settings.defaults.page_size),
        **conditions,
    )
    read = 0
    async for run in runs:
        yield run
        read += 1
        # Stop before asking for the item after the last, which may fetch a page.
        if read == limit:
            return


def read_conversation(
    run: Run, *, inputs_key: str, outputs_key: str
) -> Conversation | None:
    """Return the run's conversation, or None if inputs_key holds none.

    The messages at the dotted paths may be OpenAI dicts or LangChain messages,
    serialized or dumped; both are normalized to OpenAI format. A conversation
    must be a valid golden one, ending with a user message with text.
    """
    try:
        case = GoldenCase.model_validate(
            {"id": run.id, "messages": _messages(run.inputs, inputs_key)}
        )
    except _NOT_MESSAGES:
        return None
    return Conversation(
        messages=tuple(case.messages or ()), reply=_reply(run.outputs, outputs_key)
    )


def _messages(data: Mapping[str, Any] | None, path: str) -> list[ChatCompletionMessage]:
    """Return the list at a dotted path of data as OpenAI messages; [] if no list."""
    value: Any = data
    for key in path.split("."):
        value = value.get(key) if isinstance(value, Mapping) else None
    if not isinstance(value, list):
        return []
    # A list in gives a list of OpenAI message dicts out.
    return cast(list[ChatCompletionMessage], convert_to_openai_messages(value))


def _reply(
    outputs: Mapping[str, Any] | None, path: str
) -> tuple[ChatCompletionMessage, ...]:
    """Return the messages after the last user message at path, () if unreadable."""
    try:
        messages = _messages(outputs, path)
    except _NOT_MESSAGES:
        return ()
    user = golden_settings.data.user_role
    users = [index for index, message in enumerate(messages) if message["role"] == user]
    return tuple(messages[users[-1] + 1 :] if users else messages)


__all__ = ["Conversation", "check_limit", "query_root_runs", "read_conversation"]
