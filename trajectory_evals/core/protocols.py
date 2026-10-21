"""Structural interfaces that the harness takes as collaborators."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Protocol
from uuid import UUID

from trajectory_evals.core.types import AgentTurn, ChatCompletionMessage, DatasetExample


class Agent(Protocol):
    """A conversational agent under evaluation, sent one user message at a time."""

    async def send(
        self,
        text: str,
        *,
        context_id: str | None = None,
        history: Sequence[ChatCompletionMessage] = (),
        prompt_overrides: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> AgentTurn:
        """Send one user message and return the agent's reply turn.

        context_id None starts a new conversation whose earlier messages are
        history, in OpenAI format; history is only read when context_id is None. A
        context_id from an earlier AgentTurn continues that conversation.
        prompt_overrides maps a prompt name to the text the agent uses in its place
        for this message. context is metadata sent with the message, such as the
        channel. Raises AgentError when the agent fails, or when it did not
        acknowledge the history or prompt_overrides it was sent.
        """


class CaseSource(Protocol):
    """A named file of labelled cases."""

    @property
    def name(self) -> str:
        """Return the stem of the file the cases come from."""

    def examples(self) -> Iterator[DatasetExample]:
        """Yield every case as a dataset example."""


class DatasetSink(Protocol):
    """A store that mirrors examples into a named dataset."""

    def sync(self, dataset_name: str, examples: Iterable[DatasetExample]) -> str:
        """Mirror examples into the named dataset and return its name."""

    def example_id(self, dataset_name: str, case_id: str) -> UUID:
        """Return the id the named dataset stores the case under."""


__all__ = ["Agent", "CaseSource", "DatasetSink"]
