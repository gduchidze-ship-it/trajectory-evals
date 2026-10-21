"""Golden cases read lazily from JSONL: a conversation, its context and expectations."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trajectory_evals.core.evaluators import _has_text
from trajectory_evals.core.jsonl import CaseFormatError, read_cases
from trajectory_evals.core.types import (
    AgentInputs,
    ChatCompletionMessage,
    DatasetExample,
    Expectations,
)
from trajectory_evals.golden.settings import settings


class GoldenCase(BaseModel):
    """One golden row; keys besides the fields below are its metadata.

    The row gives exactly one of input, a single user message, or messages, an
    OpenAI-format conversation ending with a user message with text.
    """

    # allow also reaches the nested message and Expectations dicts, so validation
    # keeps tool_call_id and expected keys core does not know, like a suite's label.
    model_config = ConfigDict(frozen=True, extra="allow")

    id: str = Field(min_length=1)
    input: str | None = None
    messages: list[ChatCompletionMessage] | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    expected: Expectations = Field(default_factory=Expectations)

    @field_validator("messages", mode="before")
    @classmethod
    def _messages_null_content(cls, messages: Any) -> Any:
        """Read content null, as OpenAI exports write on tool calls, as ""."""
        return _null_content_to_empty(messages)

    @field_validator("expected", mode="before")
    @classmethod
    def _trajectory_null_content(cls, expected: Any) -> Any:
        """Read content null in expected.trajectory as "", as in messages."""
        if not isinstance(expected, dict) or "trajectory" not in expected:
            return expected
        trajectory = _null_content_to_empty(expected["trajectory"])
        return {**expected, "trajectory": trajectory}

    @model_validator(mode="after")
    def _check_row(self) -> Self:
        """Require one message source with user text, and no reserved metadata."""
        texts = settings.texts
        reserved = sorted(self.metadata.keys() & _reserved_keys())
        if reserved:
            fields = texts.list_separator.join(reserved)
            raise ValueError(texts.reserved_fields.format(fields=fields))
        if (self.input is None) == (self.messages is None):
            raise ValueError(texts.input_or_messages)
        if self.input is not None and not _has_text(self.input):
            raise ValueError(texts.blank_input)
        if self.messages is not None and not _ends_with_user_text(self.messages):
            raise ValueError(texts.last_not_user)
        return self

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the row's other keys, such as risk, difficulty or split."""
        return dict(self.model_extra or {})

    def to_example(self, source_file: str) -> DatasetExample:
        """Return the case as a dataset example whose metadata names its file."""
        inputs = AgentInputs(messages=self._conversation(), context=self.context)
        return DatasetExample(
            case_id=self.id,
            inputs=dict(inputs),
            outputs=dict(self.expected),
            metadata={**self.metadata, settings.data.source_file_key: source_file},
        )

    def _conversation(self) -> list[ChatCompletionMessage]:
        """Return messages, or input as the single user message."""
        if self.messages is not None:
            return self.messages
        return [{"role": settings.data.user_role, "content": self.input or ""}]


class JsonlGoldenSource:
    """The golden cases of one JSONL file, read lazily in file order."""

    def __init__(self, path: Path) -> None:
        """Keep the path of the JSONL file that iteration will read."""
        self._path = path

    @property
    def name(self) -> str:
        """Return the stem of the file the cases come from."""
        return self._path.stem

    def cases(self) -> Iterator[GoldenCase]:
        """Yield a validated case per non-blank line in file order; ids never repeat.

        A bad row raises CaseFormatError naming path:line and the field, without
        repeating any row value.
        """
        return read_cases(self._path, GoldenCase.model_validate, lambda case: case.id)

    def examples(self) -> Iterator[DatasetExample]:
        """Yield every case as a dataset example."""
        source_file = self._path.name
        return (case.to_example(source_file) for case in self.cases())


def _null_content_to_empty(messages: Any) -> Any:
    """Return messages with each content null as ""; other shapes go to validation."""
    if not isinstance(messages, list):
        return messages
    return [
        {**message, "content": ""}
        if isinstance(message, dict) and message.get("content", "") is None
        else message
        for message in messages
    ]


def _ends_with_user_text(messages: list[ChatCompletionMessage]) -> bool:
    """Return whether the last message is the user's and its content is text."""
    if not messages:
        return False
    last = messages[-1]
    content = last["content"]
    is_text = isinstance(content, str) and _has_text(content)
    return last["role"] == settings.data.user_role and is_text


def _reserved_keys() -> set[str]:
    """Return the metadata keys the data layer writes, which a row cannot use."""
    data = settings.data
    return {data.source_file_key, *data.reserved_keys}


__all__ = ["CaseFormatError", "GoldenCase", "JsonlGoldenSource"]
