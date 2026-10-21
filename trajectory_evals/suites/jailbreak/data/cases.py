"""Labelled jailbreak cases read lazily from JSONL files."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Self, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from trajectory_evals.suites.jailbreak.settings import settings
from trajectory_evals.suites.jailbreak.types import (
    DatasetExample,
    DetectionInputs,
    DetectionReference,
    ExpectedAction,
    HistoryTurn,
)
from trajectory_evals.suites.jailbreak.utils import _has_text
from src.toolsets.jailbreak_detection.settings import settings as detector_settings


class CaseFormatError(ValueError):
    """A JSONL row that cannot be read as a labelled case."""


class _Turn(TypedDict):
    """One role/text turn of a multi-turn row."""

    role: str
    text: str | None


class _TurnsRow(TypedDict):
    """The turns of a multi-turn row, oldest first and ending with the message."""

    turns: Annotated[list[_Turn], Field(min_length=1)]


_TURNS_ROW = TypeAdapter(_TurnsRow)


class EvalCase(BaseModel):
    """One labelled message, the turns before it, its channel and other fields."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    text: str | None
    history: tuple[HistoryTurn, ...] = ()
    expected_action: ExpectedAction
    channel: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_message(self) -> Self:
        """Require text, or a history whose newest turn is a customer turn with text."""
        if _has_text(self.text):
            return self
        if not self.history or not _is_customer_message(self.history[-1]):
            raise ValueError(settings.texts.no_message)
        return self

    def to_example(self) -> DatasetExample:
        """Return the case as a detection dataset example."""
        inputs = DetectionInputs(
            text=self.text,
            history=[turn.copy() for turn in self.history],
            channel=self.channel,
        )
        reference = DetectionReference(expected_action=self.expected_action)
        return DatasetExample(
            case_id=self.case_id,
            inputs=dict(inputs),
            outputs=dict(reference),
            metadata=dict(self.metadata),
        )


class JsonlCaseSource:
    """The labelled cases of one JSONL file, read lazily in file order."""

    def __init__(self, path: Path) -> None:
        """Keep the path of the JSONL file that iteration will read."""
        self._path = path

    @property
    def name(self) -> str:
        """Return the stem of the file the cases come from."""
        return self._path.stem

    def cases(self) -> Iterator[EvalCase]:
        """Yield a validated case per non-blank line in file order; ids never repeat."""
        seen: set[str] = set()
        for line_number, line in self._lines():
            case = self._read_case(line, line_number)
            if case.case_id in seen:
                message = settings.texts.duplicate_case_id.format(case_id=case.case_id)
                raise self._error(line_number, message)
            seen.add(case.case_id)
            yield case

    def examples(self) -> Iterator[DatasetExample]:
        """Yield every case as a dataset example."""
        return (case.to_example() for case in self.cases())

    def _lines(self) -> Iterator[tuple[int, bytes]]:
        """Yield each non-blank line with its number, reading the file lazily."""
        with self._path.open("rb") as lines:
            for line_number, line in enumerate(lines, start=1):
                if line.strip():
                    yield line_number, line

    def _read_case(self, line: bytes, line_number: int) -> EvalCase:
        """Decode and parse one line into a case, naming path:line when invalid."""
        try:
            row = json.loads(line.decode("utf-8"))
            return _case_from_row(row, self._path.name)
        except ValueError as error:
            raise self._error(line_number, _describe(error)) from None

    def _error(self, line_number: int, message: str) -> CaseFormatError:
        """Return the error for a line of this file, prefixed with path:line."""
        line_error = settings.texts.line_error
        return CaseFormatError(
            line_error.format(path=self._path, line=line_number, message=message)
        )


def _case_from_row(row: object, source_file: str) -> EvalCase:
    """Build a case from one decoded JSONL row."""
    keys, texts = settings.data, settings.texts
    if not isinstance(row, dict):
        raise ValueError(texts.not_an_object)
    reserved = sorted(row.keys() & keys.reserved_keys)
    if reserved:
        fields = texts.list_separator.join(reserved)
        raise ValueError(texts.reserved_fields.format(fields=fields))
    text, history = _message_and_history(row)
    metadata = {key: value for key, value in row.items() if key not in keys.row_keys}
    return EvalCase.model_validate(
        {
            "case_id": row.get(keys.id_key),
            "text": text,
            "history": history,
            "expected_action": row.get(keys.label_key),
            "channel": row.get(settings.channels.key),
            "metadata": {**metadata, keys.source_file_key: source_file},
        }
    )


def _message_and_history(row: dict[str, Any]) -> tuple[Any, Any]:
    """Return the message under check and the turns before it, from either shape."""
    keys, texts = settings.data, settings.texts
    sources = [key for key in keys.message_keys if key in row]
    if len(sources) != 1:
        names = texts.list_separator.join(keys.message_keys)
        raise ValueError(texts.message_sources.format(keys=names))
    if sources[0] != keys.turns_key:
        return row[sources[0]], row.get(keys.history_key, ())
    if keys.history_key in row:
        message = texts.turns_with_history
        raise ValueError(message.format(turns=keys.turns_key, history=keys.history_key))
    return _split_turns(row[keys.turns_key])


def _split_turns(turns: object) -> tuple[str | None, tuple[HistoryTurn, ...]]:
    """Split a row's turns into its newest message and the history before it."""
    texts = settings.texts
    *earlier, latest = _TURNS_ROW.validate_python({"turns": turns})["turns"]
    if latest["role"].capitalize() != detector_settings.input.customer_initiator:
        message = texts.last_turn_not_customer
        raise ValueError(message.format(turns=settings.data.turns_key))
    if not _has_text(latest["text"]):
        raise ValueError(texts.no_message)
    history = tuple(
        HistoryTurn(initiator=turn["role"].capitalize(), content=turn["text"])
        for turn in earlier
    )
    return latest["text"], history


def _is_customer_message(turn: HistoryTurn) -> bool:
    """Return whether turn is the customer's and holds text."""
    is_customer = turn["initiator"] == detector_settings.input.customer_initiator
    return is_customer and _has_text(turn["content"])


def _describe(error: ValueError) -> str:
    """Describe what is wrong with a row without echoing any of its values."""
    if not isinstance(error, ValidationError):
        return str(error)
    texts = settings.texts
    field_error = texts.field_error
    return texts.field_error_separator.join(
        field_error.format(location=_location(detail["loc"]), reason=detail["msg"])
        for detail in error.errors(include_url=False, include_input=False)
    )


def _location(path: tuple[int | str, ...]) -> str:
    """Return a validation error's field path, or the whole-row label if it is empty."""
    texts = settings.texts
    return texts.location_separator.join(map(str, path)) or texts.row_location


__all__ = ["CaseFormatError", "EvalCase", "JsonlCaseSource"]
