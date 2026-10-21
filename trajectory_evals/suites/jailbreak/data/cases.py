"""Labelled jailbreak cases read lazily from JSONL files."""

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Self, TypedDict

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from src.toolsets.jailbreak_detection.settings import settings as detector_settings

from trajectory_evals.core.jsonl import CaseFormatError, read_cases
from trajectory_evals.suites.jailbreak.settings import settings
from trajectory_evals.suites.jailbreak.types import (
    DatasetExample,
    DetectionInputs,
    DetectionReference,
    ExpectedAction,
    HistoryTurn,
)
from trajectory_evals.suites.jailbreak.utils import _has_text


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
        source_file = self._path.name
        return read_cases(
            self._path,
            lambda row: _case_from_row(row, source_file),
            lambda case: case.case_id,
        )

    def examples(self) -> Iterator[DatasetExample]:
        """Yield every case as a dataset example."""
        return (case.to_example() for case in self.cases())


def _case_from_row(row: dict[str, Any], source_file: str) -> EvalCase:
    """Build a case from one decoded JSONL row."""
    keys, texts = settings.data, settings.texts
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


__all__ = ["CaseFormatError", "EvalCase", "JsonlCaseSource"]
