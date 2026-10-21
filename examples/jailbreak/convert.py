"""Turn the prod jailbreak export into golden JSONL for the detector agent.

python -m examples.jailbreak.convert IN OUT reads every row of IN, then writes
OUT, which must not exist yet. A prod row has an id, an expected_action of
ALLOWED, BLOCKED or REVIEW, and exactly one of input_text, text or turns; a
history of {initiator, content} turns and a channel are optional, and every
other key, channel included, becomes golden metadata. A bad row stops the run,
naming path:line and the field without repeating any row text.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Self, TypedDict

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from src.toolsets.jailbreak_detection.settings import settings as detector_settings

from examples.jailbreak.settings import settings
from examples.jailbreak.types import ExpectedAction
from trajectory_evals.core.evaluators import _has_text
from trajectory_evals.core.jsonl import CaseFormatError, read_cases
from trajectory_evals.golden.cases import GoldenCase

type GoldenRow = dict[str, Any]


class HistoryTurn(TypedDict):
    """One conversation turn in the shape the sequence runner sends."""

    initiator: str
    content: str | None


class _Turn(TypedDict):
    """One role/text turn of a multi-turn row."""

    role: str
    text: str | None


class _TurnsRow(TypedDict):
    """The turns of a multi-turn row, oldest first and ending with the message."""

    turns: Annotated[list[_Turn], Field(min_length=1)]


_TURNS_ROW = TypeAdapter(_TurnsRow)


class ProdCase(BaseModel):
    """One labelled prod message, the turns before it, its channel and other keys."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
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

    def to_golden(self) -> GoldenRow:
        """Return the case as a golden row: conversation, channel, label, metadata.

        A message with no history is the row's input; otherwise the history and
        the message, if it has text, are its messages, each non-customer turn
        an assistant message named after its initiator.
        """
        conversation: GoldenRow = {"input": self.text}
        if self.history:
            messages = [_chat_message(turn) for turn in self.history]
            if _has_text(self.text):
                messages.append({"role": "user", "content": self.text})
            conversation = {"messages": messages}
        context = {} if self.channel is None else {settings.channels.key: self.channel}
        return {
            "id": self.id,
            **conversation,
            "context": context,
            "expected": {settings.keys.action: self.expected_action},
            **self.metadata,
        }


def golden_row(row: dict[str, Any]) -> GoldenRow:
    """Return one decoded prod row as a golden row; ValueError says what is wrong."""
    keys, texts = settings.data, settings.texts
    text, history = _message_and_history(row)
    metadata = {key: value for key, value in row.items() if key not in keys.row_keys}
    reserved = sorted(metadata.keys() & GoldenCase.model_fields.keys())
    if reserved:
        fields = texts.list_separator.join(reserved)
        raise ValueError(texts.reserved_fields.format(fields=fields))
    case = ProdCase.model_validate(
        {
            "id": row.get(keys.id_key),
            "text": text,
            "history": history,
            "expected_action": row.get(keys.label_key),
            "channel": row.get(settings.channels.key),
            "metadata": metadata,
        }
    )
    golden = case.to_golden()
    # The golden reader's own rules, such as its reserved metadata keys.
    GoldenCase.model_validate(golden)
    return golden


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


def _chat_message(turn: HistoryTurn) -> dict[str, str]:
    """Return a turn as a user message, or an assistant message named its initiator."""
    content = turn["content"] or ""
    if turn["initiator"] == detector_settings.input.customer_initiator:
        return {"role": "user", "content": content}
    return {"role": "assistant", "name": turn["initiator"], "content": content}


def convert(source: Path, target: Path) -> int:
    """Write source's rows to target as golden rows and return how many there were.

    Every row is read before target is created, so a bad row writes nothing.
    Raises CaseFormatError for a bad row and FileExistsError if target exists.
    """
    rows = list(read_cases(source, golden_row, lambda row: str(row["id"])))
    with target.open("x", encoding="utf-8") as file:
        file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return len(rows)


def _parser() -> argparse.ArgumentParser:
    """Return the parser of the input and output paths."""
    help_texts = settings.convert
    parser = argparse.ArgumentParser(
        prog=help_texts.prog, description=help_texts.description
    )
    parser.add_argument("source", type=Path, help=help_texts.input_help)
    parser.add_argument("target", type=Path, help=help_texts.output_help)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Convert the files argv names and return the exit code.

    Prints the case count and returns 0, or prints one error line to stderr and
    returns [convert] input_error_exit_code for an existing OUT, a missing IN or
    a bad row; OUT is then left as it was.
    """
    args = _parser().parse_args(argv)
    texts, prog = settings.texts, settings.convert.prog
    try:
        count = convert(args.source, args.target)
    except FileExistsError:
        error = texts.output_exists.format(path=args.target)
    except (CaseFormatError, FileNotFoundError) as e:
        error = str(e)
    else:
        sys.stdout.write(texts.converted.format(count=count, path=args.target) + "\n")
        return 0
    sys.stderr.write(texts.error_line.format(prog=prog, error=error) + "\n")
    return settings.convert.input_error_exit_code


__all__ = ["GoldenRow", "HistoryTurn", "ProdCase", "convert", "golden_row", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
