"""Cases read lazily from JSONL files, with errors that name path:line."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from trajectory_evals.core.settings import settings


class CaseFormatError(ValueError):
    """A JSONL row that cannot be read as a labelled case."""


def read_cases[Case](
    path: Path,
    parse: Callable[[dict[str, Any]], Case],
    case_id: Callable[[Case], str],
) -> Iterator[Case]:
    """Yield parse(row) per non-blank line in file order; case ids never repeat.

    parse raises ValueError (e.g. pydantic ValidationError) for a bad row, with a
    message that does not repeat row values.
    """
    seen: set[str] = set()
    for line_number, line in _lines(path):
        try:
            case = parse(_row(line))
        except ValueError as error:
            raise _line_error(path, line_number, _describe(error)) from None
        if case_id(case) in seen:
            message = settings.texts.duplicate_case_id.format(case_id=case_id(case))
            raise _line_error(path, line_number, message)
        seen.add(case_id(case))
        yield case


def _lines(path: Path) -> Iterator[tuple[int, bytes]]:
    """Yield each non-blank line with its number, reading the file lazily."""
    with path.open("rb") as lines:
        for line_number, line in enumerate(lines, start=1):
            if line.strip():
                yield line_number, line


def _row(line: bytes) -> dict[str, Any]:
    """Decode one line as a JSON object."""
    row = json.loads(line.decode("utf-8"))
    if not isinstance(row, dict):
        raise ValueError(settings.texts.not_an_object)
    return row


def _line_error(path: Path, line_number: int, message: str) -> CaseFormatError:
    """Return the error for a line of path, prefixed with path:line."""
    line_error = settings.texts.line_error
    return CaseFormatError(
        line_error.format(path=path, line=line_number, message=message)
    )


def _describe(error: Exception) -> str:
    """Describe what is wrong with a value without echoing any part of it."""
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


__all__ = ["CaseFormatError", "_describe", "read_cases"]
