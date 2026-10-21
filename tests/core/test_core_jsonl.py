"""Tests for reading cases lazily from JSONL with path:line errors."""

import json
import re
import traceback
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from trajectory_evals.core.jsonl import CaseFormatError, read_cases

SECRET = "card 4111"


class _Row(BaseModel):
    """A minimal case: an id and some text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    text: str


def _write(tmp_path: Path, *lines: dict[str, Any] | str) -> Path:
    """Write lines to cases.jsonl, dicts encoded as JSON, and return its path."""
    path = tmp_path / "cases.jsonl"
    encoded = (line if isinstance(line, str) else json.dumps(line) for line in lines)
    path.write_text("\n".join(encoded) + "\n", encoding="utf-8")
    return path


def _read(path: Path) -> list[_Row]:
    """Read every row of path as a _Row."""
    return list(read_cases(path, _Row.model_validate, lambda row: row.id))


def test_rows_come_out_in_file_order_skipping_blank_lines(tmp_path: Path) -> None:
    """Blank and whitespace-only lines yield nothing; the rest keep their order."""
    first, second = {"id": "a", "text": "x"}, {"id": "b", "text": "y"}
    path = _write(tmp_path, first, "", "  ", second)
    assert [row.id for row in _read(path)] == ["a", "b"]


def test_rows_are_read_lazily(tmp_path: Path) -> None:
    """A bad line raises only when iteration reaches it."""
    path = _write(tmp_path, {"id": "a", "text": "x"}, "not json")
    rows = read_cases(path, _Row.model_validate, lambda row: row.id)
    assert next(rows).id == "a"
    with pytest.raises(CaseFormatError):
        next(rows)


BAD_LINES: dict[str, tuple[dict[str, Any] | str, str]] = {
    "invalid_json": ('{"id": "x",', "Expecting"),
    "not_an_object": ("[1, 2]", "row is not a JSON object"),
    "missing_field": ({"id": "x"}, "text: Field required"),
    "empty_id": ({"id": "", "text": "x"}, "id: String should have at least 1"),
    "two_errors": ({"id": ""}, "id: String should have at least 1 .*; text: "),
    "repeated_id": ({"id": "a", "text": "y"}, "case_id 'a' appears more than once"),
}


@pytest.mark.parametrize(("line", "reason"), BAD_LINES.values(), ids=BAD_LINES.keys())
def test_bad_line_names_its_path_line_and_reason(
    tmp_path: Path, line: dict[str, Any] | str, reason: str
) -> None:
    """Every kind of bad line raises CaseFormatError as path:line: reason."""
    path = _write(tmp_path, {"id": "a", "text": "x"}, "", line)
    with pytest.raises(CaseFormatError, match=f"^{re.escape(str(path))}:3: {reason}"):
        _read(path)


def test_invalid_utf8_names_its_path_and_line(tmp_path: Path) -> None:
    """A line that is not UTF-8 is a format error naming path:line."""
    path = tmp_path / "cases.jsonl"
    path.write_bytes(b'{"id": "a", "text": "x"}\n{"id": "\xff"}\n')
    with pytest.raises(CaseFormatError, match="cases\\.jsonl:2: .*utf-8"):
        _read(path)


ECHO_LINES: dict[str, dict[str, Any] | str] = {
    "json_decode": f'{{"id": "x", "text": "{SECRET}", ',
    "wrong_type": {"id": "x", "text": [SECRET]},
    "unknown_field": {"id": "x", "text": "y", "note": SECRET},
}


@pytest.mark.parametrize("line", ECHO_LINES.values(), ids=ECHO_LINES.keys())
def test_error_traceback_never_echoes_row_text(
    tmp_path: Path, line: dict[str, Any] | str
) -> None:
    """The error drops its cause, so the full traceback holds no row text."""
    with pytest.raises(CaseFormatError) as raised:
        _read(_write(tmp_path, line))
    error = raised.value
    assert error.__cause__ is None
    assert error.__suppress_context__
    assert SECRET not in "".join(traceback.format_exception(error))
