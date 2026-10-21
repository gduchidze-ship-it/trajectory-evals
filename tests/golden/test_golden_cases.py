"""Tests for reading golden cases from JSONL into dataset examples."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.golden.golden_fakes import TOOL_CALL
from trajectory_evals.core.types import DatasetExample
from trajectory_evals.golden.cases import CaseFormatError, JsonlGoldenSource

SECRET = "card 4111"


def _write(tmp_path: Path, *rows: dict[str, Any]) -> Path:
    """Write rows to support.jsonl, one JSON object per line, and return its path."""
    path = tmp_path / "support.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
    return path


def _examples(path: Path) -> list[DatasetExample]:
    """Read every case of path as a dataset example."""
    return list(JsonlGoldenSource(path).examples())


def test_input_row_is_one_user_message_and_other_keys_are_metadata(
    tmp_path: Path,
) -> None:
    """input becomes the user message; risk and split land in metadata."""
    row = {
        "id": "refund-1",
        "input": "Refund order A1",
        "context": {"channel": "sms"},
        "expected": {"tools": ["refund_order"], "contains": ["refund"]},
        "risk": "high",
        "split": "val",
    }
    source = JsonlGoldenSource(_write(tmp_path, row))

    assert source.name == "support"
    assert list(source.examples()) == [
        DatasetExample(
            case_id="refund-1",
            inputs={
                "messages": [{"role": "user", "content": "Refund order A1"}],
                "context": {"channel": "sms"},
            },
            outputs={"tools": ["refund_order"], "contains": ["refund"]},
            metadata={"risk": "high", "split": "val", "source_file": "support.jsonl"},
        )
    ]


def test_messages_row_keeps_every_message_whole(tmp_path: Path) -> None:
    """A conversation, tool_call_id included, is the inputs' messages as written."""
    messages = [
        {"role": "user", "content": "Where is A1?"},
        {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
        {"role": "tool", "content": "shipped", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "It shipped."},
        {"role": "user", "content": "Refund it."},
    ]
    [example] = _examples(_write(tmp_path, {"id": "a", "messages": messages}))

    assert example["inputs"] == {"messages": messages, "context": {}}
    assert (example["outputs"], example["metadata"]) == (
        {},
        {"source_file": "support.jsonl"},
    )


def test_null_content_reads_as_empty_text(tmp_path: Path) -> None:
    """content null, as OpenAI exports write on tool calls, is "" in both places."""
    call = {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]}
    messages = [
        {"role": "user", "content": "Where is A1?"},
        call,
        {"role": "user", "content": "Refund it."},
    ]
    row = {"id": "a", "messages": messages, "expected": {"trajectory": [call]}}

    [example] = _examples(_write(tmp_path, row))

    empty = {**call, "content": ""}
    assert example["inputs"]["messages"][1] == empty
    assert example["outputs"]["trajectory"] == [empty]


def test_unknown_expected_keys_pass_through(tmp_path: Path) -> None:
    """A suite's own label, such as action, reaches the example untouched."""
    expected = {"action": "BLOCKED", "not_contains": ["system prompt"]}
    path = _write(tmp_path, {"id": "a", "input": "hi", "expected": expected})
    assert _examples(path)[0]["outputs"] == expected


BAD_ROWS: dict[str, tuple[dict[str, Any], str]] = {
    "input_and_messages": (
        {"id": "a", "input": "hi", "messages": [{"role": "user", "content": "hi"}]},
        "row: Value error, row needs exactly one of input or messages",
    ),
    "neither": ({"id": "a"}, "row: .*exactly one of input or messages"),
    "blank_input": ({"id": "a", "input": "  "}, "row: .*input must hold text"),
    "last_role_not_user": (
        {"id": "a", "messages": [{"role": "assistant", "content": SECRET}]},
        "row: .*the last message must be a user message with text",
    ),
    "blank_last_message": (
        {"id": "a", "messages": [{"role": "user", "content": " "}]},
        "row: .*the last message must be a user message",
    ),
    "no_messages": ({"id": "a", "messages": []}, "row: .*the last message must"),
    "message_without_role": (
        {"id": "a", "messages": [{"content": SECRET}]},
        "messages.0.role: Field required",
    ),
    "bad_expected_type": (
        {"id": "a", "input": "hi", "expected": {"tools": SECRET}},
        "expected.tools: Input should be a valid list",
    ),
    "bad_trajectory_mode": (
        {"id": "a", "input": "hi", "expected": {"trajectory_match": "fuzzy"}},
        "expected.trajectory_match: Input should be",
    ),
    "reserved_metadata": (
        {"id": "a", "input": "hi", "source_file": "x", "case_id": "y"},
        "row: .*row uses reserved fields: case_id, source_file",
    ),
}


@pytest.mark.parametrize(("row", "reason"), BAD_ROWS.values(), ids=BAD_ROWS.keys())
def test_bad_row_names_its_path_line_and_reason_without_its_text(
    tmp_path: Path, row: dict[str, Any], reason: str
) -> None:
    """Every bad row raises CaseFormatError as path:line: reason, echoing no value."""
    path = _write(tmp_path, {"id": "ok", "input": "hi"}, row)

    with pytest.raises(CaseFormatError, match=f"^{re.escape(str(path))}:2: {reason}"):
        _examples(path)
    with pytest.raises(CaseFormatError) as raised:
        _examples(path)
    assert SECRET not in str(raised.value)
