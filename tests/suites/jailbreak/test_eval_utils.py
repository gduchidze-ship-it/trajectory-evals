"""Tests for the shared errored-target and has-text checks."""

from typing import Any

import pytest

from evals.jailbreak.utils import _has_text, _target_errored


@pytest.mark.parametrize(
    ("outputs", "errored"),
    [
        (None, True),
        ({}, True),
        ({"output": None}, True),
        ({"is_jailbreak": False, "decided_by": "screen"}, False),
        ({"trajectory": [], "detections": []}, False),
        ({"output": "text"}, False),
        ({"output": None, "decided_by": "screen"}, False),
    ],
)
def test_target_errored_matches_only_langsmith_error_outputs(
    outputs: dict[str, Any] | None, errored: bool
) -> None:
    """Only empty outputs and LangSmith's raised-target outputs count as errored."""
    assert _target_errored(outputs) is errored


@pytest.mark.parametrize(
    ("value", "has_text"),
    [(None, False), ("", False), (" \n\t", False), ("Hi", True), ("  Hi ", True)],
)
def test_has_text_is_false_for_none_empty_and_whitespace(
    value: str | None, has_text: bool
) -> None:
    """Only a value with a non-whitespace character holds text."""
    assert _has_text(value) is has_text
