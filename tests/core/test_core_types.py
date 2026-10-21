"""Tests for how the core wire shapes validate."""

from pydantic import TypeAdapter

from trajectory_evals.core.types import Expectations


def test_expectations_keep_keys_core_does_not_know() -> None:
    """A suite's own expected key, such as action, survives validation."""
    expected = {"contains": ["price"], "action": "BLOCKED"}
    assert TypeAdapter(Expectations).validate_python(expected) == expected
