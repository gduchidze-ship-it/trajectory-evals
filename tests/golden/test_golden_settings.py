"""Tests for the golden package's settings and the checks on them."""

import pytest
from pydantic import ValidationError

from trajectory_evals.golden.settings import GoldenSettings, settings


def test_shipped_toml_loads() -> None:
    """The shipped golden config.toml loads into the module's settings."""
    assert GoldenSettings() == settings


BAD_OVERRIDES = {
    "criteria_placeholders": (
        "TRAJECTORY_EVALS_GOLDEN_JUDGES__CRITERIA_PROMPT",
        "Grade {transcript}",
        "judges.criteria_prompt must use exactly",
    ),
    "shared_feedback_key": (
        "TRAJECTORY_EVALS_GOLDEN_KEYS__CONTAINS",
        "criteria",
        "feedback keys must be unique",
    ),
    "source_file_key_core_writes": (
        "TRAJECTORY_EVALS_GOLDEN_DATA__SOURCE_FILE_KEY",
        "case_id",
        "metadata keys must be unique",
    ),
}


@pytest.mark.parametrize(
    ("variable", "value", "reason"), BAD_OVERRIDES.values(), ids=BAD_OVERRIDES.keys()
)
def test_an_override_code_cannot_use_is_rejected(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str, reason: str
) -> None:
    """A prompt missing a placeholder, or a key two graders share, fails to load."""
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValidationError, match=reason):
        GoldenSettings()
