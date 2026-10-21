"""Tests for core settings and the validation helpers packages share."""

import pytest
from pydantic import ValidationError

from trajectory_evals.core.settings import (
    CoreSettings,
    _require_placeholders,
    _require_unique,
    settings,
)
from trajectory_evals.suites.jailbreak.settings import settings as suite_settings

SHARED_TEXTS = {
    "not_an_object",
    "duplicate_case_id",
    "line_error",
    "field_error",
    "field_error_separator",
    "location_separator",
    "row_location",
    "at_least_one",
}


def test_shipped_toml_loads() -> None:
    """The shipped core config.toml loads into the module's settings."""
    assert CoreSettings() == settings


def test_core_and_jailbreak_suite_agree_on_shared_keys() -> None:
    """Every text and key core moved out of the suite has the same value in both."""
    core_texts = settings.texts.model_dump()
    suite_texts = suite_settings.texts.model_dump()

    assert core_texts.keys() & suite_texts.keys() == SHARED_TEXTS
    assert {key: core_texts[key] for key in SHARED_TEXTS} == {
        key: suite_texts[key] for key in SHARED_TEXTS
    }
    assert (
        settings.evaluators.errored_output_key
        == suite_settings.langsmith.errored_output_key
    )


def test_environment_overrides_one_core_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TRAJECTORY_EVALS_CORE_<SECTION>__<KEY> variable replaces only that value."""
    monkeypatch.setenv("TRAJECTORY_EVALS_CORE_TEXTS__ROW_LOCATION", "line")

    loaded = CoreSettings()

    assert loaded.texts.row_location == "line"
    assert loaded.texts.line_error == settings.texts.line_error


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("TRAJECTORY_EVALS_CORE_TEXTS__LINE_ERROR", "{path}: {message}", "exactly"),
        ("TRAJECTORY_EVALS_CORE_TEXTS__TYPO", "x", "Extra inputs are not permitted"),
        ("TRAJECTORY_EVALS_CORE_EVALUATORS__ERRORED_OUTPUT_KEY", "result", "'output'"),
    ],
)
def test_invalid_core_values_fail_at_load(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str, message: str
) -> None:
    """A stray placeholder, an unknown key or another errored key fails."""
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError, match=message):
        CoreSettings()


def test_require_unique_names_the_group_that_repeats() -> None:
    """Only a group holding a repeated name raises, and it is named with the repeat."""
    _require_unique({"roles": ("user", "assistant")})
    with pytest.raises(ValueError, match=r"^keys must be unique, but repeat \['a'\]$"):
        _require_unique({"roles": ("user", "assistant"), "keys": ("a", "b", "a")})


def test_require_placeholders_names_the_template_that_differs() -> None:
    """A template must use exactly the placeholders the code fills in."""
    _require_placeholders({"texts.ok": ("{name}: {value!r}", {"name", "value"})})
    with pytest.raises(ValueError, match=r"^texts.bad must use exactly .*\['path'\]$"):
        _require_placeholders({"texts.bad": ("{path} {line}", {"path"})})
