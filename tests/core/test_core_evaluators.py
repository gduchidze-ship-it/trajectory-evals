"""Tests for the traced evaluator wrapper and the errored-target check."""

from typing import Any

import pytest

from trajectory_evals.core import evaluators
from trajectory_evals.core.evaluators import _evaluator, _target_errored
from trajectory_evals.core.types import EvaluatorResult


def _score(*, outputs: dict[str, Any], **kwargs: Any) -> bool:
    """Pass a run whose outputs say ok."""
    return bool(outputs["ok"])


def test_evaluator_is_named_and_scores_under_its_key() -> None:
    """The wrapper's name and its feedback key are both the key it was given."""
    evaluator = _evaluator(_score, "correct", ls_framework="trajectory-evals")

    assert getattr(evaluator, "__name__") == "correct"
    assert evaluator(outputs={"ok": True}) == {
        "key": "correct",
        "score": True,
        "comment": None,
        "metadata": None,
    }


def test_evaluator_tags_its_run_with_the_given_framework(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """openevals traces the run under the feedback key and the caller's framework."""
    calls: list[dict[str, Any]] = []

    def record(**kwargs: Any) -> EvaluatorResult:
        """Record how openevals was called and return a fixed result."""
        calls.append(kwargs)
        return EvaluatorResult(
            key="correct", score=True, comment=None, metadata=None, source_run_id=None
        )

    monkeypatch.setattr(evaluators, "_base_run_evaluator", record)
    _evaluator(_score, "correct", ls_framework="golden-evals")(outputs={"ok": True})

    [call] = calls
    assert (call["run_name"], call["feedback_key"], call["ls_framework"]) == (
        "correct",
        "correct",
        "golden-evals",
    )


@pytest.mark.parametrize(
    ("outputs", "errored"),
    [
        (None, True),
        ({}, True),
        ({"output": None}, True),
        ({"output": "text"}, False),
        ({"text": "Hi", "messages": []}, False),
    ],
)
def test_target_errored_matches_only_langsmith_error_outputs(
    outputs: dict[str, Any] | None, errored: bool
) -> None:
    """Only empty outputs and LangSmith's raised-target outputs count as errored."""
    assert _target_errored(outputs) is errored
