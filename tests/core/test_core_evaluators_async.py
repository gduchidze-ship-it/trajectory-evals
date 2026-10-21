"""Tests for the traced async evaluator wrapper."""

from typing import Any

import pytest

from trajectory_evals.core import evaluators
from trajectory_evals.core.evaluators import _async_evaluator
from trajectory_evals.core.types import EvaluatorResult


async def _score(*, outputs: dict[str, Any], **kwargs: Any) -> bool:
    """Pass a run whose outputs say ok."""
    return bool(outputs["ok"])


@pytest.mark.asyncio
async def test_async_evaluator_is_named_and_scores_under_its_key() -> None:
    """The wrapper's name and its feedback key are both the key it was given."""
    evaluator = _async_evaluator(_score, "correct", ls_framework="trajectory-evals")

    assert getattr(evaluator, "__name__") == "correct"
    assert await evaluator(outputs={"ok": False}) == {
        "key": "correct",
        "score": False,
        "comment": None,
        "metadata": None,
    }


@pytest.mark.asyncio
async def test_async_evaluator_tags_its_run_with_the_given_framework(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """openevals traces the run under the feedback key and the caller's framework."""
    calls: list[dict[str, Any]] = []

    async def record(**kwargs: Any) -> EvaluatorResult:
        """Record how openevals was called and return a fixed result."""
        calls.append(kwargs)
        return EvaluatorResult(
            key="correct", score=True, comment=None, metadata=None, source_run_id=None
        )

    monkeypatch.setattr(evaluators, "_base_arun_evaluator", record)
    await _async_evaluator(_score, "correct", ls_framework="golden-evals")(
        outputs={"ok": True}
    )

    [call] = calls
    assert (call["run_name"], call["feedback_key"], call["ls_framework"]) == (
        "correct",
        "correct",
        "golden-evals",
    )
