"""Row evaluators that report which cascade step decided each case."""

from typing import Any

from trajectory_evals.suites.jailbreak.detection.utils import _escalated, _failed_open
from trajectory_evals.suites.jailbreak.settings import settings
from trajectory_evals.suites.jailbreak.types import (
    DetectionOutputs,
    SimpleAsyncEvaluator,
    SimpleEvaluator,
)
from trajectory_evals.suites.jailbreak.utils import _async_evaluator, _evaluator, _target_errored


def _escalation_scorer(
    *, outputs: DetectionOutputs, **_kwargs: Any
) -> bool | tuple[None, str]:
    """Return whether the case escalated, or no score when the target errored."""
    if _target_errored(outputs):
        return None, settings.langsmith.target_errored_comment
    return _escalated(outputs)


def _fail_open_scorer(
    *, outputs: DetectionOutputs, **_kwargs: Any
) -> bool | tuple[None, str]:
    """Return whether the case failed open, or no score when the target errored."""
    if _target_errored(outputs):
        return None, settings.langsmith.target_errored_comment
    return _failed_open(outputs)


def create_escalation_evaluator() -> SimpleEvaluator:
    """Create an evaluator that flags each case a layer-one detector escalated."""
    return _evaluator(_escalation_scorer, settings.detection.escalated_key)


def create_async_escalation_evaluator() -> SimpleAsyncEvaluator:
    """Create an async evaluator that flags each case a layer-one detector escalated."""
    return _async_evaluator(_escalation_scorer, settings.detection.escalated_key)


def create_fail_open_evaluator() -> SimpleEvaluator:
    """Create an evaluator that flags each case the fail policy allowed."""
    return _evaluator(_fail_open_scorer, settings.detection.failed_open_key)


def create_async_fail_open_evaluator() -> SimpleAsyncEvaluator:
    """Create an async evaluator that flags each case the fail policy allowed."""
    return _async_evaluator(_fail_open_scorer, settings.detection.failed_open_key)
