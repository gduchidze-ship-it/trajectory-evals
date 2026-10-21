"""Evaluator wrappers tagged as jailbreak-evals, and the detector call prod makes."""

from collections.abc import Callable, Sequence
from typing import Any

from src.toolsets.jailbreak_detection.settings import settings as detector_settings
from src.toolsets.jailbreak_detection.types import ConversationTurn

from trajectory_evals.core import evaluators as core_evaluators
from trajectory_evals.core.evaluators import _has_text, _target_errored
from trajectory_evals.suites.jailbreak.settings import settings
from trajectory_evals.suites.jailbreak.types import (
    SimpleAsyncEvaluator,
    SimpleEvaluator,
)


def _detector_arguments(
    channel: str | None, text: str | None, prior: Sequence[ConversationTurn]
) -> tuple[str | None, tuple[ConversationTurn, ...]]:
    """Return the text and history prod sends the detector for a customer turn."""
    channels = settings.channels
    if not prior and channel == channels.opener_text_only:
        return text, ()
    current = ConversationTurn(
        initiator=detector_settings.input.customer_initiator, content=text
    )
    history = (*prior, current)
    if channel == channels.conversation_only:
        return None, history
    return text, history


def _evaluator(scorer: Callable[..., Any], feedback_key: str) -> SimpleEvaluator:
    """Wrap scorer as a traced evaluator tagged with the suite's ls_framework."""
    return core_evaluators._evaluator(
        scorer, feedback_key, ls_framework=settings.langsmith.ls_framework
    )


def _async_evaluator(
    scorer: Callable[..., Any], feedback_key: str
) -> SimpleAsyncEvaluator:
    """Wrap scorer as an async evaluator tagged with the suite's ls_framework."""
    return core_evaluators._async_evaluator(
        scorer, feedback_key, ls_framework=settings.langsmith.ls_framework
    )


__all__ = [
    "_async_evaluator",
    "_detector_arguments",
    "_evaluator",
    "_has_text",
    "_target_errored",
]
