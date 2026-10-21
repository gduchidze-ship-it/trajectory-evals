"""Evaluator wrappers tagged as jailbreak-evals, and the detector call prod makes."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeGuard

from openevals.types import EvaluatorResult, SimpleAsyncEvaluator, SimpleEvaluator
from openevals.utils import _arun_evaluator as _base_arun_evaluator
from openevals.utils import _run_evaluator as _base_run_evaluator

from trajectory_evals.suites.jailbreak.settings import settings
from src.toolsets.jailbreak_detection.settings import settings as detector_settings
from src.toolsets.jailbreak_detection.types import ConversationTurn

_ERRORED_OUTPUTS: Mapping[str, Any] = {settings.langsmith.errored_output_key: None}


def _has_text(value: str | None) -> TypeGuard[str]:
    """Return whether value holds anything but whitespace."""
    return bool(value and value.strip())


def _target_errored(outputs: Mapping[str, Any] | None) -> bool:
    """Return whether outputs are what LangSmith passes for a raised target."""
    return not outputs or outputs == _ERRORED_OUTPUTS


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
    """Wrap scorer as a traced evaluator whose name, run and feedback are the key."""

    def wrapper(
        *,
        outputs: Mapping[str, Any],
        reference_outputs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> EvaluatorResult | list[EvaluatorResult]:
        """Score one run under the feedback key."""
        return _base_run_evaluator(
            run_name=feedback_key,
            scorer=scorer,
            feedback_key=feedback_key,
            ls_framework=settings.langsmith.ls_framework,
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )

    wrapper.__name__ = feedback_key
    return wrapper


def _async_evaluator(
    scorer: Callable[..., Any], feedback_key: str
) -> SimpleAsyncEvaluator:
    """Wrap scorer as an async evaluator whose name, run and feedback are the key."""

    async def wrapper(
        *,
        outputs: Mapping[str, Any],
        reference_outputs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> EvaluatorResult | list[EvaluatorResult]:
        """Score one run under the feedback key."""
        return await _base_arun_evaluator(
            run_name=feedback_key,
            scorer=scorer,
            feedback_key=feedback_key,
            ls_framework=settings.langsmith.ls_framework,
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )

    wrapper.__name__ = feedback_key
    return wrapper


__all__ = [
    "_async_evaluator",
    "_detector_arguments",
    "_evaluator",
    "_has_text",
    "_target_errored",
]
