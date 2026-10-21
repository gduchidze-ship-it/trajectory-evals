"""Evaluator wrappers traced under their feedback key, and checks evaluators share.

Each wrapper names inputs, outputs and reference_outputs, because LangSmith's
evaluate passes an evaluator only the arguments its signature names.
"""

from collections.abc import Callable, Mapping
from typing import Any, TypeGuard

from openevals.utils import _arun_evaluator as _base_arun_evaluator
from openevals.utils import _run_evaluator as _base_run_evaluator

from trajectory_evals.core.settings import settings
from trajectory_evals.core.types import (
    EvaluatorResult,
    SimpleAsyncEvaluator,
    SimpleEvaluator,
)

_ERRORED_OUTPUTS: Mapping[str, Any] = {settings.evaluators.errored_output_key: None}


def _has_text(value: str | None) -> TypeGuard[str]:
    """Return whether value holds anything but whitespace."""
    return bool(value and value.strip())


def _target_errored(outputs: Mapping[str, Any] | None) -> bool:
    """Return whether outputs are what LangSmith passes for a raised target."""
    return not outputs or outputs == _ERRORED_OUTPUTS


# shortcut: LangSmith keys a raising wrapper's error row by its def name in the
# source, "wrapper", not __name__; return a RunEvaluator if that row needs the key.
def _evaluator(
    scorer: Callable[..., Any], feedback_key: str, *, ls_framework: str
) -> SimpleEvaluator:
    """Wrap scorer as a traced evaluator whose name, run and feedback are the key."""

    def wrapper(
        *,
        inputs: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any],
        reference_outputs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> EvaluatorResult | list[EvaluatorResult]:
        """Score one run under the feedback key."""
        return _base_run_evaluator(
            run_name=feedback_key,
            scorer=scorer,
            feedback_key=feedback_key,
            ls_framework=ls_framework,
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )

    wrapper.__name__ = feedback_key
    return wrapper


def _async_evaluator(
    scorer: Callable[..., Any], feedback_key: str, *, ls_framework: str
) -> SimpleAsyncEvaluator:
    """Wrap scorer as an async evaluator whose name, run and feedback are the key."""

    async def wrapper(
        *,
        inputs: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any],
        reference_outputs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> EvaluatorResult | list[EvaluatorResult]:
        """Score one run under the feedback key."""
        return await _base_arun_evaluator(
            run_name=feedback_key,
            scorer=scorer,
            feedback_key=feedback_key,
            ls_framework=ls_framework,
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )

    wrapper.__name__ = feedback_key
    return wrapper


__all__ = [
    "_async_evaluator",
    "_evaluator",
    "_has_text",
    "_target_errored",
]
