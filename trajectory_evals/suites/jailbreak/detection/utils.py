"""Rules on detection outputs and labels that several evaluators share."""

from typing import get_args

from src.toolsets.jailbreak_detection.types import DecidedBy

from trajectory_evals.suites.jailbreak.settings import settings
from trajectory_evals.suites.jailbreak.types import (
    DetectionOutputs,
    DetectionReference,
    ExpectedAction,
)

_EXPECTED_ACTIONS: frozenset[str] = frozenset(get_args(ExpectedAction))


def _expected_action(reference_outputs: DetectionReference) -> ExpectedAction:
    """Return the case's label, raising ValueError for one ExpectedAction lacks."""
    expected_action = reference_outputs["expected_action"]
    if expected_action not in _EXPECTED_ACTIONS:
        raise ValueError(settings.texts.unknown_label.format(label=expected_action))
    return expected_action


def _escalated(outputs: DetectionOutputs) -> bool:
    """Return whether a layer-one detector sent the case past the screen."""
    # src is Any to mypy without the detector repo on MYPYPATH.
    return outputs["decided_by"] != DecidedBy.SCREEN.value  # type: ignore[no-any-return]


def _failed_open(outputs: DetectionOutputs) -> bool:
    """Return whether the judge reached no verdict, so the case was allowed."""
    # src is Any to mypy without the detector repo on MYPYPATH.
    return outputs["decided_by"] == DecidedBy.FAIL_POLICY.value  # type: ignore[no-any-return]
