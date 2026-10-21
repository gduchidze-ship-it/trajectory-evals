"""Experiment rows built from only the fields a core test cares about."""

from collections.abc import Mapping
from typing import Any

from trajectory_evals.core.runner import ExperimentRow


def make_row(
    case_id: str = "case-0",
    feedback: Mapping[str, float | bool | None] | None = None,
    **fields: Any,
) -> ExperimentRow:
    """Return a row of case_id with the given feedback and fields, others empty."""
    defaults: dict[str, Any] = {
        "inputs": {},
        "outputs": {},
        "reference_outputs": {},
        "metadata": {},
        "comments": {},
        "error": None,
        "latency_ms": None,
    }
    return ExperimentRow(
        case_id=case_id, feedback=dict(feedback or {}), **{**defaults, **fields}
    )


__all__ = ["make_row"]
