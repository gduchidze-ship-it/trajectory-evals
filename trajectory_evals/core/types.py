"""Typed shapes that agents, datasets, targets and evaluators exchange."""

from typing import Any, TypedDict

from openevals.types import (
    ChatCompletionMessage,
    EvaluatorResult,
    SimpleAsyncEvaluator,
    SimpleEvaluator,
)


class DatasetExample(TypedDict):
    """One labelled case ready to mirror into a LangSmith dataset."""

    case_id: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    metadata: dict[str, Any]


__all__ = [
    "ChatCompletionMessage",
    "DatasetExample",
    "EvaluatorResult",
    "SimpleAsyncEvaluator",
    "SimpleEvaluator",
]
