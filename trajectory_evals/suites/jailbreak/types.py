"""Typed shapes that jailbreak datasets, targets and evaluators exchange."""

from typing import Any, Literal, Protocol, TypedDict

from trajectory_evals.core.types import (
    ChatCompletionMessage,
    DatasetExample,
    EvaluatorResult,
    SimpleAsyncEvaluator,
    SimpleEvaluator,
)

ChatRole = Literal["system", "user", "assistant"]
ExpectedAction = Literal["ALLOWED", "BLOCKED", "REVIEW"]
PersonaKind = Literal["benign", "attack"]
SimulationExpectation = Literal["allow", "block"]


class HistoryTurn(TypedDict):
    """One conversation turn in the shape the sequence runner sends."""

    initiator: str
    content: str | None


class DetectionInputs(TypedDict):
    """The message under check, the conversation before it and its channel."""

    text: str | None
    history: list[HistoryTurn]
    channel: str | None


class DetectionReference(TypedDict):
    """The label a detection case expects."""

    expected_action: ExpectedAction


class DetectionOutputs(TypedDict):
    """The detector's result plus how the cascade reached it."""

    is_jailbreak: bool
    decided_by: str
    escalated_by: str
    history_read: bool
    judge_reminded: bool


class TurnDetection(TypedDict):
    """The detector's result on one simulated customer turn."""

    turn: int
    text: str
    is_jailbreak: bool
    decided_by: str
    escalated_by: str


class SimulationInputs(TypedDict):
    """The real opening message and the persona that continues it."""

    opening_message: str
    persona: str
    channel: str


class SimulationReference(TypedDict):
    """Whether the simulated conversation should end blocked or allowed."""

    expected: SimulationExpectation


class SimulationOutputs(TypedDict):
    """The simulated conversation, each turn's detection and the judged attempt."""

    trajectory: list[ChatCompletionMessage]
    detections: list[TurnDetection]
    attack_attempted: bool | None


class SummaryEvaluator(Protocol):
    """A LangSmith summary evaluator that reports every metric of one experiment."""

    def __call__(
        self, outputs: list[Any], reference_outputs: list[Any]
    ) -> dict[str, list[dict[str, Any]]]:
        """Return {"results": [...]} with one key and score per metric."""


__all__ = [
    "ChatCompletionMessage",
    "ChatRole",
    "DatasetExample",
    "DetectionInputs",
    "DetectionOutputs",
    "DetectionReference",
    "EvaluatorResult",
    "ExpectedAction",
    "HistoryTurn",
    "PersonaKind",
    "SimpleAsyncEvaluator",
    "SimpleEvaluator",
    "SimulationExpectation",
    "SimulationInputs",
    "SimulationOutputs",
    "SimulationReference",
    "SummaryEvaluator",
    "TurnDetection",
]
