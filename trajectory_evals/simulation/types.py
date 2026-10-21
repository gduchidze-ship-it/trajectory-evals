"""Typed shapes that simulation cases, the target and its evaluators exchange."""

from typing import Any, Literal, TypedDict

from trajectory_evals.core.types import AgentOutputs

PersonaKind = Literal["benign", "attack"]


class ScenarioInputs(TypedDict):
    """A simulation case's inputs: who the user is, what they want, how it starts.

    persona names a persona of the agent. opening_message, when set, is the user's
    first message verbatim; otherwise the simulated user writes it. max_turns, when
    set, caps this case's user messages unless the command line caps them. context
    is metadata sent with every message, such as the channel.
    """

    persona: str
    scenario: str
    opening_message: str | None
    max_turns: int | None
    context: dict[str, Any]


class SimulationOutputs(AgentOutputs):
    """A simulated conversation: the transcript, and how its persona fared.

    turns counts the user messages the agent answered; the stop token is never
    one. broken_turn is the first of those turns in which the attack judge found
    the agent's reply in violation of the policy, or None. judge_errors counts
    the turns the judge failed to grade, each treated as not broken.
    """

    persona: str
    kind: PersonaKind
    broken_turn: int | None
    turns: int
    judge_errors: int


__all__ = ["PersonaKind", "ScenarioInputs", "SimulationOutputs"]
