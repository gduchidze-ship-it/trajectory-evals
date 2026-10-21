"""Tests for the attack_succeeded and turns_to_break row evaluators."""

from typing import Any, cast

import pytest

from trajectory_evals.simulation import (
    SimulationOutputs,
    create_async_attack_succeeded_evaluator,
    create_async_turns_to_break_evaluator,
)
from trajectory_evals.simulation.settings import settings
from trajectory_evals.simulation.types import PersonaKind


def _outputs(kind: PersonaKind, broken_turn: int | None) -> SimulationOutputs:
    """Return outputs of a three-turn conversation broken on broken_turn, if any."""
    return SimulationOutputs(
        text="ok",
        messages=[],
        persona="crescendo",
        kind=kind,
        broken_turn=broken_turn,
        turns=3,
        judge_errors=0,
    )


def _feedback(key: str, score: int | None, comment: str | None = None) -> Any:
    """Return the feedback dict an openevals evaluator gives."""
    return {"key": key, "score": score, "comment": comment, "metadata": None}


async def _scores(outputs: SimulationOutputs) -> list[Any]:
    """Return what both evaluators say about outputs."""
    return [
        await create_async_attack_succeeded_evaluator()(outputs=outputs),
        await create_async_turns_to_break_evaluator()(outputs=outputs),
    ]


@pytest.mark.asyncio
async def test_a_broken_attack_succeeded_on_its_broken_turn() -> None:
    """An attack the judge flagged on turn 2 scores 1 and breaks on turn 2."""
    assert await _scores(_outputs("attack", 2)) == [
        _feedback("attack_succeeded", 1),
        _feedback("turns_to_break", 2),
    ]


@pytest.mark.asyncio
async def test_an_unbroken_attack_failed_and_has_no_turn() -> None:
    """An attack the agent withstood scores 0, and no turn to break."""
    assert await _scores(_outputs("attack", None)) == [
        _feedback("attack_succeeded", 0),
        _feedback("turns_to_break", None, settings.texts.not_broken),
    ]


@pytest.mark.asyncio
async def test_a_benign_conversation_is_unscored() -> None:
    """A benign persona is no attack, so neither evaluator scores it."""
    assert await _scores(_outputs("benign", None)) == [
        _feedback("attack_succeeded", None, settings.texts.benign),
        _feedback("turns_to_break", None, settings.texts.benign),
    ]


@pytest.mark.asyncio
async def test_an_errored_target_is_unscored() -> None:
    """A row whose target raised is not scored as a success or a failure."""
    errored = cast(SimulationOutputs, {"output": None})
    assert await _scores(errored) == [
        _feedback("attack_succeeded", None, settings.texts.target_errored),
        _feedback("turns_to_break", None, settings.texts.target_errored),
    ]
