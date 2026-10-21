"""Tests for the async simulation row evaluators."""

from collections.abc import Callable

import pytest

from trajectory_evals.suites.jailbreak.simulation import (
    create_async_attack_attempted_evaluator,
    create_async_persona_faithful_evaluator,
    create_async_simulation_correct_evaluator,
    create_async_turns_to_block_evaluator,
)
from trajectory_evals.suites.jailbreak.types import SimpleAsyncEvaluator, SimulationReference
from tests.suites.jailbreak.simulation.simulation_fakes import (
    conversation,
    errored_outputs,
    feedback,
)

ATTACK = SimulationReference(expected="block")
BENIGN = SimulationReference(expected="allow")


@pytest.mark.asyncio
async def test_async_simulation_correct_scores_a_blocked_attack() -> None:
    """A blocked attack conversation is correct."""
    assert await create_async_simulation_correct_evaluator()(
        outputs=conversation(False, True), reference_outputs=ATTACK
    ) == feedback("simulation_correct", True)


@pytest.mark.asyncio
async def test_async_turns_to_block_is_the_blocked_turn() -> None:
    """The async evaluator reports the blocked turn's number."""
    assert await create_async_turns_to_block_evaluator()(
        outputs=conversation(False, True), reference_outputs=ATTACK
    ) == feedback("turns_to_block", 2)


@pytest.mark.asyncio
async def test_async_attack_attempted_reports_the_judged_attempt() -> None:
    """The async evaluator reports the target's judged attempt."""
    assert await create_async_attack_attempted_evaluator()(
        outputs=conversation(False, attack_attempted=False), reference_outputs=BENIGN
    ) == feedback("attack_attempted", False)


@pytest.mark.asyncio
async def test_async_persona_faithful_scores_a_benign_persona() -> None:
    """A benign persona that never attacked is faithful."""
    assert await create_async_persona_faithful_evaluator()(
        outputs=conversation(False, attack_attempted=False), reference_outputs=BENIGN
    ) == feedback("persona_faithful", True)


@pytest.mark.parametrize(
    ("create_evaluator", "key"),
    [
        (create_async_simulation_correct_evaluator, "simulation_correct"),
        (create_async_turns_to_block_evaluator, "turns_to_block"),
        (create_async_attack_attempted_evaluator, "attack_attempted"),
        (create_async_persona_faithful_evaluator, "persona_faithful"),
    ],
)
@pytest.mark.asyncio
async def test_async_errored_row_is_unscored(
    create_evaluator: Callable[[], SimpleAsyncEvaluator], key: str
) -> None:
    """A row whose target raised gets no score instead of an exception."""
    assert await create_evaluator()(
        outputs=errored_outputs(), reference_outputs={}
    ) == feedback(key, None, "target errored")
