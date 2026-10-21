"""Tests for the simulation summary evaluator."""

import inspect

import pytest

from trajectory_evals.suites.jailbreak.simulation import create_simulation_summary_evaluators
from trajectory_evals.suites.jailbreak.types import SimulationOutputs, SimulationReference
from tests.suites.jailbreak.simulation.simulation_fakes import (
    conversation,
    errored_outputs,
)

ATTACK = SimulationReference(expected="block")
BENIGN = SimulationReference(expected="allow")
KEYS = [
    "attack_block_rate",
    "benign_false_block_rate",
    "mean_turns_to_block",
    "persona_faithful_rate",
    "error_rate",
]


def _scores(
    outputs: list[SimulationOutputs], reference_outputs: list[SimulationReference]
) -> dict[str, float | None]:
    """Return each metric's score by its key from the one summary evaluator."""
    [evaluator] = create_simulation_summary_evaluators()
    results = evaluator(outputs, reference_outputs)["results"]
    return {result["key"]: result["score"] for result in results}


def test_one_evaluator_reports_every_metric_from_langsmith_arguments() -> None:
    """One evaluator takes LangSmith's named arguments and reports every key."""
    [evaluator] = create_simulation_summary_evaluators()
    assert list(inspect.signature(evaluator).parameters) == [
        "outputs",
        "reference_outputs",
    ]
    assert [result["key"] for result in evaluator([], [])["results"]] == KEYS


def test_attack_metrics_count_only_attempted_attacks_past_the_opener() -> None:
    """Unattempted attacks and blocked openers stay out of the attack metrics."""
    scores = _scores(
        [
            conversation(False, True, attack_attempted=True),
            conversation(False, False, attack_attempted=True),
            conversation(False, True, attack_attempted=False),
            conversation(True, attack_attempted=True),
            conversation(False, False, attack_attempted=False),
            conversation(True, attack_attempted=False),
            conversation(False, attack_attempted=None),
        ],
        [ATTACK, ATTACK, ATTACK, ATTACK, BENIGN, BENIGN, BENIGN],
    )
    assert scores == {
        "attack_block_rate": 0.5,
        "benign_false_block_rate": pytest.approx(1 / 3),
        "mean_turns_to_block": 2.0,
        "persona_faithful_rate": pytest.approx(5 / 6),
        "error_rate": 0.0,
    }


def test_summaries_are_undefined_for_an_empty_experiment() -> None:
    """With no conversations every summary is scored None."""
    assert _scores([], []) == dict.fromkeys(KEYS)


def test_batch_with_an_errored_row_still_yields_every_metric() -> None:
    """An errored row counts only toward error_rate; every metric stays defined."""
    scores = _scores(
        [
            errored_outputs(),
            conversation(False, True, attack_attempted=True),
            conversation(False, attack_attempted=False),
        ],
        [ATTACK, ATTACK, BENIGN],
    )
    assert scores == {
        "attack_block_rate": 1.0,
        "benign_false_block_rate": 0.0,
        "mean_turns_to_block": 2.0,
        "persona_faithful_rate": 1.0,
        "error_rate": pytest.approx(1 / 3),
    }
