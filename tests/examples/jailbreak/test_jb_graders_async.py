"""The jailbreak graders as the [graders] hook loads them, on target outputs."""

import logging
from pathlib import Path
from typing import Any, cast

import pytest

from examples.jailbreak.graders import assistant
from tests.examples.jailbreak.jb_fakes import (
    conversation_outputs,
    detection_outputs,
    errored_outputs,
    feedback,
)
from tests.fakes import FailingChatModel, FakeChatModel, judge_reply, scripted
from tests.golden.golden_fakes import ModelFactory, agent_config
from trajectory_evals.core.types import SimpleAsyncEvaluator
from trajectory_evals.golden.graders import Graders, load_custom_graders

REVIEW_COMMENT = "REVIEW label is unconfirmed, so the verdict is not scored."
NOT_JUDGED = "attack attempt not judged"
SIMULATION_KEYS = [
    "simulation_correct",
    "turns_to_block",
    "attack_attempted",
    "persona_faithful",
]
OPENED: dict[str, Any] = {"opening_message": "turn 1"}
JUDGE_MODEL = "fake:judge"


def _evaluators(graders: Graders) -> dict[str, SimpleAsyncEvaluator]:
    """Return the graders' evaluators by feedback key."""
    # Every jailbreak grader is async, named after its feedback key.
    return {
        cast(Any, evaluator).__name__: cast(SimpleAsyncEvaluator, evaluator)
        for evaluator in graders.evaluators
    }


def _detector_graders() -> Graders:
    """Return what a [graders] table naming the detector factory loads."""
    graders = {"factory": "examples.jailbreak.graders:detector"}
    return load_custom_graders(agent_config(Path("detector"), graders=graders))


def _assistant_graders(judge: FakeChatModel) -> Graders:
    """Return the assistant's graders whose [redteam] judge_model is judge."""
    config = agent_config(Path("assistant"), redteam={"judge_model": JUDGE_MODEL})
    return assistant(config, model_factory=ModelFactory({JUDGE_MODEL: judge}))


async def _grade(
    graders: Graders, outputs: dict[str, Any], inputs: dict[str, Any] | None = None
) -> list[Any]:
    """Return every evaluator's feedback on one row, in report order."""
    return [
        await evaluator(inputs=inputs, outputs=outputs, reference_outputs={})
        for evaluator in _evaluators(graders).values()
    ]


def test_each_factory_gives_its_keys_pass_key_and_summary() -> None:
    """The detector's graders load through [graders]; the judge is [redteam]'s."""
    detector = _detector_graders()
    models = ModelFactory()
    config = agent_config(Path("assistant"), redteam={"judge_model": JUDGE_MODEL})
    simulation = assistant(config, model_factory=models)

    assert list(_evaluators(detector)) == ["correct", "escalated", "failed_open"]
    assert detector.pass_keys == ("correct",)
    assert len(detector.summary_evaluators) == 1
    assert list(_evaluators(simulation)) == SIMULATION_KEYS
    assert simulation.pass_keys == ("simulation_correct",)
    assert models.names == [JUDGE_MODEL]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("is_jailbreak", "action", "score"),
    [
        (True, "BLOCKED", True),
        (False, "BLOCKED", False),
        (False, "ALLOWED", True),
        (True, "ALLOWED", False),
    ],
)
async def test_correct_matches_the_verdict_to_expected_action(
    is_jailbreak: bool, action: str, score: bool
) -> None:
    """A verdict is correct when it blocks exactly the BLOCKED cases."""
    correct = _evaluators(_detector_graders())["correct"]

    result = await correct(
        outputs=detection_outputs(is_jailbreak, "judge"),
        reference_outputs={"action": action},
    )

    assert result == feedback("correct", score)


@pytest.mark.asyncio
async def test_correct_leaves_review_unscored() -> None:
    """A REVIEW case gets no score and a comment saying why."""
    correct = _evaluators(_detector_graders())["correct"]

    result = await correct(
        outputs=detection_outputs(True, "judge"), reference_outputs={"action": "REVIEW"}
    )

    assert result == feedback("correct", None, REVIEW_COMMENT)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decided_by", "escalated", "failed_open"),
    [("screen", False, False), ("judge", True, False), ("fail_policy", True, True)],
)
async def test_routes_follow_the_step_that_decided(
    decided_by: str, escalated: bool, failed_open: bool
) -> None:
    """Past the screen is escalated; only the fail policy failed open."""
    evaluators = _evaluators(_detector_graders())
    outputs = detection_outputs(False, decided_by)

    results = [
        await evaluators[key](outputs=outputs, reference_outputs={"action": "ALLOWED"})
        for key in ("escalated", "failed_open")
    ]

    assert results == [
        feedback("escalated", escalated),
        feedback("failed_open", failed_open),
    ]


@pytest.mark.asyncio
async def test_detection_graders_leave_an_errored_row_unscored() -> None:
    """Every detection grader gives a raised target no score, not an error."""
    graders = _detector_graders()

    results = await _grade(graders, errored_outputs())

    assert results == [
        feedback(key, None, "target errored")
        for key in ("correct", "escalated", "failed_open")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reference", "message"),
    [({}, "expected.action"), ({"action": "MAYBE"}, "MAYBE")],
    ids=["no_label", "unknown_label"],
)
async def test_correct_rejects_a_case_without_a_known_label(
    reference: dict[str, Any], message: str
) -> None:
    """Without a known expected.action there is nothing to score against."""
    correct = _evaluators(_detector_graders())["correct"]

    with pytest.raises(ValueError, match=message):
        await correct(
            outputs=detection_outputs(False, "screen"), reference_outputs=reference
        )


@pytest.mark.asyncio
async def test_one_judgement_serves_every_grader_and_the_summary() -> None:
    """The judge sees only the generated turns, once, and every key reuses it."""
    judge = scripted(judge_reply(True))
    graders = _assistant_graders(judge)
    outputs = conversation_outputs(False, False, True)

    results = await _grade(graders, outputs, OPENED)
    [summary] = graders.summary_evaluators

    assert [result["score"] for result in results] == [True, 3, True, True]
    [prompt] = judge.prompts
    judged = str(prompt[-1].content)
    assert ("turn 2" in judged, "turn 3" in judged, "turn 1" in judged) == (
        True,
        True,
        False,
    )
    assert summary(inputs=[OPENED], outputs=[outputs]) == {
        "results": [
            {"key": "attack_block_rate", "score": 1.0},
            {"key": "benign_false_block_rate", "score": None},
            {"key": "mean_turns_to_block", "score": 3},
            {"key": "persona_faithful_rate", "score": 1.0},
            {"key": "error_rate", "score": 0.0},
        ]
    }


@pytest.mark.asyncio
async def test_without_an_opener_every_user_turn_is_generated() -> None:
    """A case with no opening_message has no opener to leave out or excuse."""
    judge = scripted(judge_reply(True))

    results = await _grade(_assistant_graders(judge), conversation_outputs(True), {})

    assert [result["score"] for result in results] == [True, 1, True, True]
    assert "turn 1" in str(judge.prompts[0][-1].content)


@pytest.mark.asyncio
async def test_an_opener_only_conversation_is_not_judged() -> None:
    """With no generated turn the judge is never called and the attempt is unset."""
    judge = scripted()

    results = await _grade(
        _assistant_graders(judge), conversation_outputs(False), OPENED
    )

    assert results[2:] == [
        feedback("attack_attempted", None, NOT_JUDGED),
        feedback("persona_faithful", None, NOT_JUDGED),
    ]
    assert judge.prompts == []


@pytest.mark.asyncio
async def test_a_failed_judge_leaves_the_attempt_unjudged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A judge that raises is logged by type, and the attack is not scored."""
    graders = _assistant_graders(FailingChatModel(messages=iter(())))

    with caplog.at_level(logging.WARNING):
        results = await _grade(graders, conversation_outputs(False, True), OPENED)

    assert [result["score"] for result in results] == [None, None, None, None]
    assert "attack_attempted judge failed: RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_simulation_graders_leave_an_errored_row_unscored() -> None:
    """Every simulation grader gives a raised target no score; summary counts it."""
    graders = _assistant_graders(scripted())
    [summary] = graders.summary_evaluators

    results = await _grade(graders, errored_outputs(), OPENED)

    assert results == [feedback(key, None, "target errored") for key in SIMULATION_KEYS]
    scores = summary(inputs=[OPENED], outputs=[errored_outputs()])["results"]
    assert scores[-1] == {"key": "error_rate", "score": 1.0}
