"""DetectionTarget's prod-shaped detector call and its report of the cascade's route."""

import asyncio
from collections.abc import Sequence

import pytest

from trajectory_evals.suites.jailbreak.detection import DetectionTarget
from trajectory_evals.suites.jailbreak.types import DetectionInputs, DetectionOutputs, HistoryTurn
from src.toolsets.jailbreak_detection._call_log import CallLog, current_call
from src.toolsets.jailbreak_detection.types import (
    ConversationTurn,
    JailbreakDetectionResult,
)

QUIET_ROUTE = CallLog()
QUESTION = "Is the red one available?"
EARLIER = [
    HistoryTurn(initiator="Customer", content="Hi, I want a sedan."),
    HistoryTurn(initiator="Agent", content="Happy to help!"),
]
EARLIER_TURNS = [ConversationTurn.model_validate(turn) for turn in EARLIER]
ASKED = ConversationTurn(initiator="Customer", content=QUESTION)


class FakeDetector:
    """A Detector that returns a scripted result and logs a scripted route."""

    def __init__(self, result: JailbreakDetectionResult, route: CallLog) -> None:
        """Hold the result to return and the call log fields to record."""
        self._result = result
        self._route = route
        self.calls: list[tuple[str | None, list[ConversationTurn]]] = []

    async def check(
        self, text: str | None, history: Sequence[ConversationTurn] | None = None
    ) -> JailbreakDetectionResult:
        """Record the call, log the route from a child task, and return the result."""
        self.calls.append((text, list(history or ())))
        await asyncio.create_task(self._record_route())
        return self._result

    async def _record_route(self) -> None:
        """Write the scripted route into the running call's log, as a detector does."""
        call = current_call()
        call.escalated_by = self._route.escalated_by
        call.history_read = self._route.history_read
        call.judge_reminded = self._route.judge_reminded


def _target(
    is_jailbreak: bool = False, decided_by: str = "screen", route: CallLog = QUIET_ROUTE
) -> DetectionTarget:
    """Build a target around a FakeDetector scripted with this result and route."""
    result = JailbreakDetectionResult(is_jailbreak=is_jailbreak, decided_by=decided_by)
    return DetectionTarget(FakeDetector(result, route))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_jailbreak, decided_by, escalated_by, history_read, judge_reminded",
    [
        (False, "screen", "none", False, False),
        (True, "judge", "screen", True, True),
        (False, "fail_policy", "guardrail", False, False),
    ],
)
async def test_target_reports_the_verdict_and_the_route_logged_during_the_call(
    is_jailbreak: bool,
    decided_by: str,
    escalated_by: str,
    history_read: bool,
    judge_reminded: bool,
) -> None:
    """The output joins the detector's result with what its steps logged."""
    route = CallLog(
        escalated_by=escalated_by,
        history_read=history_read,
        judge_reminded=judge_reminded,
    )
    target = _target(is_jailbreak, decided_by, route)
    outputs = await target(
        DetectionInputs(text="Is it in stock?", history=[], channel="sms")
    )
    assert outputs == DetectionOutputs(
        is_jailbreak=is_jailbreak,
        decided_by=decided_by,
        escalated_by=escalated_by,
        history_read=history_read,
        judge_reminded=judge_reminded,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inputs", "call"),
    [
        (
            DetectionInputs(text=QUESTION, history=[], channel="sms"),
            (QUESTION, [ASKED]),
        ),
        (
            DetectionInputs(text=QUESTION, history=[], channel="email"),
            (None, [ASKED]),
        ),
        (
            DetectionInputs(text=QUESTION, history=[], channel="submission_form"),
            (QUESTION, []),
        ),
        (
            DetectionInputs(text=QUESTION, history=EARLIER, channel="sms"),
            (QUESTION, [*EARLIER_TURNS, ASKED]),
        ),
        (
            DetectionInputs(text=QUESTION, history=EARLIER, channel="submission_form"),
            (QUESTION, [*EARLIER_TURNS, ASKED]),
        ),
        (
            DetectionInputs(
                text=None,
                history=[*EARLIER, HistoryTurn(initiator="Customer", content=QUESTION)],
                channel="sms",
            ),
            (QUESTION, [*EARLIER_TURNS, ASKED]),
        ),
    ],
    ids=[
        "sms_row",
        "email_row",
        "submission_form_row",
        "turns_row",
        "submission_form_later_turn",
        "history_row_without_text",
    ],
)
async def test_target_calls_the_detector_as_prod_does(
    inputs: DetectionInputs, call: tuple[str | None, list[ConversationTurn]]
) -> None:
    """Email omits the text, a form opener the history; else text plus all turns."""
    detector = FakeDetector(
        JailbreakDetectionResult(is_jailbreak=False, decided_by="screen"), QUIET_ROUTE
    )
    await DetectionTarget(detector)(inputs)
    assert detector.calls == [call]


@pytest.mark.asyncio
async def test_concurrent_cases_each_report_their_own_route() -> None:
    """Cases aevaluate runs at once never see each other's call log."""
    escalated = _target(True, "judge", CallLog(escalated_by="guardrail"))
    quiet = _target()
    inputs = DetectionInputs(text="Hello", history=[], channel="sms")
    escalated_outputs, quiet_outputs = await asyncio.gather(
        escalated(inputs), quiet(inputs)
    )
    assert (escalated_outputs["escalated_by"], quiet_outputs["escalated_by"]) == (
        "guardrail",
        "none",
    )
