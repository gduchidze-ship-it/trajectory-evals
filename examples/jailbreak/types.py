"""Typed shapes the jailbreak responder, graders and converter share."""

from typing import Literal, TypedDict

ExpectedAction = Literal["ALLOWED", "BLOCKED", "REVIEW"]


class DetectionResult(TypedDict):
    """The detect_jailbreak tool result the responder reports on every turn.

    escalated_by names the detector that escalated first, or "none", as the
    detector's CallLog recorded it; failed_open is decided_by == "fail_policy".
    history_read and judge_reminded are the CallLog's diagnostics: whether the
    screen read the history and whether the judge needed its reminder. No
    grader scores them.
    """

    is_jailbreak: bool
    decided_by: str
    escalated_by: str
    failed_open: bool
    history_read: bool
    judge_reminded: bool


__all__ = ["DetectionResult", "ExpectedAction"]
